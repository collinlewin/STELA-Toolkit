from copy import deepcopy
import ast
import pickle
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from ._check_inputs import _CheckInputs
from .preprocessing import Preprocessing


class GaussianProcess:
    """
    Fit and sample from a Gaussian Process (GP) model for light curve interpolation.

    This class models light curve data as stochastic processes governed by a Gaussian Process prior.
    Given a LightCurve object with time, rate, and, optionally, errors, the GP learns the, 
    covariance structure between all the data points in the set, including measurement uncertainty
    and the underlying, empirical variability.

    Key preprocessing and modeling features:
    
    - If the flux distribution is non-Gaussian, an optional Box-Cox transformation can be applied.
    - Data is standardized (zero mean, unit variance) prior to training for numerical stability.
    - You can model measurement uncertainties directly or optionally include a learned white noise component.
    - If you don’t specify a kernel, the model will automatically try several and select the best one using AIC.

    Once trained, the model can generate posterior samples from the predictive distribution—
    these are realistic realizations of what the light curve *could* look like, given your data and uncertainties.
    These samples are used downstream in STELA for computing frequency-domain products like power spectra,
    coherence, cross-spectra, and lags.

    Kernel selection is highly flexible:

    - You can pass a simple string like 'RBF', 'Matern32', or 'SpectralMixture, 6'
    - Or define arbitrary compositions using + and * operators, e.g.:
        - 'RBF + Periodic * RQ'
        - '(Matern32 + Periodic) * RQ'
    - Composite kernels are parsed using Python syntax and safely evaluated into GPyTorch objects.

    Noise handling:

    - If your light curve includes error bars, they are treated as fixed noise.
    - If not, or if you want to include extra variability, you can learn a white noise term.

    Training is performed using exact inference via GPyTorch and gradient descent.
    You can configure the number of optimization steps, learning rate, and whether to visualize training loss.

    Parameters
    ----------
    lightcurve : LightCurve
        The input light curve to model.

    kernel_form : str or list, optional
        Kernel expression or list of candidate kernel names.
        Examples include:
        - 'Matern32'
        - 'SpectralMixture, 4'
        - '(Periodic + RBF) * RQ'
        - If 'auto', the model tries several standard kernels and selects the best using AIC.

    white_noise : bool, optional
        Whether to include a learned white noise component in addition to measurement errors.

    enforce_normality : bool, optional
        Whether to apply a Box-Cox transformation to make the flux distribution more Gaussian.

    run_training : bool, optional
        Whether to train the GP model immediately upon initialization.

    plot_training : bool, optional
        Whether to plot the training loss as optimization progresses.

    num_iter : int, optional
        Number of training iterations for gradient descent.

    learn_rate : float, optional
        Learning rate for the optimizer.

    sample_time_grid : array-like, optional
        Time grid on which to generate posterior samples after training.

    num_samples : int, optional
        Number of posterior samples to draw from the trained GP.

    verbose : bool, optional
        Whether to print diagnostic information about training, sampling, and kernel selection.

    Attributes
    ----------
    model : gpytorch.models.ExactGP
        The trained GP model used for inference and sampling.

    likelihood : gpytorch.likelihoods.Likelihood
        The likelihood object (fixed or learnable noise) used during training.

    train_times : torch.Tensor
        The training time grid (from the input light curve).

    train_rates : torch.Tensor
        The training rate values (preprocessed and standardized).

    train_errors : torch.Tensor
        The measurement error bars (or empty if not provided).

    samples : ndarray
        Posterior GP samples drawn after training (used in downstream STELA modules).

    pred_times : torch.Tensor
        The time grid over which posterior samples were drawn.

    kernel_form : str
        The user-provided or auto-selected kernel expression used in the final model.
    """

    def __init__(self,
                 lightcurve,
                 kernel_form='auto',
                 white_noise=True,
                 enforce_normality=False,
                 run_training=True,
                 plot_training=False,
                 num_iter=500,
                 learn_rate=1e-1,
                 sample_time_grid=[],
                 num_samples=1000,
                 verbose=False):

        # To Do: reconsider noise prior, add a mean function function for forecasting, more verbose options
        _CheckInputs._check_input_data(lightcurve, req_reg_samp=False)
        self.lc = deepcopy(lightcurve)
        
        # Save original mean, std, boxcox parameter for reversing standardization
        self.lc_mean = getattr(self.lc, 'unstandard_mean', np.mean(self.lc.rates))
        self.lc_std = getattr(self.lc, 'unstandard_std', np.std(self.lc.rates))
        self.lambda_boxcox = getattr(self.lc, "lambda_boxcox", None)

        # Check normality and apply boxcox if user specifies
        if enforce_normality:
            self.enforce_normality()

        # Standardize data
        if not getattr(self.lc, "is_standard", False):
            Preprocessing.standardize(self.lc)

        # Convert light curve data to pytorch tensors
        self.train_times = torch.tensor(self.lc.times, dtype=torch.float32)
        self.train_rates = torch.tensor(self.lc.rates, dtype=torch.float32)
        if self.lc.errors.size > 0:
            self.train_errors = torch.tensor(self.lc.errors, dtype=torch.float32)
        else:
            self.train_errors = torch.tensor([])

        # Training
        self.white_noise = white_noise
        if kernel_form == 'auto' or isinstance(kernel_form, list):
            # Automatically select the best kernel based on AIC
            if isinstance(kernel_form, list):
                kernel_list = kernel_form
            else:
                kernel_list = ['Matern12', 'Matern32',
                               'Matern52', 'RQ', 'RBF', 'SpectralMixture, 4']

            best_model, best_likelihood = self.find_best_kernel(
                kernel_list, num_iter=num_iter, learn_rate=learn_rate, verbose=verbose
            )
            self.model = best_model
            self.likelihood = best_likelihood
        else:
            # Use specified kernel
            self.likelihood = self.set_likelihood(self.white_noise, train_errors=self.train_errors)
            self.model = self.create_gp_model(self.likelihood, kernel_form)

            # Separate training needed only if kernel not automatically selected
            if run_training:
                self.train(num_iter=num_iter, learn_rate=learn_rate, plot=plot_training, verbose=verbose)

        # Generate samples if sample_time_grid is provided
        if sample_time_grid:
            self.samples = self.sample(sample_time_grid, num_samples=num_samples)
            if verbose:
                print(f"Samples generated: {self.samples.shape}, access with 'samples' attribute.")

        # Unstandardize the data
        Preprocessing.unstandardize(self.lc)

        # Undo boxcox transformation if needed
        if getattr(self.lc, "is_boxcox_transformed", False):
            Preprocessing.reverse_boxcox_transform(self.lc)

    def enforce_normality(self):
        """
        Check normality of the input data and apply a Box-Cox transformation if needed.

        This method first checks if the light curve's flux distribution appears normal.
        If not, a Box-Cox transformation is applied to improve it. STELA automatically
        selects the most appropriate test (Shapiro-Wilk or Lilliefors) based on sample size.
        """
        print("Checking normality of input light curve...")

        is_normal_before, pval_before = Preprocessing.check_normal(self.lc, plot=False, verbose=False)

        if is_normal_before:
            print(f"\n - Light curve appears normal (p = {pval_before:.4f}). No transformation applied.")
            return

        print(f"\n - Light curve is not normal (p = {pval_before:.4f}). Applying Box-Cox transformation...")

        # Apply Box-Cox transformation
        Preprocessing.boxcox_transform(self.lc)

        if self.lambda_boxcox is not None:
            print(" -- Note: The input was already Box-Cox transformed. No additional transformation made.")
        else:
            self.lambda_boxcox = getattr(self.lc, "lambda_boxcox", None)

        # Re-check normality
        is_normal_after, pval_after = Preprocessing.check_normal(self.lc, plot=False, verbose=False)

        if is_normal_after:
            print(f" - Normality sufficiently achieved after Box-Cox (p = {pval_after:.4f})! Proceed as normal!\n")
        else:
            print(f" - Data still not normal after Box-Cox (p = {pval_after:.4f}). Proceed with caution.\n")


    def create_gp_model(self, likelihood, kernel_form):
        """
        Build a GP model with the specified likelihood and kernel.

        Parameters
        ----------
        likelihood : gpytorch.likelihoods.Likelihood
            The likelihood model to use (e.g., Gaussian or FixedNoise).
        
        kernel_form : str
            The kernel type (e.g., 'Matern32', 'SpectralMixture, 4').

        Returns
        -------
        GPModel
            A subclass of gpytorch.models.ExactGP for training.
        """

        class GPModel(gpytorch.models.ExactGP):
            def __init__(gp_self, train_times, train_rates, likelihood):
                super(GPModel, gp_self).__init__(train_times, train_rates, likelihood)
                gp_self.mean_module = gpytorch.means.ZeroMean()
                gp_self.covar_module = self.set_kernel(kernel_form)

            def forward(gp_self, x):
                mean_x = gp_self.mean_module(x)
                covar_x = gp_self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        return GPModel(self.train_times, self.train_rates, likelihood)

    def set_likelihood(self, white_noise, train_errors=torch.tensor([])):
        """
        Set up the GP likelihood model based on user input and data characteristics.

        If error bars are available, uses a FixedNoiseGaussianLikelihood. Otherwise, defaults to a
        GaussianLikelihood with optional white noise. If white noise is enabled, the noise level
        is initialized based on Poisson statistics or variance in the data.

        Parameters
        ----------
        white_noise : bool
            Whether to include a learnable noise term in the model.
        
        train_errors : torch.Tensor, optional
            Measurement errors from the light curve.

        Returns
        -------
        likelihood : gpytorch.likelihoods.Likelihood
            GPyTorch subclass, also used for training.
        """

        if white_noise:
            noise_constraint = gpytorch.constraints.Interval(1e-5, 1)
        else:
            noise_constraint = gpytorch.constraints.Interval(1e-40, 1e-39)

        if train_errors.size(dim=0) > 0:
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=self.train_errors ** 2,
                learn_additional_noise=white_noise,
                noise_constraint=noise_constraint
            )

        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=noise_constraint,
            )

            if white_noise:
                counts = np.abs(self.train_rates[1:].numpy()) * np.diff(self.train_times.numpy())
                # begin with a slight underestimation to prevent overfitting
                norm_poisson_var = 1 / (2 * np.mean(counts))
                likelihood.noise = norm_poisson_var

        # initialize noise parameter at the variance of the data
        return likelihood

    def set_kernel(self, kernel_expr):
        """
        Set the GP kernel (covariance function) using a simple kernel form string, or a composite expression.
        Compositions are created by using '+', '*', and parentheses. Also handles 'SpectralMixture, N'.

        All kernels are wrapped in a ScaleKernel at the end, multiplying the overall covariance by a constant.
        Parameters
        ----------
        kernel_expr : str
            A string expression like 'RBF + Periodic * RQ' or 'SpectralMixture, 6'.

        Returns
        -------
        covar_module : gpytorch.kernels.Kernel
            Final kernel.
        """

        kernel_expr = kernel_expr.strip()

        # Handle SpectralMixture, N syntax
        if 'SpectralMixture' in kernel_expr:
            if ',' not in kernel_expr:
                raise ValueError("Use 'SpectralMixture, N' to specify number of components.")
            base, n = kernel_expr.split(',')
            kernel_expr = 'SpectralMixture'
            num_mixtures = int(n.strip())
        else:
            num_mixtures = 4

        kernel_mapping = {
            'Matern12': gpytorch.kernels.MaternKernel(nu=0.5),
            'Matern32': gpytorch.kernels.MaternKernel(nu=1.5),
            'Matern52': gpytorch.kernels.MaternKernel(nu=2.5),
            'RQ': gpytorch.kernels.RQKernel(),
            'RBF': gpytorch.kernels.RBFKernel(),
            'Periodic': gpytorch.kernels.PeriodicKernel(),
            'SpectralMixture': gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures),
        }

        # Parse and evaluate the expression
        expr_ast = ast.parse(kernel_expr, mode='eval')
        kernel_obj = self._eval_kernel_ast(expr_ast.body, kernel_mapping)

        # Initialize SpectralMixture if used
        if 'SpectralMixture' in kernel_expr:
            kernel_obj.initialize_from_data(self.train_times, self.train_rates)

        else:
            # Set initial lengthscale for base kernels
            init_lengthscale = (self.train_times[-1] - self.train_times[0]) / 10
            for name, kernel in kernel_mapping.items():
                if getattr(kernel, 'has_lengthscale', False):
                    kernel.lengthscale = init_lengthscale

        self.kernel_form = kernel_expr
        return gpytorch.kernels.ScaleKernel(kernel_obj)

    def train(self, num_iter=500, learn_rate=1e-1, plot=False, verbose=False):
        """
        Train the GP model using the Adam optimizer to minimize the negative log marginal likelihood (NLML).

        By default, prints progress periodically and optionally plots the NLML loss curve over training iterations.
        This function is typically called after initialization unless `run_training=True` was set earlier.

        Parameters
        ----------
        num_iter : int, optional
            Number of optimization steps to perform. Default is 500.
        
        learn_rate : float, optional
            Learning rate for the Adam optimizer. Default is 0.1.
        
        plot : bool, optional
            If True, display a plot of the NLML loss as training progresses.
        
        verbose : bool, optional
            If True, print progress updates at regular intervals during training.
        """

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        print_every = max(1, num_iter // 20)

        if plot:
            plt.figure(figsize=(8, 5))

        for i in range(num_iter):
            optimizer.zero_grad()
            output = self.model(self.train_times)
            loss = -mll(output, self.train_rates)
            loss.backward()

            if verbose and (i == num_iter - 1 or i % print_every == 0):
                if self.white_noise:
                    if self.train_errors.size(dim=0) > 0:
                        noise_param = self.model.likelihood.second_noise.item()
                    else:
                        noise_param = self.model.likelihood.noise.item()

                if self.kernel_form == 'SpectralMixture':
                    mixture_scales = self.model.covar_module.base_kernel.mixture_scales
                    mixture_scales = mixture_scales.detach().numpy().flatten()
                    mixture_weights = self.model.covar_module.base_kernel.mixture_weights
                    mixture_weights = mixture_weights.detach().numpy().flatten()

                    if self.white_noise:
                        print('Iter %d/%d - loss: %.3f   mixture_lengthscales: %s   mixture_weights: %s   noise: %.1e' % (
                            i + 1, num_iter, loss.item(),
                            mixture_scales.round(3),
                            mixture_weights.round(3),
                            noise_param
                        ))
                    else:
                        print('Iter %d/%d - loss: %.3f   mixture_lengthscales: %s   mixture_weights: %s' % (
                            i + 1, num_iter, loss.item(),
                            mixture_scales.round(3),
                            mixture_weights.round(3)
                        ))

                elif self.kernel_form == 'Periodic':
                    if self.white_noise:
                        print('Iter %d/%d - loss: %.3f   period length: %.3f   lengthscale: %.3f   noise: %.1e' % (
                            i + 1, num_iter, loss.item(),
                            self.model.covar_module.base_kernel.period_length.item(),
                            self.model.covar_module.base_kernel.lengthscale.item(),
                            noise_param
                        ))
                    else:
                        print('Iter %d/%d - loss: %.3f   lengthscale: %.1e' % (
                            i + 1, num_iter, loss.item(),
                            self.model.covar_module.base_kernel.lengthscale.item()
                        ))

                else:
                    if self.white_noise:
                        print('Iter %d/%d - loss: %.3f   lengthscale: %.3f   noise: %.1e' % (
                            i + 1, num_iter, loss.item(),
                            self.model.covar_module.base_kernel.lengthscale.item(),
                            noise_param
                        ))
                    else:
                        print('Iter %d/%d - loss: %.3f   lengthscale: %.1e' % (
                            i + 1, num_iter, loss.item(),
                            self.model.covar_module.base_kernel.lengthscale.item()
                        ))

            optimizer.step()

            if plot:
                plt.scatter(i, loss.item(), color='black', s=2)

        if verbose:
            final_hypers = self.get_hyperparameters()
            print(
                "Training complete. \n"
                f"   - Final loss: {loss.item():0.5}\n"
                f"   - Final hyperparameters:")
            for key, value in final_hypers.items():
                print(f"      {key:42}: {np.round(value, 4)}")

        if plot:
            plt.xlabel('Iteration')
            plt.ylabel('Negative Marginal Log Likelihood')
            plt.title('Training Progress')
            plt.show()

    def find_best_kernel(self, kernel_list, num_iter=500, learn_rate=1e-1, verbose=False):
        """
        Search over a list of kernels and return the best one by AIC.

        Trains the model separately with each kernel in the list, computes the AIC,
        and returns the model with the lowest value.

        Parameters
        ----------
        kernel_list : list of str
            Kernel names to try.
        
        num_iter : int
            Number of iterations per training run.
        
        learn_rate : float
            Learning rate for the optimizer.
        
        verbose : bool
            Whether to print progress for each kernel.

        Returns
        -------
        best_model : GPModel
            The model trained with the best-performing kernel.
        
        best_likelihood : gpytorch.likelihoods.Likelihood
            Corresponding likelihood for the best model.
        """

        aics = []
        best_model = None
        for kernel_form in kernel_list:
            self.likelihood = self.set_likelihood(self.white_noise, train_errors=self.train_errors)
            self.model = self.create_gp_model(self.likelihood, kernel_form)
            # suppress output, even for verbose=True
            self.train(num_iter=num_iter, learn_rate=learn_rate, verbose=False)

            # compute aic and store best model
            aic = self.aic()
            aics.append(aic)
            if aic <= min(aics):
                best_model = self.model
                best_likelihood = self.likelihood

        best_aic = min(aics)
        best_kernel = kernel_list[aics.index(best_aic)]

        if verbose:
            kernel_results = zip(kernel_list, aics)
            print(
                "Kernel selection complete.\n"
                f"   Kernel AICs (lower is better):"
            )
            for kernel, aic in kernel_results:
                print(f"     - {kernel:15}: {aic:0.5}")

            print(f"   Best kernel: {best_kernel} (AIC: {best_aic:0.5})")

        self.kernel_form = best_kernel
        return best_model, best_likelihood

    def get_hyperparameters(self):
        """
        Return the learned GP hyperparameters (lengthscale, noise, weights, etc.).

        Returns
        -------
        hyper_dict : dict
            Dictionary mapping parameter names to their (transformed) values.
                Note: All rate-associated hyperparameters (e.g., not lengthscale) 
                are in units of the standardized data, not the original flux/time units.
        """

        raw_hypers = self.model.named_parameters()
        hypers = {}
        for param_name, param in raw_hypers:
            # Split the parameter name into hierarchy
            parts = param_name.split('.')
            module = self.model

            # Traverse structure of the model to get the constraint
            for part in parts[:-1]:  # last part is parameter
                module = getattr(module, part, None)
                if module is None:
                    raise AttributeError(
                        f"Module '{part}' not found while traversing '{param_name}'.")

            final_param_name = parts[-1]
            constraint_name = f"{final_param_name}_constraint"
            constraint = getattr(module, constraint_name, None)

            if constraint is None:
                raise AttributeError(
                    f"Constraint '{constraint_name}' not found in module '{module}'.")

            # Transform the parameter using the constraint
            transform_param = constraint.transform(param)

            # Remove 'raw_' prefix from the parameter name for readability
            param_name_withoutraw = param_name.replace('raw_', '')

            if self.kernel_form == 'SpectralMixture':
                transform_param = transform_param.detach().numpy().flatten()
            else:
                transform_param = transform_param.item()

            hypers[param_name_withoutraw] = transform_param

        return hypers

    def bic(self):
        """
        Compute the Bayesian Information Criterion (BIC) for the trained model.

        Returns
        -------
        bic : float
            The BIC value (lower is better).
        """

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        log_marg_like = mll(
            self.model(self.train_times), self.train_rates
        ).item()

        num_params = sum([p.numel() for p in self.model.parameters()])
        num_data = len(self.train_times)

        bic = -2 * log_marg_like + num_params * np.log(num_data)
        return bic

    def aic(self):
        """
        Compute the Akaike Information Criterion (AIC) for the trained model.

        Returns
        -------
        aic : float
            The AIC value (lower is better).
        """

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        log_marg_like = mll(
            self.model(self.train_times), self.train_rates
        ).item()

        num_params = sum([p.numel() for p in self.model.parameters()])

        aic = -2 * log_marg_like + 2 * num_params
        return aic

    def sample(self, pred_times, num_samples, save_path=None, _save_to_state=True):
        """
        Generate posterior samples from the trained GP model.

        These samples represent plausible realizations of the light curve. These are what is used
        by the coherence, power spectrum, and lag modules when a GP model is passed in.

        Parameters
        ----------
        pred_times : array-like
            Time points where samples should be drawn.
        
        num_samples : int
            Number of realizations to generate.
        
        save_path : str, optional
            File path to save the samples.
        
        _save_to_state : bool, optional
            Whether to store results in the object (used by other classes).

        Returns
        -------
        samples : ndarray
            Array of sampled light curves with shape (num_samples, len(pred_times)).
        """

        pred_times_tensor = torch.tensor(pred_times, dtype=torch.float32)
        self.model.eval()
        self.likelihood.eval()

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = self.likelihood(self.model(pred_times_tensor))
            post_samples = pred_dist.sample(sample_shape=torch.Size([num_samples]))

        samples = post_samples.numpy()
        samples = self._undo_transforms(samples)

        if save_path:
            samples_with_time = np.insert(pred_times, num_samples, 0)
            file_ext = save_path.split(".")[-1]
            if file_ext == "npy":
                np.save(save_path, samples_with_time)
            else:
                np.savetxt(save_path, samples_with_time)

        if _save_to_state:
            self.pred_times = pred_times
            self.samples = samples
        return samples

    def predict(self, pred_times):
        """
        Compute the posterior mean and 2-sigma confidence intervals at specified times.

        Parameters
        ----------
        pred_times : array-like
            Time values to predict.

        Returns
        -------
        mean, lower, upper : ndarray
            Predicted mean and lower/upper bounds of the 95 percent confidence interval.
        """

        # Check if pred_times is a torch tensor
        if not isinstance(pred_times, torch.Tensor):
            try:
                pred_times = torch.tensor(pred_times, dtype=torch.float32)
            except TypeError:
                raise TypeError("pred_times must be a torch tensor or convertible to one.")

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = self.likelihood(self.model(pred_times))
            mean = pred_dist.mean
            lower, upper = pred_dist.confidence_region()

        # Unstandardize/unboxcox
        mean = self._undo_transforms(mean)
        lower = self._undo_transforms(lower)
        upper = self._undo_transforms(upper)
        return mean.numpy(), lower.numpy(), upper.numpy()

    def plot(self, pred_times=None):
        """
        Plot the GP fit including mean, confidence intervals, one posterior sample, and data.

        Parameters
        ----------
        pred_times : array_like, optional
            Time grid to show prediction, samples. If not specificed, a grid of 1000 points will be automatically used.
        """

        if pred_times is None:
            step = (self.train_times.max() - self.train_times.min()) / 1000
            pred_times = np.arange(self.train_times.min(), self.train_times.max() + step, step)

        predict_mean, predict_lower, predict_upper = self.predict(pred_times)

        plt.figure(figsize=(8, 4.5))

        plt.fill_between(pred_times, predict_lower, predict_upper,
                         color='dodgerblue', alpha=0.2, label=r'Prediction 2$\sigma$ CI')
        plt.plot(pred_times, predict_mean, color='dodgerblue', label='Prediction Mean')

        sample = self.sample(pred_times, num_samples=1, _save_to_state=False)
        plt.plot(pred_times, sample[0], color='orange', lw=1, label='Sample')

        if self.train_errors.size(dim=0) > 0:
            plt.errorbar(self.lc.times, self.lc.rates, yerr=self.lc.errors,
                         fmt='o', color='black', lw=1.5, ms=3)
        else:
            plt.scatter(self.lc.times, self.lc.rates, color='black', s=6)

        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Rate', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tick_params(which='both', direction='in', length=6, width=1,
                        top=True, right=True, labelsize=12)
        plt.show()

    def save(self, file_path):
        """
        Save the trained GP model to a file using pickle.

        Parameters
        ----------
        file_path : str
            Path to save the model.
        """

        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        print(f"GaussianProcess instance saved to {file_path}.")

    @staticmethod
    def load(file_path):
        """
        Load a saved GaussianProcess model from file.

        Parameters
        ----------
        file_path : str
            Path to the saved file.

        Returns
        -------
        GaussianProcess
            Restored instance of the model.
        """

        with open(file_path, "rb") as f:
            instance = pickle.load(f)
        print(f"GaussianProcess instance loaded from {file_path}.")
        return instance
    
    def _undo_transforms(self, array):
        """
        Reverse Box-Cox and standardization transformations applied to GP outputs.

        Parameters
        ----------
        array : ndarray
            Input values in transformed space.

        Returns
        -------
        array : ndarray
            Values in original flux units.
        """

        if self.lambda_boxcox is not None:
            if self.lambda_boxcox == 0:
                array = np.exp(array)
            else:
                array = (array * self.lambda_boxcox + 1) ** (1 / self.lambda_boxcox)

        array = array * self.lc_std + self.lc_mean
        return array
    
    def _eval_kernel_ast(self, node, kernel_mapping):
        """
        Recursively to build the composite kernels using an abstract tree.

        Parameters
        ----------
        node : ast.AST
            A node in the tree.
        
        kernel_mapping : dict
            Maps kernel names (e.g., 'RBF') to GPyTorch kernel objects.

        Returns
        -------
        gpytorch.kernels.Kernel
            The composed kernel object.
        """
        # Check if node is a binary operation (+ or *)
        if isinstance(node, ast.BinOp):
            # Traverse
            left = self._eval_kernel_ast(node.left, kernel_mapping)
            right = self._eval_kernel_ast(node.right, kernel_mapping)

            # Perform addition or multiplication on the kernel objects
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Mult):
                return left * right
            # Tell users that we only support * and +
            else:
                raise ValueError(f"Unsupported operator: {ast.dump(node.op)}")

        # Otherwise check if node is just the kernel name
        elif isinstance(node, ast.Name):
            if node.id not in kernel_mapping:
                raise ValueError(f"Unknown kernel: {node.id}")
            return kernel_mapping[node.id]

        # If the node is a full expression (not just a name), evaluate the contained 
        # expression recursively
        elif isinstance(node, ast.Expr):
            return self._eval_kernel_ast(node.value, kernel_mapping)

        # Unsupported something found
        else:
            raise ValueError(f"Unsupported expression: {ast.dump(node)}")