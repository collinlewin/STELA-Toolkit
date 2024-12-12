import torch
import gpytorch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from ._check_inputs import _CheckInputs
from .preprocessing import Preprocessing


class GaussianProcess:
    """
    Models a light curve using a Gaussian Process (GP).

    This class fits a GP model to light curve data using specified kernels and likelihoods. 
    It supports kernel selection, training, prediction, sampling, and model evaluation using 
    criteria such as AIC and BIC.

    Parameters:
    - lightcurve (LightCurve): A `LightCurve` object containing the data.
    - kernel_form (str or list, optional): Kernel type or list of types for GP. Defaults to 'auto'.
    - white_noise (bool, optional): Whether to include a white noise component. Defaults to True.
    - run_training (bool, optional): Whether to train the model after initialization. Defaults to True.
    - num_iter (int, optional): Number of training iterations. Defaults to 1000.
    - learn_rate (float, optional): Learning rate for training. Defaults to 1e-1.
    - sample_time_grid (array-like, optional): Time grid for sampling from the GP. Defaults to [].
    - num_samples (int, optional): Number of samples to draw from the GP posterior. Defaults to 1000.
    - plot_gp (bool, optional): Whether to plot the GP predictions and samples. Defaults to False.
    - verbose (bool, optional): Whether to display detailed output. Defaults to True.

    Attributes:
    - model (gpytorch.models.ExactGP): The trained GP model.
    - likelihood (gpytorch.likelihoods.Likelihood): The likelihood associated with the GP model.
    - train_times (torch.Tensor): Training time points.
    - train_rates (torch.Tensor): Training data rates.
    - train_errors (torch.Tensor): Errors for the training data rates (if provided).
    - samples (torch.Tensor): Samples from the GP posterior (if generated).
    """
    def __init__(self,
                 lightcurve,
                 kernel_form='auto',
                 white_noise=True,
                 run_training=True,
                 plot_training=False,
                 num_iter=1000,
                 learn_rate=1e-1,
                 sample_time_grid=[],
                 num_samples=1000,
                 plot_gp=False,
                 verbose=True):

        # To Do: reconsider noise prior, add a mean function function for forecasting, more verbose options
        try:
            _CheckInputs._check_input_data(lightcurve, req_reg_samp=False)
        except ValueError as e:
            raise ValueError(f"Invalid LightCurve object: {e}")
        
        self.lightcurve = lightcurve

        # Standardize the light curve data to match zero mean function
        Preprocessing.standardize(self.lightcurve)

        # Convert light curve data to PyTorch tensors
        self.train_times = torch.tensor(self.lightcurve.times).float()
        self.train_rates = torch.tensor(self.lightcurve.rates).float()
        if self.lightcurve.errors.size > 0:
            self.train_errors = torch.tensor(self.lightcurve.errors).float()
        else:
            self.train_errors = torch.tensor([])

        # Training
        self.white_noise = white_noise
        self.plot_training = plot_training
        if kernel_form == 'auto' or isinstance(kernel_form, list):
            # Automatically select the best kernel based on AIC
            if isinstance(kernel_form, list):
                kernel_list = kernel_form
            else:
                kernel_list = ['Matern12', 'Matern32', 'Matern52', 'RQ', 'RBF', 'SpectralMixture, 4']

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
                self.train_model(num_iter=num_iter, learn_rate=learn_rate, verbose=verbose)

        # Generate samples if sample_time_grid is provided
        if sample_time_grid:
            self.samples = self.sample(sample_time_grid, num_samples=num_samples)
            if verbose:
                print(f"Samples generated: {self.samples.shape}, access with 'samples' attribute.")

        # unstandardize the data
        Preprocessing.unstandardize(self.lightcurve)

        if plot_gp:
            self.plot(sample_time_grid)

    def create_gp_model(self, likelihood, kernel_form):
        """
        Creates and returns a Gaussian Process model.

        Parameters:
        - train_times (torch.Tensor): Training time points.
        - train_rates (torch.Tensor): Training data rates.
        - likelihood (gpytorch.likelihoods.Likelihood): Likelihood function for the GP.
        - kernel (str): Kernel type to use for the GP.

        Returns:
        - GPModel: A GP model with the specified kernel and likelihood.
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
        Configures the likelihood for the GP model.

        Parameters:
        - white_noise (bool): Whether to include a white noise component.
        - train_errors (torch.Tensor, optional): Errors for the training data.

        Returns:
        - gpytorch.likelihoods.Likelihood: The configured likelihood.
        """
        if white_noise:
            noise_constraint = gpytorch.constraints.Interval(1e-3, 1e2)
        else:
            noise_constraint = gpytorch.constraints.Interval(1e-40, 1e-39)

        if train_errors.size(dim=0) > 0:
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=self.train_errors ** 2,
                learn_additional_noise=white_noise,
                noise_constraint = noise_constraint
            )

        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint = noise_constraint,
            )

            if white_noise:
                counts = np.abs(self.train_rates[1:].numpy()) * np.diff(self.train_times.numpy())
                norm_poisson_var = 1 / (2 * np.mean(counts)) # begin with a slight underestimation to prevent overfitting
                likelihood.noise = norm_poisson_var

        # initialize noise parameter at the variance of the data
        return likelihood

    def set_kernel(self, kernel_form):
        """
        Configures the covariance kernel for the GP model.

        Parameters:
        - kernel_form (str): Kernel type (e.g., 'Matern32', 'RBF', 'SpectralMixture, N').

        Returns:
        - gpytorch.kernels.Kernel: The configured kernel.
        """
        kernel_form = kernel_form.strip()
        if 'SpectralMixture' in kernel_form:
            if ',' not in kernel_form:
                raise ValueError(
                    "Invalid Spectral Mixture kernel format (use 'SpectralMixture, N').\n"
                    "N=4 is a good starting point."
                )
            else:
                kernel_form, num_mixtures_str = kernel_form.split(',')
                num_mixtures = int(num_mixtures_str.strip())
        else:
            num_mixtures = 4  # set num_mixtures for kernel_mapping when Spectral Mixture kernel not used

        kernel_mapping = {
            'Matern12': gpytorch.kernels.MaternKernel(nu=0.5), 
            'Matern32': gpytorch.kernels.MaternKernel(nu=1.5),
            'Matern52': gpytorch.kernels.MaternKernel(nu=2.5),
            'RQ': gpytorch.kernels.RQKernel(),
            'RBF': gpytorch.kernels.RBFKernel(),
            'SpectralMixture': gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures)
        }

        # Assign kernel if type is valid
        if kernel_form in kernel_mapping:
            kernel = kernel_mapping[kernel_form]
        else:
            raise ValueError(
                f"Invalid kernel functional form '{kernel_form}'. Choose from {list(kernel_mapping.keys())}.")
        
        if kernel_form == 'SpectralMixture':
            kernel.initialize_from_data(self.train_times, self.train_rates)
        else:
            # initialize at 1/10th of full lc length
            init_lengthscale = (self.train_times[-1] - self.train_times[0]) / 10
            kernel.lengthscale = init_lengthscale

        # Scale the kernel by a constant factor
        covar_module = gpytorch.kernels.ScaleKernel(kernel)
        self.kernel_form = kernel_form

        return covar_module

    def train_model(self, num_iter=1000, learn_rate=1e-1, verbose=False):
        """
        Trains the Gaussian Process model, hyperparameters optimized 
        using Adam optimizer given number of training iterations and 
        learning rate.

        If verbose is True, prints the training progress and final hyperparameters.
        If hyperparameters not converged...
            - monotonic progression? --> consider increasing num_iter.
            - fluctuating around potential optimum? --> consider decreasing learn_rate.

        Parameters:
        - num_iter (int): Number of training iterations.
        - learn_rate (float): Learning rate for the optimizer.
        - verbose (bool): Whether to print detailed information during training.
        """
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        if self.plot_training:
            plt.figure(figsize=(8, 5))

        for i in range(num_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()

            # Model output
            output = self.model(self.train_times)

            # Calc negative likelihood and backprop gradients
            loss = -mll(output, self.train_rates)
            loss.backward()

            if verbose:
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
            
            if self.plot_training:
                plt.scatter(i, loss.item(), color='black', s=2)

        if verbose:
            final_hypers = self.get_hyperparameters()
            print(
                "Training complete. \n"
                f"   - Final loss: {loss.item():0.5}\n"
                f"   - Final hyperparameters:"
            )
            for key, value in final_hypers.items():
                print(f"      {key:42}: {np.round(value, 4)}")

        if self.plot_training:
            plt.xlabel('Iteration')
            plt.ylabel('Negative Marginal Log Likelihood')
            plt.title('Training Progress')
            plt.show()

    def find_best_kernel(self, kernel_list, num_iter=1000, learn_rate=1e-1, verbose=False):
        """
        Finds the best kernel based on the Akaike Information Criterion (AIC).

        Iteratively trains the GP model using different kernels from the provided list
        and selects the kernel with the lowest AIC.

        Parameters:
        - kernel_list (list): List of kernel types to evaluate.
        - num_iter (int): Number of training iterations for each kernel.
        - learn_rate (float): Learning rate for the optimizer.
        - verbose (bool): Whether to print details about the kernel selection process.

        Returns:
        - best_model (GPModel): The trained model with the best kernel.
        - best_likelihood (gpytorch.likelihoods.Likelihood): The likelihood associated with the best model.
        """
        aics = []
        best_model = None
        for kernel_form in kernel_list:
            self.likelihood = self.set_likelihood(self.white_noise, train_errors=self.train_errors)
            self.model = self.create_gp_model(self.likelihood, kernel_form)
            # suppress output, even for verbose=True
            self.train_model(num_iter=num_iter, learn_rate=learn_rate, verbose=False) 

            # compute aic and store best model
            aic = self.akaike_inf_crit()
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
        Retrieves the hyperparameters of the GP model.

        Returns:
        - dict: Dictionary of hyperparameter names and their values.
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

    def bayesian_inf_crit(self):
        """
        Calculates the Bayesian Information Criterion (BIC) for the model.

        Returns:
        - float: The BIC value for the trained model.
        """
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        log_marg_like = mll(
            self.model(self.train_times), self.train_rates
            ).item()

        num_params = sum([p.numel() for p in self.model.parameters()])
        num_data = len(self.train_times)

        bic = -2 * log_marg_like + num_params * np.log(num_data)
        return bic

    def akaike_inf_crit(self):
        """
        Calculates the Akaike Information Criterion (AIC) for the model.

        Returns:
        - float: The AIC value for the trained model.
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
        Generates samples from the posterior predictive distribution.

        Uses the trained GP model to sample from the posterior at specified
        prediction times.

        Parameters:
        - pred_times (torch.Tensor): Time points for posterior sampling.
        - num_samples (int): Number of samples to draw from the posterior.

        Returns:
        - torch.Tensor: Samples from the posterior predictive distribution.
        """
        # Check if pred_times is a torch tensor
        if not isinstance(pred_times, torch.Tensor):
            try: 
                pred_times = torch.tensor(pred_times, dtype=torch.float32)
            except TypeError:
                raise TypeError("pred_times must be a torch tensor or convertible to one.")

        # Predictive posterior mode
        self.model.eval()
        self.likelihood.eval()

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = self.likelihood(self.model(pred_times))
            samples = pred_dist.sample(sample_shape=torch.Size([num_samples]))

        # Unstandardize
        samples = samples * self.lightcurve.unstandard_std + self.lightcurve.unstandard_mean
        samples = samples.numpy()

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
        Predicts the posterior mean and 2-sigma confidence
        intervals at the specified prediction times.

        Parameters:
        - pred_times (torch.Tensor): Time points for predictions.

        Returns:
        - mean (torch.Tensor): Predicted posterior mean.
        - lower (torch.Tensor): Lower bound of the 2-sigma confidence interval.
        - upper (torch.Tensor): Upper bound of the 2-sigma confidence interval.
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

        # Unstandardize
        mean = mean * self.lightcurve.unstandard_std + self.lightcurve.unstandard_mean
        lower = lower * self.lightcurve.unstandard_std + self.lightcurve.unstandard_mean 
        upper = upper * self.lightcurve.unstandard_std + self.lightcurve.unstandard_mean

        return mean.numpy(), lower.numpy(), upper.numpy()
    
    def plot(self, pred_times=None):
        """
        Plots the Gaussian Process prediction and samples.

        Parameters:
        - pred_times (torch.Tensor or None): Time points for predictions (optional).
        """
        if pred_times is None:
            step = (self.train_times.max()-self.train_times.min())/1000
            pred_times = np.arange(self.train_times.min(), self.train_times.max()+step, step)
        
        predict_mean, predict_lower, predict_upper = self.predict(pred_times)
        plt.fill_between(pred_times, predict_lower, predict_upper, color='dodgerblue', alpha=0.2, label=r'Prediction 2$\sigma$ CI')
        plt.plot(pred_times, predict_mean, color='dodgerblue', label='Prediction Mean')

        sample = self.sample(pred_times, num_samples=1, _save_to_state=False)
        plt.plot(pred_times, sample[0], color='orange', lw=1, label='Sample')

        if self.train_errors.size(dim=0) > 0:
            plt.errorbar(
            self.lightcurve.times, self.lightcurve.rates,
            yerr=self.lightcurve.errors, fmt='o', color='black', label='Data', lw=1, ms=2
            )
        else:
            plt.scatter(self.lightcurve.times, self.lightcurve.rates, color='black', label='Data', s=2)

        plt.xlabel('Time')
        plt.ylabel('Rate')
        plt.title('Gaussian Process Prediction')
        plt.legend()
        plt.show()

    def save(self, file_path):
        """
        Saves the full GaussianProcess instance to a file.

        Parameters:
        - file_path (str): The path to save the instance.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        print(f"GaussianProcess instance saved to {file_path}.")

    @staticmethod
    def load(file_path):
        """
        Loads a GaussianProcess instance from a file.

        Parameters:
        - file_path (str): The path to load the instance from.

        Returns:
        - GaussianProcess: The loaded instance.
        """
        with open(file_path, "rb") as f:
            instance = pickle.load(f)
        print(f"GaussianProcess instance loaded from {file_path}.")
        return instance
