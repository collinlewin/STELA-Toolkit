import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

from .data_loader import TimeSeries
from .preprocessing import Preprocessing


class GaussianProcess():
    def __init__(self,
                 timeseries,
                 kernel_form='auto',
                 white_noise=True,
                 run_training=True,
                 train_iter=1000,
                 learn_rate=1e-2,
                 sample_time_grid=[],
                 num_samples=1000,
                 plot_gp=False,
                 verbose=True):

        # To Do: reconsider noise prior, add a mean function function for forecasting, more verbose options
        """
        Initializes the GaussianProcess object and trains the model.

        The method sets up the GP model based either 1) the user-defined kernel, or
        2) the best kernel found based on post-training AIC.
        It optionally trains the model and generates samples
        from the posterior if a sampling grid is provided.

        Parameters:
        - timeseries (TimeSeries): The time series data to model.
        - kernel_form (str or list): Kernel to use ('auto' for selection or a specific kernel).
        - white_noise (bool): Whether to include an additional learnable white noise term.
        - run_training (bool): Whether to train the model upon initialization.
        - train_iter (int): Number of iterations for model training.
        - learn_rate (float): Learning rate for the optimizer.
        - sample_time_grid (array-like): Time grid for posterior sampling (optional).
        - num_samples (int): Number of posterior samples to generate (if sampling grid provided).
        - plot_gp (bool): Whether to plot the GP prediction upon initialization.
        - verbose (bool): Whether to print detailed output during operations.
        """
        if isinstance(timeseries, TimeSeries):
            self.timeseries = timeseries
        else:
            raise TypeError("Expected timeseries to be a TimeSeries object.\n"
                            "Please input your data into the TimeSeries class first."
                        )

        # Standardize the time series data to match zero mean function
        try:
            Preprocessing.standardize(self.timeseries)
        except ValueError:
            if verbose:
                print("Data is already standardized. Skipping standardization.")

        # Convert time series data to PyTorch tensors
        self.train_times = torch.tensor(self.timeseries.times, dtype=torch.float32)
        self.train_values = torch.tensor(self.timeseries.values, dtype=torch.float32)
        if self.timeseries.errors.size > 0:
            self.train_errors = torch.tensor(self.timeseries.errors, dtype=torch.float32)

        self.white_noise = white_noise

        # Find best kernel based on AIC
        if kernel_form == 'auto' or isinstance(kernel_form, list):
            if isinstance(kernel_form, list):
                kernel_list = kernel_form
            else:
                kernel_list = ['Matern12', 'Matern32', 'Matern52', 'RQ', 'RBF', 'SpectralMixture, 4']

            best_model, best_likelihood = self.find_best_kernel(
                kernel_list, train_iter=train_iter, learn_rate=learn_rate, verbose=verbose
                )
            self.model = best_model
            self.likelihood = best_likelihood

        # Use specified kernel
        else:
            self.likelihood = self.set_likelihood(self.white_noise, train_errors=self.train_errors)
            self.model = self.create_gp_model(
                self.train_times, self.train_values, self.likelihood, kernel_form
                )
            
            # Separate training needed only if kernel not automatically selected
            if run_training:
                self.train_model(train_iter=train_iter, learn_rate=learn_rate, verbose=verbose)

        # Generate samples if sample_time_grid is provided
        if sample_time_grid:
            self.samples = self.sample(sample_time_grid, num_samples=num_samples)
            if verbose:
                print(f"Samples generated: {self.samples.shape}, access with 'samples' attribute.")

        if plot_gp:
            self.plot(sample_time_grid)

    def create_gp_model(self, train_times, train_values, likelihood, kernel):
        """
        Creates and returns a Gaussian Process model.

        Uses the specified kernel and likelihood to define the GP model.
        The GP model is built as a subclass of gpytorch.models.ExactGP.

        Parameters:
        - train_times (torch.Tensor): Training time points.
        - train_values (torch.Tensor): Training data values.
        - likelihood (gpytorch.likelihoods.Likelihood): The likelihood function for the GP.
        - kernel (str): Kernel type to use for the GP.

        Returns:
        - GPModel: A Gaussian Process model with the specified kernel and likelihood.
        """
        class GPModel(gpytorch.models.ExactGP):
            def __init__(gp_self, train_times, train_values, likelihood):
                super(GPModel, gp_self).__init__(train_times, train_values, likelihood)
                gp_self.mean_module = gpytorch.means.ZeroMean()
                gp_self.covar_module = self.set_kernel(kernel)

            def forward(gp_self, x):
                mean_x = gp_self.mean_module(x)
                covar_x = gp_self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            
        return GPModel(train_times, train_values, likelihood)

    def set_likelihood(self, white_noise, train_errors=torch.tensor([])):
        """
        Sets the likelihood for the GP model. 

        If errors are provided, uses heteroscedastic noise. Otherwise, uses homoscedastic noise.
        If white_noise is True, includes a learnable (homoscedastic) white noise term.

        Parameters:
        - white_noise (bool): Whether to include a learnable white noise term.
        - train_errors (torch.Tensor): Errors for the training data values.

        Returns:
        - gpytorch.likelihoods.Likelihood: Configured likelihood for the GP model.
        """
        if train_errors.size(dim=0) > 0:
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=self.train_errors ** 2,
                learn_additional_noise=white_noise
            )

        else:
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=1e-8,
                learn_additional_noise=white_noise
            )

        return likelihood

    def set_kernel(self, kernel_form):
        """
        Sets the covariance kernel for the Gaussian Process model, supporting a range
        of kernel types including Matern, Rational Quadratic, RBF, and Spectral Mixture.

        Parameters:
        - kernel_form (str): Type of kernel to use (e.g., 'Matern32', 'RBF', 'SpectralMixture, N').

        Returns:
        - gpytorch.kernels.Kernel: Configured kernel for the GP model.
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
            num_mixtures = 4  # set value for kernel_mapping when Spectral Mixture kernel not used

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
            if kernel_form == 'SpectralMixture':

                kernel.initialize_from_data(self.train_times, self.train_values)

            covar_module = gpytorch.kernels.ScaleKernel(kernel)

        else:
            raise ValueError(
                f"Invalid kernel functional form '{kernel_form}'. Choose from {list(kernel_mapping.keys())}.")

        self.kernel_form = kernel_form
        return covar_module

    def find_best_kernel(self, kernel_list, train_iter=1000, learn_rate=1e-2, verbose=False):
        """
        Finds the best kernel based on the Akaike Information Criterion (AIC).

        Iteratively trains the GP model using different kernels from the provided list
        and selects the kernel with the lowest AIC.

        Parameters:
        - kernel_list (list): List of kernel types to evaluate.
        - train_iter (int): Number of training iterations for each kernel.
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
            self.model = self.create_gp_model(
                self.train_times, self.train_values, self.likelihood, kernel_form
                )
            # suppress output, even for verbose=True
            self.train_model(train_iter=train_iter, learn_rate=learn_rate, verbose=False) 

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

    def train_model(self, train_iter=1000, learn_rate=1e-2, verbose=False):
        """
        Trains the Gaussian Process model, hyperparameters optimized 
        using Adam optimizer given number of training iterations and 
        learning rate.

        If verbose is True, prints the training progress and final hyperparameters.
        If hyperparameters not converged...
            - monotonic progression? --> consider increasing train_iter.
            - fluctuating around potential optimum? --> consider decreasing learn_rate.

        Parameters:
        - train_iter (int): Number of training iterations.
        - learn_rate (float): Learning rate for the optimizer.
        - verbose (bool): Whether to print detailed information during training.
        """
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(train_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()

            # Model output
            output = self.model(self.train_times)

            # Calc negative likelihood and backprop gradients
            loss = -mll(output, self.train_values)
            loss.backward()
            optimizer.step()

            if verbose:
                if self.kernel_form == 'SpectralMixture':
                    mixture_scales = self.model.covar_module.base_kernel.mixture_scales.detach().numpy().flatten()
                    mixture_weights = self.model.covar_module.base_kernel.mixture_weights.detach().numpy().flatten()
                    
                    if self.white_noise:
                        print('Iter %d/%d - Loss: %.3f   mixture_lengthscales: %s   mixture_weights: %s   noise: %.3f' % (
                            i + 1, train_iter, loss.item(),
                            mixture_scales.round(3),
                            mixture_weights.round(3),
                            self.model.likelihood.second_noise.item()
                        ))
                    else:
                        print('Iter %d/%d - Loss: %.3f   mixture_lengthscales: %s   mixture_weights: %s' % (
                            i + 1, train_iter, loss.item(),
                            mixture_scales.round(3),
                            mixture_weights.round(3)
                        ))

                else:
                    if self.white_noise:
                        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                            i + 1, train_iter, loss.item(),
                            self.model.covar_module.base_kernel.lengthscale.item(),
                            self.model.likelihood.second_noise.item()
                        ))
                    else:
                        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f' % (
                            i + 1, train_iter, loss.item(),
                            self.model.covar_module.base_kernel.lengthscale.item()
                        ))

        if verbose:
            final_hypers = self.get_hyperparameters()
            print(
                "Training complete. \n"
                f"   - Final loss (NLML): {loss.item():0.5}\n"
                f"   - Final hyperparameters:"
            )
            for key, value in final_hypers.items():
                print(f"      {key:42}: {np.round(value, 4)}")

    def get_hyperparameters(self):
        """
        Retrieves the hyperparameters of the GP model.

        Extracts and transforms the model's hyperparameters for reporting or debugging.

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
            self.model(self.train_times), self.train_values
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
            self.model(self.train_times), self.train_values
            ).item()

        num_params = sum([p.numel() for p in self.model.parameters()])

        aic = -2 * log_marg_like + 2 * num_params
        return aic

    def sample(self, pred_times, num_samples):
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
        samples = samples * self.timeseries.unstandard_std + self.timeseries.unstandard_mean
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
        mean = mean * self.timeseries.unstandard_std + self.timeseries.unstandard_mean
        lower = lower * self.timeseries.unstandard_std + self.timeseries.unstandard_mean
        upper = upper * self.timeseries.unstandard_std + self.timeseries.unstandard_mean

        return mean, lower, upper
    
    def plot(self, pred_times=None):
        """
        Plots the Gaussian Process prediction and samples.

        Parameters:
        - pred_times (torch.Tensor or None): Time points for predictions (optional).
        """
        if pred_times is None:
            pred_times = torch.linspace(self.train_times.min(), self.train_times.max(), 1000)

        predict_mean, predict_lower, predict_upper = self.predict(pred_times)
        plt.fill_between(pred_times, predict_lower, predict_upper, color='dodgerblue', alpha=0.2, label=r'Prediction 2$\sigma$ CI')
        plt.plot(pred_times, predict_mean, color='dodgerblue', label='Prediction Mean')

        sample = self.sample(pred_times, num_samples=1)
        plt.plot(pred_times, sample[0], color='orange', lw=1, label='Sample')

        if self.train_errors.size(dim=0) > 0:
            plt.errorbar(
            self.train_times, self.train_values,
            yerr=self.train_errors, fmt='o', color='black', label='Data', lw=1, ms=2
            )
        else:
            plt.scatter(self.train_times, self.train_values, color='black', label='Data')

        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title('Gaussian Process Prediction')
        plt.legend()
        plt.show()

    def save_model(self, path):
        """
        Saves the GP model to a specified path.
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """
        Loads a GP model from a specified path.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        self.likelihood.eval()
        print(f"Model loaded from {path}.")
