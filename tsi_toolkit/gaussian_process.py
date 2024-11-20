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
        if isinstance(timeseries, TimeSeries):
            self.timeseries = timeseries
        else:
            raise TypeError("Expected timeseries to be a TimeSeries object."
                            "Please input your data into the TimeSeries class first."
                        )

        # Standardize the time series data to match zero mean function
        Preprocessing.standardize(self.timeseries)

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

        else:
            self.likelihood = self.set_likelihood(self.white_noise, train_errors=self.train_errors)
            self.model = self.create_gp_model(
                self.train_times, self.train_values, self.likelihood, kernel_form
                )
            
            if run_training:
                self.train_model(train_iter=train_iter, learn_rate=learn_rate, verbose=verbose)

        if sample_time_grid:
            self.samples = self.sample(sample_time_grid, num_samples=num_samples)
            if verbose:
                print(f"Samples generated: {self.samples.shape}, access with 'samples' attribute.")

        if plot_gp:
            self.plot(sample_time_grid)

    def create_gp_model(self, train_times, train_values, likelihood, kernel):
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
        Sets the covariance module (kernel) for the Gaussian Process model.

        Parameters
        ----------
        kernel_form : str
            Specifies the type of kernel to use. Options are:
            - 'Matern12' for Matern kernel with nu=0.5
            - 'Matern32' for Matern kernel with nu=1.5
            - 'Matern52' for Matern kernel with nu=2.5
            - 'RQ' for Rational Quadratic kernel
            - 'RBF' for Radial Basis Function (RBF) kernel
            - 'SpectralMixture, N' for Spectral Mixture kernel with N mixtures, where N is an integer

        Raises
        ------
        ValueError
            If an invalid kernel type is provided or if the mixture count is not valid.
        """
        kernel_form = kernel_form.strip()
        if 'SpectralMixture' in kernel_form:
            if ',' not in kernel_form:
                raise ValueError(
                    "Invalid Spectral Mixture kernel format (use 'SpectralMixture, N'). N=4 is a good starting point."
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
                "Kernel selection complete. \n"
                f"   Kernel AICs (lower is better):"
            )
            for kernel, aic in kernel_results:
                print(f"     - {kernel:15}: {aic:0.5}")

            print(f"   Best kernel: {best_kernel} (AIC: {best_aic:0.5})")

        self.kernel_form = best_kernel
        return best_model, best_likelihood

    def train_model(self, train_iter=1000, learn_rate=1e-2, verbose=False):
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
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        log_marg_like = mll(
            self.model(self.train_times), self.train_values
            ).item()

        num_params = sum([p.numel() for p in self.model.parameters()])
        num_data = len(self.train_times)

        bic = -2 * log_marg_like + num_params * np.log(num_data)
        return bic

    def akaike_inf_crit(self):
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        log_marg_like = mll(
            self.model(self.train_times), self.train_values
            ).item()

        num_params = sum([p.numel() for p in self.model.parameters()])

        aic = -2 * log_marg_like + 2 * num_params
        return aic

    def sample(self, pred_times, num_samples):
        # Predictive posterior mode
        self.model.eval()
        self.likelihood.eval()

        # Make predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = self.likelihood(self.model(pred_times))
            samples = pred_dist.sample(sample_shape=torch.Size([1]))

        return samples
    
    def predict(self, pred_times):
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = self.likelihood(self.model(pred_times))
            mean = pred_dist.mean
            lower, upper = pred_dist.confidence_region()

        return mean, lower, upper
    
    def plot(self, pred_times=np.array([])):
        if pred_times.size == 0:
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
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        self.likelihood.eval()
        print(f"Model loaded from {path}.")
