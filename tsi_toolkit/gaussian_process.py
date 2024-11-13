import torch
import gpytorch
import numpy as np

from .data_loader import TimeSeries

# to do: consider noise prior
class GaussianProcess(gpytorch.models.ExactGP):
    def __init__(self, timeseries=None, train_t=[], train_y=[], train_e=[], kernel_form='auto', white_noise=True, run_training=True, 
                 train_iter=1000, learn_rate=0.01, sample_time_grid=[], num_samples=1000, verbose=True):

        if timeseries:
            if isinstance(timeseries, TimeSeries):
                self.timeseries = timeseries
            else:
                raise TypeError("Expected timeseries to be a TimeSeries object.")
        
        elif train_t and train_y:
            self.timeseries = TimeSeries(times=train_t, values=train_y, errors=train_e)
        
        else:
            raise ValueError(
                "Please provide either a TimeSeries object as 'timeseries' "
                "or arrays for 'train_t' and 'train_y' (and optionally 'train_e')."
            )
    
        # Standardize the time series data to match zero mean function
        self.timeseries.standardize()

        # Convert time series data to PyTorch tensors
        self.train_t = torch.tensor(self.timeseries.times)
        self.train_y = torch.tensor(self.timeseries.rates)
        if self.timeseries.errors:
            self.train_e = torch.tensor(self.timeseries.errors)

        # Set likelihood
        self.likelihood = self.set_likelihood(white_noise, train_e=self.train_e)

        # Find best kernel based on AIC
        if kernel_form in ['auto', 'advise me, please?'] or isinstance(kernel_form, list):
            if isinstance(kernel_form, list):
                kernel_list = kernel_form
            else:
                kernel_list = ['Matern12', 'Matern32', 'Matern52', 'RQ', 'RBF', 'SpectralMixture, 4']
            self.model = self.find_best_kernel(kernel_list, train_iter, learn_rate, verbose)

        else:
            self.model = self.create_gp_model(self.train_t, self.train_y, self.likelihood, kernel_form)
            if run_training:
                self.train(train_iter, learn_rate, verbose)

            # Calculate marginal log likelihood, BIC, and optimal hyperparameters
        if sample_time_grid:
            self.samples = self.sample(sample_time_grid, num_samples)
            if verbose:
                print(f"Samples generated: {self.samples.shape}, access with 'samples' attribute.")

    def create_gp_model(train_t, train_y, likelihood, kernel):

        class GPModel(gpytorch.models.ExactGP):
            def __init__(self, train_t, train_y, likelihood):
                super(GPModel, self).__init__(train_t, train_y, likelihood)
                self.mean_module = gpytorch.means.ZeroMean()
                self.covar_module = self.set_kernel(kernel)

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        return GPModel(train_t, train_y, likelihood)
    
    def set_likelihood(self, white_noise, train_e=None):
        
        if train_e:
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=self.train_e ** 2, 
                learn_additional_noise=white_noise
            )

        else:
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                    noise=1e-8,
                    learn_additional_noise=white_noise
            )

        return likelihood

    def set_kernel(self, kernel_form: str):
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
        # Parse kernel type and optional number of mixtures
        if ',' in kernel_form:
            kernel_type, num_mixtures_str = kernel_form.split(',')
            kernel_type = kernel_type.strip()
            try:
                num_mixtures = int(num_mixtures_str.strip())
            except ValueError:
                raise ValueError(f"Invalid number of mixtures '{num_mixtures_str}' for Spectral Mixture kernel.")
        else:
            kernel_type = kernel_form.strip()
        
        # Kernel mapping with special handling for Spectral Mixture kernel
        kernel_mapping = {
            'Matern12': gpytorch.kernels.MaternKernel(nu=0.5),
            'Matern32': gpytorch.kernels.MaternKernel(nu=1.5),
            'Matern52': gpytorch.kernels.MaternKernel(nu=2.5),
            'RQ': gpytorch.kernels.RQKernel(),
            'RBF': gpytorch.kernels.RBFKernel(),
            'SpectralMixture': gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures)
        }
        
        # Assign kernel if type is valid
        if kernel_type in kernel_mapping:
            kernel = kernel_mapping[kernel_type]
            if kernel_type == 'SpectralMixture':
                kernel.initialize_from_data(self.train_t, self.train_y)

            covar_module = gpytorch.kernels.ScaleKernel(kernel)
            
        else:
            raise ValueError(f"Invalid kernel type '{kernel_type}'. Choose from {list(kernel_mapping.keys())}.")
        
        return covar_module
    
    def find_best_kernel(self, kernel_list, train_iter, learn_rate, verbose=True):
        aics = []
        best_model = None
        for kernel_form in kernel_list:
            self.model = self.create_gp_model(self.train_t, self.train_y, self.likelihood, kernel_form)
            self.model, aic, _ = self.train(train_iter, learn_rate, verbose)
            aics.append(aic)
            if aic <= min(aics):
                best_model = self.model
        
        best_aic = min(aics)
        best_kernel = kernel_list[aics.index(best_aic)]
        if verbose:
            print(f"Kernel AICs (lower is better): {for (k, a) in zip(kernel_list, aics)}")
            print(f"Best kernel: {best_kernel} with AIC: {best_aic}")

        return best_model
    
    def train(self, model, train_iter, learn_rate, verbose=False):
        model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(train_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()

            # Model output
            output = model(self.train_t)

            # Calc negative likelihood and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()

            if verbose:
                if self.kernel_form == 'SpectralMixture':
                    print('Iter %d/%d - Loss: %.3f   mixture_lengthscales: %.3f   mixture_weights: %s' % (
                        i + 1, train_iter, loss.item(),
                        model.covar_module.base_kernel.mixture_scales.item(),
                        model.covar_module.base_kernel.mixture_weights.item()
                    ))

                else:
                    if self.white_noise:
                        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                            i + 1, train_iter, loss.item(),
                            model.covar_module.base_kernel.lengthscale.item(),
                            model.likelihood.noise.item()
                        ))
                    
                    else:
                        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f' % (
                            i + 1, train_iter, loss.item(),
                            model.covar_module.base_kernel.lengthscale.item()
                        ))

            optimizer.step()

        if verbose:
            print(f"Training complete. Final loss: {loss.item()}. Final hyperparameters: {self.get_hyperparameters()}")

    def get_hyperparameters(self):
        hypers = self.model.covar_module.base_kernel.hypers
        if self.white_noise:
            hypers += [self.model.likelihood.second_noise.item()]

        return hypers

    def bayesian_inf_crit(self):
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        log_marg_like = mll(self.model(self.train_t), self.train_y)

        # need to check if this includes the white noise hyperparameter!!
        num_params = sum([p.numel() for p in self.model.parameters()])
        num_data = len(self.train_t)

        bic = -2 * log_marg_like + num_params * np.log(num_data)
        return bic
    
    def akaike_inf_crit(self, log_marg_like, num_params):
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        log_marg_like = mll(self.model(self.train_t), self.train_y)

        # need to check if this includes the white noise hyperparameter!!
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
            samples = pred_dist.sample(sample_shape=torch.Size([num_samples]))

        return samples

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        self.likelihood.eval()
        print(f"Model loaded from {path}.")