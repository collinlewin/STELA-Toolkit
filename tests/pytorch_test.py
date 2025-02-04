import gpytorch
import torch
def bic(lml, num_params, num_data):
    return -2 * lml + num_params * np.log(num_data)

def train_gp_model(train_x, train_y, kernel, lr = 0.01, training_iter = 2000, verbal = False):
    
    def create_gp_model(train_x, train_y, likelihood, kernel):
        class GPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(GPModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ZeroMean()
                if kernel == 'Matern12':
                    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))
                elif kernel == 'Matern32':
                    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
                elif kernel == 'Matern52':
                    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
                elif kernel == 'RQ':
                    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
                elif kernel == 'RBF':
                    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                else:
                    raise(ValueError('Invalid kernel type'))

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        return GPModel(train_x, train_y, likelihood)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        learn_additional_noise=True,
        noise_prior=gpytorch.priors.NormalPrior(0.1, 0.5), 
    )

    # create gp model
    model = create_gp_model(train_x, train_y, likelihood, kernel)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()

        if verbal:
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    model.covar_module.base_kernel.lengthscale.item(),
                    model.likelihood.noise.item()
                ))

        optimizer.step()

    num_params = sum([p.numel() for p in model.parameters()])
    inf_crit = bic(-loss.item(), num_params, len(train_x))
    
    return model, likelihood, inf_crit