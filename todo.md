Essential before release:
- Test GP functionalities
- Tutorial
- Documentation
- Errors in simulations

Next/first actions:
- Create methods for multiple light curves
- Allow the user to make composite kernels.

Second actions:
- Detection for training convergence, etc.
- Detection for learning rate, iterations
- Method for saving data products
- GPU parallelization for GPs

Testing:
- Distribution testing
- Transformations, including in GaussianProcess
- Coherences for different types of input data 
    - how does GP noise parameter contrast to noise level of PSD and to the approximations made in coherence.py?
    * the value that is fit is the variance of the white noise at each point, which we expect to be ~counts
- Periodic kernel 