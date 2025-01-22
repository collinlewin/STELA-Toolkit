First actions:
- create methods for multiple light curves
- The input errors are not right in coherence, LFS, LES. Need to pass in errors properly, and figure out what to do with them.
- All handling of noise needs to consider if the lightcurve is in counts or not, which will inform Poisson (scrap that parameter)
    - Go through noise treatment in Gaussian process in general
- Create tutorial, which will test all of the functionalities thus far
- Allowing the user to make composite kernels.

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
