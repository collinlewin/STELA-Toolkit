To do :

First (essential) actions:
- Need to make sure boxcox undoing makes sense in gaussian_process. If I undo the transformation, then how will the class know to undo the predictions? I think we need to save if the data was boxcoxed during training using an attribute.
- create methods for both gp samples and multiple light curves
- Create lag_energy spectrum
- All handling of noise needs to consider if the lightcurve is in counts or not, which will inform Poisson (scrap that parameter)
 - Honestly just go through noise treatment in Gaussian process in general
- Create tutorial, which will test all of the functionalities thus far
    
Second actions:
- Detection for training convergence, etc.
- Detection for learning rate, iterations
- Method for saving data products
- GPU parallelization for GPs

Brainstorming:
- Creating an LLM to work with my code?

Testing:
- Distribution testing
- Transformations, including in GaussianProcess
- Coherences for different types of input data 
    - how does GP noise parameter contrast to noise level of PSD and to the approximations made in coherence.py?
    * the value that is fit is the variance of the white noise at each point, which we expect to be ~counts
