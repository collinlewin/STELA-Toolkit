To do :
First (essential) actions:
- Finish changing all classes to import model
    - go back and change it so that we can detect if a lightcurve or a model, and just have a single input, so users don't think they need to put in both a lightcurve and a model.
- create methods for both gp samples and multiple light curves
- Log transformation
- distribution testing
- Create lag_energy spectrum
- All handling of noise needs to consider if the lightcurve is in counts or not, which will inform Poisson (scrap that parameter)
- Create tutorial, which will test all of the functionalities thus far
- Test coherence noise method for GPs
    how does GP noise parameter contrast to noise level of PSD and to the approximations made in coherence.py?
    * the value that is fit is the variance of the white noise at each point, which we expect to be ~counts

Second actions:
- Detection for training convergence, etc.
- Detection for learning rate, iterations
- Method for saving data products
- GPU parallelization for GPs

Brainstorming:
- Creating an LLM to work with my code?
