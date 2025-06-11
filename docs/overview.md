# STELA Toolkit Overview

STELA (Sampling Time for Even Lightcurve Analysis) is a Python package for interpolating astrophysical light curves using Gaussian Processes (more ML models to come!) in order to compute frequency-domain and standard time domain data products.
---

## Core Workflow

### 1. Load and Inspect Your Light Curve

STELA begins with the `LightCurve` class, which allows users to load irregularly or regularly sampled light curves from formats including FITS, CSV, and plain text. These objects are used directly in downstream analyses or as input to Gaussian Process models.

---

### 2. Preprocess and Clean the Data

STELA includes robust preprocessing tools that:

- Check for normality using Shapiro-Wilk or Lilliefors tests, chosen automatically based on sample size
- Apply Box-Cox transformations to approximate Gaussianity
- Standardize the light curve to zero mean and unit variance
- Remove NaNs and outliers
- Detrend via polynomial fitting and trimming
- Visualize Gaussianity using Q-Q plots

All transformations can either overwrite the original light curve (`save=True`).

---

### 3. Gaussian Process (GP) Modeling

STELA uses Gaussian Processes to model AGN light curves in a Bayesian framework:

- Functions for testing normality assumption via Shapiro-Wilk or Lilliefors tests, which is chosen automatically based on sample size

    - **Box-Cox transformation** transforms the data to reach normality by optimizing a parameter from the data

- Fits kernel hyperparameters by minimizing the **negative log marginal likelihood (NLML)**

- Allows easy selection among multiple kernel types, including automatic kernel selection by comparing post-trained kernel Akaike information criterion (AIC):
    - Radial Basis Function (RBF)
    - Rational Quadratic (RQ)
    - Matern (1/2, 3/2, 5/2)
    - Spectral Mixture (for advanced periodic behavior)

- Supports **white noise fitting** alongside observational error

After training, you can generate posterior samples or predictions at any times of interest, enabling accurate interpolation as well as uncertainty propagation by drawing more and more evenly sampled realizations/samples to compute the data product of interest.

---

### 2. Frequency-Domain Tools

Taking inputs of either a trained GP model, or evenly-sampled data defined using the `LightCurve` class, STELA can compute:

- **Power Spectrum**: Measures the amount of variability/power at different frequencies (normalized periodogram). Includes support for fitting analytic models (e.g. power laws or power law + Lorentzian) to the PSD using a maximum likelihood approach.
- **Cross Spectrum**: Measures the relationship between two light curves via both real and imaginary parts
- **Lag-Frequency Spectrum**: Time delay as a function of frequency, from the phase of the cross spectrum
- **Lag-Energy Spectrum**: Time lag across energy bands
- **Coherence Spectrum**: Quantifies how correlated the signal is at each frequency. Bias due to noise can be optionally accounted for.

When Gaussian Process realizations are used, STELA performs Monte Carlo sampling to derive uncertainties: Uncertainty is propagated by computing the data product of interest for each pair of realizations, resulting in a distribution of the final quantity in each frequency bin.

---

### 5. Time-Domain Lag Analysis

STELA provides two approaches for time-domain lag estimation:

- **Interpolated Cross-Correlation (ICCF)**: 
  - Linearly interpolates one light curve onto another
  - Computes peak and centroid lag
  - Uses Monte Carlo simulations for uncertainty estimation via redrawing flux values

- **GP-Based Cross-Correlation**:
  - Computes lag distributions across many GP realizations
  - Outputs mean and standard deviation of peak and centroid lags

---

### 6. Light Curve Simulation

STELA allows for simulating synthetic light curves using the method of Timmer and Konig:

- Specified power spectral properties (e.g., power-law slopes)
- Injected time lags with configurable impulse response functions
- Control over sampling, noise level, and structure

These simulations are useful for benchmarking recovery of lags and variability structure.

---

## Unified API Design

Every major class in STELA (e.g., `PowerSpectrum`, `LagFrequencySpectrum`, `GaussianProcess`) includes:

- `.plot()` method with easily visualizing results
- Accepts inputs of either raw `LightCurve` objects or trained `GaussianProcess` models.

    - If samples have not been previously generated for a Gaussian process model, STELA will do so itself, generating 1000 samples on an evenly sampled time grid of 1000 points.

---

## Next Steps

- [Install STELA](installation.md)
- [Understand Gaussian Processes](gaussian_process_intro.md)
- [Run the tutorial notebook](tutorial.ipynb)
- [Explore the module reference](reference/gaussian_process.md)