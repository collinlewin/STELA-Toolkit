# STELA Toolkit Overview

STELA (Sampling Time for Even Lightcurve Analysis) is a Python package for interpolating astrophysical light curves using Gaussian Processes in order to compute frequency-domain and standard time domain data products.

This package was designed for researchers who need to:

- Interpolate irregular, noisy light curves
- Quantify variability in the time and frequency domains
- Model lag phenomena using cross-correlations and GP-informed analysis
- Simulate synthetic light curves with physically motivated structure

---

## Core Capabilities

### 1. Gaussian Process (GP) Modeling

STELA uses Gaussian Processes to model AGN light curves in a Bayesian framework:

- Standardizes input data and removes trends

- Applies optional:

    - **Box-Cox transformations** to improve normality
    - **Standardization/unstandardization** for numerical stability

- Fits kernel hyperparameters by minimizing the **negative log marginal likelihood (NLML)**

- Allows easy selection among multiple kernel types:
    - Radial Basis Function (RBF)
    - Rational Quadratic (RQ)
    - Matern (1/2, 3/2, 5/2)
    - Spectral Mixture (for advanced periodic behavior)

- Supports **white noise fitting** alongside observational error

After training, you can generate posterior samples or predictions at arbitrary times — enabling accurate interpolation and uncertainty propagation.

---

### 2. Frequency-Domain Tools

Using GP-modeled light curves or evenly-sampled data, STELA enables:

- **Power Spectrum** — Measures variability power at different frequencies
- **Cross Spectrum** — Frequency-domain relationship between two light curves
- **Coherence** — Quantifies signal correlation at each frequency
- **Lag-Frequency Spectrum** — Time delay as a function of frequency
- **Lag-Energy Spectrum** — Time lag across energy bands

All tools propagate uncertainty using GP samples or Monte Carlo simulations.

---

### 3. Time-Domain Lag Analysis

STELA supports two methods for measuring time-domain lags:

- **Interpolated Cross-Correlation Function (ICCF)**

    - Interpolates one light curve onto the other's grid
    - Estimates peak or centroid of the correlation curve

- **GP-Based Cross-Correlation**

    - Uses GP realizations to compute posterior-distributed lags
    - Reflects realistic uncertainty and sampling effects

---

### 4. Data Simulation and Preprocessing

STELA includes tools to:

- Simulate synthetic light curves with:

    - Power-law power spectra
    - Specified variability amplitude
    - Injected time lags

- Load time series from `.dat`, `.csv`, or FITS files

- Automatically detect and apply preprocessing steps:

    - Normality correction
    - Standardization
    - Resampling to regular time grid

---

## Unified API Design

Each major object (e.g., `PowerSpectrum`, `LagFrequencySpectrum`, `GaussianProcess`) includes:

- `.plot()` method with consistent styling
- Uncertainty-aware results
- Fully documented parameters and attributes

STELA accepts either raw `LightCurve` objects or trained `GaussianProcess` models in most functions — allowing users to apply the pipeline flexibly.

---

## Next Steps

- [Install STELA](installation.md)
- [Understand Gaussian Processes](gaussian_process_intro.md)
- [Run the tutorial notebook](tutorial.ipynb)
- [Explore the module reference](reference/gaussian_process.md)

This package was designed for researchers who need to:
- Interpolate irregular, noisy light curves
- Quantify variability in the time and frequency domains
- Model lag phenomena using cross-correlations and GP-informed analysis
- Simulate synthetic light curves with physically motivated structure