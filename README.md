![Python](https://img.shields.io/pypi/pyversions/stela-toolkit?cacheSeconds=3600)
[![PyPI](https://img.shields.io/pypi/v/stela-toolkit)](https://pypi.org/project/stela-toolkit/)
![IMG_7099 (1)](https://github.com/user-attachments/assets/55d66b16-e91c-4d8a-a3b9-047002945dc1)

<p align="right" style="font-size: 0.9em;">
  <em>Logo design by Elizabeth Jarboe</em>
</p>


**STELA (Sampling Time for Even Lightcurve Analysis) Toolkit** is a Python package for interpolating astrophysical light curves using Gaussian Processes (more ML models to come!) in order to compute frequency-domain and standard time domain data products.

---

## Documentation

Full documentation is available at:  
[https://collinlewin.github.io/stela-toolkit/](https://collinlewin.github.io/STELA-Toolkit/)

The documentation includes:
- **Overview** of all features
- **Tutorial notebook** with GP modeling and lag analysis
- **Gaussian Process guide** with detailed Bayesian background
- **Module reference** for all functions and classes

---

## Features

- Gaussian Process regression for all ML skill levels! Built-in normality testing and transformations to help reach normality, kernel selection, hyperparameter training, and much more.
- Frequency-domain products: power and cross spectra, coherence, lag-frequency and lag-energy spectra
- Time-domain lag analysis using ICCF and GP-based cross-correlation
- Simulation of synthetic light curves with custom underlying power spectra and injected lags
- Preprocessing: outlier removal methods, polynomial detrending, trimming, etc.
- Convenient plotting in every class using .plot()

---

## Requirements

STELA Toolkit requires Python ≥ 3.8 and the following core packages:

- numpy ≥ 1.20, <2.0
- scipy ≥ 1.7
- matplotlib ≥ 3.5
- astropy ≥ 5.0
- torch ≥ 1.10
- gpytorch ≥ 1.9
- statsmodels ≥ 0.13

You can install all dependencies automatically when installing the package (see below).

---

## Installation

You can install the package using pip:

```bash
pip install stela-toolkit
```

If you are installing directly from the GitHub repository:

```bash
git clone https://github.com/collinlewin/STELA-Toolkit.git
cd STELA-Toolkit
pip install .
```

You can verify the installation by running:

```python
import stela_toolkit
print(stela_toolkit.__version__)
```

If this runs without an error, you're good to go!
