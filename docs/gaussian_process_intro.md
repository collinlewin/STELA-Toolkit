# Introduction to Gaussian Processes

A **Gaussian Process (GP)** is one of the most powerful tools in Bayesian statistics for modeling functions. In STELA, we use GPs to interpolate noisy, irregularly sampled light curves in order to generate frequency-resolved data products using Fourier techniques.
---

## 1. From Parametric to Nonparametric Models

In classical modeling approaches—like linear regression—we assume a specific functional form:

$$
y = X \beta + \epsilon
$$

This is a **parametric model**, where the function is fully described by a finite set of parameters such as \( \beta \). But what if we don’t know the correct form of the function? What if the variability is too complex to capture with some formula of y = f(X)?

### Gaussian Processes: A Nonparametric Prior

A **Gaussian Process (GP)** is a **nonparametric Bayesian model**: instead of defining a formula with a small number of parameters, it places a prior directly over functions themselves, instead letting the data inform the shape of the function.

A GP defines a distribution such that...

> **Any finite collection of function values should follow a multivariate normal distribution.**

This means:

- Each time point \( t \) corresponds to a random variable \( f(t) \)
- Any set \( [f(t_1), \dots, f(t_n)] \) is jointly Gaussian
- The function's behavior is therefore, like any Gaussian, entirely determined by its mean and covariance

This flexibility allows the GP to capture complex trends and variability directly from the data.

Because the GP framework assumes normally distributed data, it's important that the observed fluxes approximate Gaussianity. STELA includes both hypothesis testing to check this assumption, as well as a **Box-Cox transformation** to help normalize the data before modeling. After inference, we invert the transformation so that predictions are returned in the original flux units.

---

## 2. Mean and Covariance/Kernel Functions

A GP is fully defined by:

- A **mean function**:  
  \( m(t) = \mathbb{E}[f(t)] \)
- A **covariance function** (or kernel):  
  \( k(t, t') = \text{Cov}(f(t), f(t')) \)

This is written as:

$$
f(t) \sim \mathcal{GP}(m(t), k(t, t'))
$$

### In STELA:

- We assume \( m(t) = 0 \) by standardizing the light curve data before modeling.
- The covariance structure is encoded via a **kernel**, which determines the shape, smoothness, and periodicity of the model. Different kernels encode different assumptions:
  - Smooth trends (e.g. RBF)
  - Quasi-periodic variability (e.g. Spectral Mixture)
  - Long- and short-term variability (e.g. Rational Quadratic)

This combination of nonparametric flexibility and probabilistic structure is what enables STELA to robustly interpolate, sample, and propagate uncertainty across all downstream Fourier analyses.

---

## 3. Kernel Functions

The kernel defines the relationship between any two inputs. Common kernels include:

- **RBF (Squared Exponential)**

  $$
  k(t, t') = \sigma^2 \exp\left(-\frac{(t - t')^2}{2\ell^2}\right)
  $$

- **Matern** (1/2, 3/2, 5/2)
- **Rational Quadratic**
- **Spectral Mixture**

The functional forms for the kernels, including additional kernels, can be found on the [GPyTorch kernel documentation](https://docs.gpytorch.ai/en/latest/kernels.html). Let me know if you would like more kernels added to STELA!

Each kernel has **hyperparameters**:

- \( \ell \): length scale
- \( \sigma \): output variance
- For spectral: mixture weights, frequencies

---

### 4. Noise Handling

STELA handles noise in two ways:

- **Explicit error bars** on light curve points (heteroscedastic noise).
- **White noise model**:

  Adds a diagonal term \( \sigma_w^2 I \) to the kernel for unaccounted stochastic noise on top of the uncertainties (homoscedastic noise).

---

## 5. Training the GP

We don’t sample hyperparameters, we **optimize ("train") them** by maximizing the **marginal likelihood**:

$$
p(\mathbf{y} \mid \theta) = \mathcal{N}(0, K_\theta + \sigma_n^2 I)
$$

Its log form:

$$
\log p(\mathbf{y}) = -\frac{1}{2} \mathbf{y}^\top K^{-1} \mathbf{y}
- \frac{1}{2} \log |K|
- \frac{n}{2} \log(2\pi)
$$

The three terms correspond to different aspects of model performance, respectively:

- **Data fit**: how well the model explains the data
- **Complexity penalty**: penalizes overfitting
- **Normalization** of the Gaussian

STELA **minimizes the Negative Log Marginal Likelihood (NLML)** using the Adam optimizer. The step size and number of steps to take (number of iterations) can be varied to ensure convergence of the final kernel hyperparameters. Use `plot_training=True` and/or `verbose=True` to check for convergence.

---

## 6. Bayesian Inference and GP Predictions

### Goal:

Given noisy data \( \mathbf{y} \) at times \( \mathbf{t} \), predict \( f_* \) at new times \( \mathbf{t}_* \).

We start with the prior:

$$
\begin{bmatrix} \mathbf{y} \\ f_* \end{bmatrix}
\sim \mathcal{N}\left(0, 
\begin{bmatrix}
K + \sigma_n^2 I & K_*^\top \\
K_* & K_{**}
\end{bmatrix}
\right)
$$

Where:

- \( K \): covariance of training points
- \( K_* \): covariance between training and test
- \( K_{**} \): covariance of test points
- \( \sigma_n^2 \): noise variance

### Posterior Prediction vs. Samples

In our GP framework, we distinguish between the **posterior prediction** (via `.predict`) and **posterior samples** (via `.sample`). Both are derived from the GP posterior but serve different purposes:

**Prediction (`predict`)** computes the *posterior mean* and *covariance* of the function at new time points:

$$
\mathbb{E}[f_*] = K_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{y}
$$

$$
\text{Cov}[f_*] = K_{**} - K_*^\top (K + \sigma_n^2 I)^{-1} K_*
$$

This provides a Gaussian distribution over function values at the desired points, with uncertainty encoded in the posterior covariance.


**Samples (`sample`)** draw individual function realizations from this posterior distribution. These are full, self-consistent samples of the latent function, incorporating both the posterior mean and covariance structure.

  In our use case, we use samples to compute the Fourier-domain products such as power spectra, cross-spectra, and frequency-resolved time lags. This allows for propagating the uncertainty in the modeling through to the downstream quantities of interest in a fully Bayesian manner.