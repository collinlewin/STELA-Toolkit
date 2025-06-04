# Introduction to Gaussian Processes

A **Gaussian Process (GP)** is one of the most powerful tools in Bayesian statistics for modeling functions. In STELA, we use GPs to interpolate AGN light curves — enabling advanced time and frequency analyses even with noisy, irregularly sampled data.

This page serves as both a conceptual and practical introduction to GPs, with an emphasis on **Bayesian foundations**, **statistical structure**, and how this model supports the goals of time-domain astronomy.

---

## 1. From Parametric Models to Function Distributions

In classical modeling, such as linear regression, we assume:

$$
y = X \beta + \epsilon
$$

This assumes a **parametric form** — a line or polynomial — and estimates parameters like \( \beta \). But what if we don't know the form of the function?

A **Gaussian Process** is a **distribution over functions**.

Instead of picking a specific formula, we say:

> "Any set of values from this unknown function should follow a multivariate normal distribution."

This means:

- Every time point \( t \) is a random variable \( f(t) \)
- Any collection \( [f(t_1), ..., f(t_n)] \) is jointly Gaussian

---

## 2. GP Specification: Mean and Covariance Functions

A GP is fully defined by:

- A **mean function** \( m(t) = \mathbb{E}[f(t)] \)
- A **covariance function** \( k(t, t') = \text{Cov}(f(t), f(t')) \)

We write this as:

$$
f(t) \sim \mathcal{GP}(m(t), k(t, t'))
$$

### In STELA:

- We assume \( m(t) = 0 \) after standardizing the data.
- The covariance function is called the **kernel**, which determines the shape, smoothness, and periodicity of the model.

---

## 3. Kernel Functions: The Heart of the GP

The kernel defines the relationship between any two inputs. Common kernels include:

- **RBF (Squared Exponential)**

  $$
  k(t, t') = \sigma^2 \exp\left(-\frac{(t - t')^2}{2\ell^2}\right)
  $$

- **Matern** (1/2, 3/2, 5/2)
- **Rational Quadratic**
- **Spectral Mixture**

Each kernel has **hyperparameters**:

- \( \ell \): length scale
- \( \sigma \): output variance
- For spectral: mixture weights, frequencies

---

## 4. Bayesian Inference and GP Predictions

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

### Posterior prediction:

$$
\mathbb{E}[f_*] = K_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{y}
$$

$$
\text{Cov}[f_*] = K_{**} - K_*^\top (K + \sigma_n^2 I)^{-1} K_*
$$

---

## 5. Noise Handling in STELA

STELA handles noise in two ways:

- **Explicit error bars** on light curve points
- **White noise model**:

  Adds a diagonal term \( \sigma_w^2 I \) to the kernel for unaccounted stochastic noise

---

## 6. Training the GP: Marginal Likelihood

We don’t sample hyperparameters — we **optimize them** by maximizing the **marginal likelihood**:

$$
p(\mathbf{y} \mid \theta) = \mathcal{N}(0, K_\theta + \sigma_n^2 I)
$$

Its log form:

$$
\log p(\mathbf{y}) = -\frac{1}{2} \mathbf{y}^\top K^{-1} \mathbf{y}
- \frac{1}{2} \log |K|
- \frac{n}{2} \log(2\pi)
$$

Each term has an interpretation:

- **Data fit**: how well the model explains the data
- **Complexity penalty**: penalizes overfitting
- **Normalization**: adjusts for scale

STELA **minimizes the Negative Log Marginal Likelihood (NLML)** using the Adam optimizer.

---

## 7. Sampling from the Posterior

STELA allows you to:

- Sample multiple **posterior realizations**
- Feed each into time or frequency-domain tools
- Propagate uncertainty to downstream metrics

---

## 8. Summary

GPs offer:

- Flexibility: no fixed model form
- Interpretability: kernel tells you about variability
- Uncertainty: quantified at every prediction
- Consistency: grounded in Bayesian statistics

They form the mathematical foundation for STELA’s light curve modeling and variability analysis.