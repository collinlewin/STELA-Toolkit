import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
import torch
import torch.optim as optim
from ._check_inputs import _CheckInputs
from .frequency_binning import FrequencyBinning
from .data_loader import LightCurve


class PowerSpectrum:
    """
    Compute the power spectrum of a light curve using the FFT.

    This class accepts either a STELA LightCurve object or a trained GaussianProcess model.
    If a GaussianProcess is passed, the most recently generated samples are used. 
    If no samples exist, the toolkit will automatically generate 1000 posterior realizations
    on a 1000-point grid.

    For single light curves, the FFT is applied directly to the time series.
    For GP models, the power spectrum is computed for each sampled realization,
    and the mean and standard deviation across all samples are returned.

    Power spectra are computed in variance units by default (i.e., normalized to units
    of squared flux), allowing for direct interpretation in the context of variability
    amplitude and fractional RMS.

    Frequency binning is supported via linear, logarithmic, or user-defined bins.

    Parameters
    ----------
    lc_or_model : LightCurve or GaussianProcess
        Input light curve or trained GP model.
    
    fmin : float or 'auto', optional
        Minimum frequency to include. If 'auto', uses the lowest nonzero FFT frequency.
    
    fmax : float or 'auto', optional
        Maximum frequency to include. If 'auto', uses the Nyquist frequency.
    
    num_bins : int, optional
        Number of frequency bins.
    
    bin_type : str, optional
        Binning type: 'log' or 'linear'.
    
    bin_edges : array-like, optional
        Custom bin edges (overrides `num_bins` and `bin_type`).
    
    norm : bool, optional
        Whether to normalize the power spectrum to variance units (i.e., PSD units).

    Attributes
    ----------
    freqs : array-like
        Center frequencies of each bin.
    
    freq_widths : array-like
        Bin widths for each frequency bin.
    
    powers : array-like
        Power spectrum values (or mean if using GP samples).
    
    power_errors : array-like
        Uncertainties in the power spectrum (std across GP samples if applicable).
    """

    def __init__(self,
                 lc_or_model,
                 fmin='auto',
                 fmax='auto',
                 num_bins=None,
                 bin_type="log",
                 bin_edges=[],
                 norm=True):
        
        # To do: ValueError for norm=True acting on mean=0 (standardized data)
        input_data = _CheckInputs._check_lightcurve_or_model(lc_or_model)
        if input_data['type'] == 'model':
            self.times, self.rates = input_data['data']
        else:
            self.times, self.rates, _ = input_data['data']
        _CheckInputs._check_input_bins(num_bins, bin_type, bin_edges)

        # Use absolute min and max frequencies if set to 'auto'
        self.dt = np.diff(self.times)[0]
        self.fmin = np.fft.rfftfreq(len(self.rates), d=self.dt)[1] if fmin == 'auto' else fmin
        self.fmax = np.fft.rfftfreq(len(self.rates), d=self.dt)[-1] if fmax == 'auto' else fmax  # nyquist frequency
        self.num_bins = num_bins
        self.bin_type = bin_type
        self.bin_edges = bin_edges
        self.norm = norm

        # if multiple light curve are provided, compute the stacked power spectrum
        if len(self.rates.shape) == 2:
            power_spectrum = self.compute_stacked_power_spectrum(norm=norm)
        else:
            power_spectrum = self.compute_power_spectrum(norm=norm)

        self.freqs, self.freq_widths, self.powers, self.power_errors = power_spectrum

    def compute_power_spectrum(self, times=None, rates=None, norm=True):
        """
        Compute the power spectrum for a single light curve.

        Applies the FFT to the light curve and optionally normalizes the result
        to variance (PSD) units. If binning is enabled, returns binned power.

        Parameters
        ----------
        times : array-like, optional
            Time array to use (defaults to internal value).
        
        rates : array-like, optional
            Rate array to use (defaults to internal value).
        
        norm : bool, optional
            Whether to normalize to variance units.

        Returns
        -------
        freqs : array-like
            Frequencies of the power spectrum.
        
        freq_widths : array-like or None
            Bin widths (if binned).
        
        powers : array-like
            Power spectrum values.
        
        power_errors : array-like or None
            Power spectrum uncertainties (if binned).
        """

        times = self.times if times is None else times
        rates = self.rates if rates is None else rates
        length = len(rates)

        freqs, fft = LightCurve(times=times, rates=rates).fft()
        powers = np.abs(fft) ** 2

        # Filter frequencies within [fmin, fmax]
        valid_mask = (freqs >= self.fmin) & (freqs <= self.fmax)
        freqs = freqs[valid_mask]
        powers = powers[valid_mask]

        if norm:
            powers /= length * np.mean(rates) ** 2 / (2 * self.dt)

        # Apply binning
        if self.num_bins or self.bin_edges:
            
            if self.bin_edges:
                bin_edges = FrequencyBinning.define_bins(self.fmin, self.fmax, num_bins=self.num_bins, 
                                                         bin_type=self.bin_type, bin_edges=self.bin_edges
                                                        )

            elif self.num_bins:
                bin_edges = FrequencyBinning.define_bins(self.fmin, self.fmax, num_bins=self.num_bins, bin_type=self.bin_type)

            else:
                raise ValueError("Either num_bins or bin_edges must be provided.\n"
                                 "In other words, you must specify the number of bins or the bin edges.")

            binned_power = FrequencyBinning.bin_data(freqs, powers, bin_edges)
            freqs, freq_widths, powers, power_errors = binned_power
        else:
            freq_widths, power_errors = None, None

        return freqs, freq_widths, powers, power_errors

    def compute_stacked_power_spectrum(self, norm=True):
        """
        Compute power spectrum for each GP sample and return the mean and std.
        This method is used automatically when a GP model with samples is passed.

        Parameters
        ----------
        norm : bool, optional
            Whether to normalize to variance units.

        Returns
        -------
        freqs : array-like
            Frequencies of the power spectrum.
        
        freq_widths : array-like
            Widths of frequency bins.
        
        power_mean : array-like
            Mean power spectrum values.
        
        power_std : array-like
            Standard deviation of power values across realizations.
        """

        powers = []
        for i in range(self.rates.shape[0]):
            power_spectrum = self.compute_power_spectrum(self.times, self.rates[i], norm=norm)
            freqs, freq_widths, power, _ = power_spectrum
            powers.append(power)

        # Stack the collected powers and errors
        powers = np.vstack(powers)
        power_mean = np.mean(powers, axis=0)
        power_std = np.std(powers, axis=0)

        return freqs, freq_widths, power_mean, power_std
    
    def fit(self, model_type='powerlaw', initial_params=None, lr=1e-3, max_iter=5000, tol=1e-6):
        r"""
        Fit the binned power spectrum using a maximum likelihood approach based on the Gamma distribution.

        This method assumes that each binned PSD value represents the average of M independent
        chi-squared-distributed powers (DOF=2), resulting in a Gamma distribution (DOF=2 chi-squared
        is an exponential, which is a Gamma, and the sum of exponentials is also a Gamma).
        The fit is performed by maximizing the corresponding Gamma likelihood.

        Supported models:
        - 'powerlaw':  
          $$
          P(f) = N \\cdot f^{-\\alpha}
          $$

        - 'powerlaw_lorentzian':  
          $$
          P(f) = N \\cdot f^{-\\alpha} + \\frac{R^2 \\cdot \\Delta / \\pi}{(f - f_0)^2 + \\Delta^2}
          $$  
          where \( R \) is the fractional rms amplitude of the QPO, \( f_0 \) is the central frequency,
          and \( \\Delta \) is the half-width at half-maximum (HWHM) of the Lorentzian.

        The best-fit model type and parameters are stored as class attributes:
        - self.model_type : str  
            Name of the fitted model ('powerlaw' or 'powerlaw_lorentzian')
        - self.model_params : array-like  
            Optimized model parameters.

        Parameters
        ----------
        model_type : str, optional  
            Type of model to fit: 'powerlaw' or 'powerlaw_lorentzian' (default: 'powerlaw').

        initial_params : list of float, optional  
            Initial guess for the model parameters. If None, reasonable defaults are chosen.

        lr : float, optional  
            Learning rate for the PyTorch Adam optimizer (default: 1e-3).

        max_iter : int, optional  
            Maximum number of gradient descent steps to run (default: 5000).

        tol : float, optional  
            Convergence tolerance on the change in negative log-likelihood (default: 1e-8).

        Returns
        -------
        result : dict  
            Dictionary with the following keys:
            - 'params': array-like, best-fit model parameters  
            - 'log_likelihood': float, maximum log-likelihood value at the solution
        """

        if self.freq_widths is not None:
            dof = 2 * self.count_frequencies_in_bins()
        else:
            dof = 2 * np.ones_like(self.freqs)

        freqs = torch.tensor(self.freqs, dtype=torch.float64)
        powers = torch.tensor(self.powers, dtype=torch.float64)
        k = torch.tensor(dof / 2, dtype=torch.float64)

        if initial_params is None:
            alpha_init = 2
            N_init = self.powers[0] / self.freqs[0] ** (-alpha_init)

        if model_type == 'powerlaw':
            if initial_params is None:
                initial_params = [N_init, alpha_init]

            log_N = torch.tensor(np.log(initial_params[0]), dtype=torch.float64, requires_grad=True)
            alpha = torch.tensor(initial_params[1], dtype=torch.float64, requires_grad=True)
            params = [log_N, alpha]

        elif model_type == 'powerlaw_lorentzian':
            if initial_params is None:
                R_init = np.std(self.powers)
                f0_init = np.median(self.freqs)
                delta_init = 0.1 * f0_init
                initial_params = [N_init, alpha_init, R_init, f0_init, delta_init]

            log_N = torch.tensor(np.log(initial_params[0]), dtype=torch.float64, requires_grad=True)
            alpha = torch.tensor(initial_params[1], dtype=torch.float64, requires_grad=True)
            log_R = torch.tensor(np.log(initial_params[2]), dtype=torch.float64, requires_grad=True)
            f0 = torch.tensor(initial_params[3], dtype=torch.float64, requires_grad=True)
            log_delta = torch.tensor(np.log(initial_params[4]), dtype=torch.float64, requires_grad=True)
            params = [log_N, alpha, log_R, f0, log_delta]

        else:
            raise ValueError(f"Unsupported model type '{model_type}'.")

        optimizer = optim.Adam(params, lr=lr)
        prev_loss = None

        for _ in range(max_iter):
            optimizer.zero_grad()

            N = torch.exp(log_N)
            mu = N * freqs ** (-alpha)

            if model_type == 'powerlaw_lorentzian':
                R = torch.exp(log_R)
                delta = torch.exp(log_delta)
                lorentz = (R**2 * delta / np.pi) / ((freqs - f0)**2 + delta**2)
                mu = mu + lorentz

            mu = torch.clamp(mu, min=1e-12)

            logL = (
                k * torch.log(k / mu)
                + (k - 1) * torch.log(powers)
                - (k * powers / mu)
                - torch.tensor(gammaln(dof / 2), dtype=torch.float64)
            )
            loss = -torch.sum(logL)
            loss.backward()
            optimizer.step()

            if prev_loss is not None and abs(prev_loss - loss.item()) < tol:
                break
            prev_loss = loss.item()

        self.model_type = model_type
        if model_type == 'powerlaw':
            self.model_params = [torch.exp(log_N).item(), alpha.item()]
        else:
            self.model_params = [
                torch.exp(log_N).item(),
                alpha.item(),
                torch.exp(log_R).item(),
                f0.item(),
                torch.exp(log_delta).item()
            ]

        return {
            'params': self.model_params,
            'log_likelihood': -loss.item()
        }
 
    def plot(self, step=False):
        """
        Plot the PSD and (if available) the best-fit model.

        Parameters
        ----------
        step : bool, optional
            If True, plot the unbinned PSD as a step function instead of points (default: False).
        """

        fig, ax = plt.subplots(figsize=(8, 4.5))

        if step:
            # Construct step function manually
            f = self.freqs
            p = self.powers
            f_step = np.repeat(f, 2)[1:]
            p_step = np.repeat(p, 2)[:-1]
            ax.plot(f_step, p_step, drawstyle='steps-pre', color='black', lw=1.5)
        else:
            ax.errorbar(
                self.freqs, self.powers,
                xerr=self.freq_widths,
                yerr=self.power_errors,
                fmt='o', ms=3, lw=1.5,
                color='black',
            )

        # Overlay best-fit model if user previously ran .fit()
        if hasattr(self, 'model_type') and hasattr(self, 'model_params'):
            freqs = np.linspace(self.freqs[0], self.freqs[-1], 1000)
            params = self.model_params

            if self.model_type == 'powerlaw':
                N, alpha = params
                model_vals = N * freqs**(-alpha)

            elif self.model_type == 'powerlaw_lorentzian':
                N, alpha, R, f0, delta = params
                lorentz = (R**2 * delta / np.pi) / ((freqs - f0)**2 + delta**2)
                model_vals = N * freqs**(-alpha) + lorentz

            ax.plot(freqs, model_vals, linestyle='--', color='orange')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Frequency', fontsize=12)

        ylabel = r"Power [$(\mathrm{rms}/\mathrm{mean})^2\,\mathrm{freq}^{-1}$]" \
            if self.norm else r"Power [$\mathrm{flux}^2\,\mathrm{freq}^{-1}$]"
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.tick_params(
            which='both', direction='in', length=6,
            width=1, top=True, right=True, labelsize=12
        )
        plt.show()

    def count_frequencies_in_bins(self, fmin=None, fmax=None, num_bins=None, bin_type=None, bin_edges=[]):
        """
        Counts the number of frequencies in each frequency bin.
        Wrapper method to use FrequencyBinning.count_frequencies_in_bins with class attributes.
        """

        return FrequencyBinning.count_frequencies_in_bins(
            self, fmin=fmin, fmax=fmax, num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges
        )