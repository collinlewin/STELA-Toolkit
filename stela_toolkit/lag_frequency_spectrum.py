import numpy as np
import matplotlib.pyplot as plt
from ._check_inputs import _CheckInputs
from ._clarify_warnings import _ClearWarnings
from .coherence import Coherence
from .cross_spectrum import CrossSpectrum
from .data_loader import LightCurve
from .frequency_binning import FrequencyBinning


class LagFrequencySpectrum:
    """
    Compute the time lag as a function of frequency between two time series.

    This class computes the lag-frequency spectrum using either:
    - Two `LightCurve` objects (with regularly sampled time arrays), or
    - Two trained `GaussianProcess` models with generated posterior samples.

    If GP models are passed as inputs, the most recently generated samples are used.
    If none exist, the toolkit will generate 1000 samples on a 1000-point grid by default.

    A **positive lag** means that the first input (`lc_or_model1`) lags behind the 
    second/reference band (`lc_or_model2`).

    Uncertainties are estimated using:
    - **Analytical propagation** from coherence if inputs are light curves.
    - **Empirical variance** across posterior samples if inputs are Gaussian Process realizations.

    Parameters
    ----------
    lc_or_model1 : LightCurve or GaussianProcess
        First light curve or GP model.

    lc_or_model2 : LightCurve or GaussianProcess
        Second/reference light curve or GP model (must match shape of `lc_or_model1`).

    fmin : float or 'auto', optional
        Minimum frequency for the lag spectrum. If 'auto', uses the lowest nonzero FFT frequency.

    fmax : float or 'auto', optional
        Maximum frequency for the lag spectrum. If 'auto', uses the Nyquist frequency.

    num_bins : int, optional
        Number of frequency bins to use (ignored if `bin_edges` is given).

    bin_type : str, optional
        Type of frequency binning: "log" or "linear" (default: "log").

    bin_edges : array-like, optional
        Custom frequency bin edges (overrides `num_bins` and `bin_type` if provided).

    subtract_coh_bias : bool, optional
        Whether to subtract Poisson noise bias from the coherence estimate (default: True).

    Attributes
    ----------
    freqs : ndarray
        Center frequency of each bin.

    freq_widths : ndarray
        Width of each frequency bin.

    lags : ndarray
        Time lag values at each frequency.

    lag_errors : ndarray
        Uncertainties on the lag values.

    cohs : ndarray
        Coherence values at each frequency.

    coh_errors : ndarray
        Uncertainties on the coherence values.
    """

    def __init__(self,
                 lc_or_model1,
                 lc_or_model2,
                 fmin='auto',
                 fmax='auto',
                 num_bins=None,
                 bin_type="log",
                 bin_edges=[],
                 subtract_coh_bias=True):
        
        input_data = _CheckInputs._check_lightcurve_or_model(lc_or_model1)
        if input_data['type'] == 'model':
            self.times1, self.rates1 = input_data['data']
        else:
            self.times1, self.rates1, _ = input_data['data']

        input_data = _CheckInputs._check_lightcurve_or_model(lc_or_model2)
        if input_data['type'] == 'model':
            self.times2, self.rates2 = input_data['data']
        else:
            self.times2, self.rates2, _ = input_data['data']

        _CheckInputs._check_input_bins(num_bins, bin_type, bin_edges)

        if not np.allclose(self.times1, self.times2):
            raise ValueError("The time arrays of the two light curves must be identical.")

        # Use absolute min and max frequencies if set to 'auto'
        self.dt = np.diff(self.times1)[0]
        self.fmin = np.fft.rfftfreq(len(self.rates1), d=self.dt)[1] if fmin == 'auto' else fmin
        self.fmax = np.fft.rfftfreq(len(self.rates1), d=self.dt)[-1] if fmax == 'auto' else fmax  # nyquist frequency

        self.num_bins = num_bins
        self.bin_type = bin_type
        self.bin_edges = bin_edges

        if len(self.rates1.shape) == 2 and len(self.rates2.shape) == 2:
            lag_spectrum = self.compute_stacked_lag_spectrum()
        else:
            lag_spectrum = self.compute_lag_spectrum(subtract_coh_bias=subtract_coh_bias)

        self.freqs, self.freq_widths, self.lags, self.lag_errors, self.cohs, self.coh_errors = lag_spectrum

    def compute_lag_spectrum(self, 
                             times1=None, rates1=None,
                             times2=None, rates2=None,
                             subtract_coh_bias=True):
        """
        Compute the lag spectrum for a single pair of light curves or model realizations.

        The phase of the cross-spectrum is converted to time lags, and uncertainties are
        computed either from coherence (for raw light curves) or from GP sampling (if using
        stacked realizations).

        Parameters
        ----------
        times1 : array-like, optional
            Time values for the first time series.

        rates1 : array-like, optional
            Rate or flux values for the first time series.

        times2 : array-like, optional
            Time values for the second time series.

        rates2 : array-like, optional
            Rate or flux values for the second time series.

        subtract_coh_bias : bool, optional
            Whether to subtract noise bias from the coherence estimate.

        Returns
        -------
        freqs : array-like
            Center of each frequency bin.

        freq_widths : array-like
            Width of each frequency bin.

        lags : array-like
            Time lag values at each frequency.

        lag_errors : array-like
            Uncertainties on the lag values.

        cohs : array-like
            Coherence values at each frequency.

        coh_errors : array-like
            Uncertainties on the coherence values.
        """

        times1 = times1 if times1 is not None else self.times1
        times2 = times2 if times2 is not None else self.times2
        rates1 = rates1 if rates1 is not None else self.rates1
        rates2 = rates2 if rates2 is not None else self.rates2 

        lc1 = LightCurve(times=times1, rates=rates1)
        lc2 = LightCurve(times=times2, rates=rates2)

        # Compute the cross spectrum
        cross_spectrum = CrossSpectrum(lc1, lc2,
                                       fmin=self.fmin, fmax=self.fmax,
                                       num_bins=self.num_bins, bin_type=self.bin_type,
                                       bin_edges=self.bin_edges,
                                       norm=False
                                    )

        lags = np.angle(cross_spectrum.cs) / (2 * np.pi * cross_spectrum.freqs)

        coherence = Coherence(lc1, lc2,
                              fmin=self.fmin, fmax=self.fmax,
                              num_bins=self.num_bins, bin_type=self.bin_type, bin_edges=self.bin_edges,
                              subtract_noise_bias=subtract_coh_bias
                            )    
        cohs = coherence.cohs
        coh_errors = coherence.coh_errors

        num_freq = self.count_frequencies_in_bins()

        phase_errors = _ClearWarnings.run(
            lambda: np.sqrt((1 - coherence.cohs) / (2 * coherence.cohs * num_freq)),
            explanation="Error from sqrt when computing (unbinned) phase errors here is common "
                        "and typically due to >1 coherence at the minimum frequency."
        )

        lag_errors = phase_errors / (2 * np.pi * cross_spectrum.freqs)

        return cross_spectrum.freqs, cross_spectrum.freq_widths, lags, lag_errors, cohs, coh_errors

    def compute_stacked_lag_spectrum(self):
        """
        Compute lag-frequency spectrum for stacked GP samples.

        This method assumes the input light curves are model-generated and include
        multiple realizations. Returns mean and standard deviation of lag and coherence.

        Returns
        -------
        freqs : array-like
            Frequency bin centers.
        
        freq_widths : array-like
            Frequency bin widths.
        
        lags : array-like
            Mean time lags across samples.
        
        lag_errors : array-like
            Standard deviation of lags.
        
        cohs : array-like
            Mean coherence values.
        
        coh_errors : array-like
            Standard deviation of coherence values.
        """

        # Compute lag spectrum for each pair of realizations
        lag_spectra = []
        coh_spectra = []
        for i in range(self.rates1.shape[0]):
            lag_spectrum = self.compute_lag_spectrum(times1=self.times1, rates1=self.rates1[i],
                                                     times2=self.times2, rates2=self.rates2[i],
                                                     subtract_coh_bias=False
                                                    )
            lag_spectra.append(lag_spectrum[2])
            coh_spectra.append(lag_spectrum[4])

        # Average lag spectra
        lag_spectra_mean = np.mean(lag_spectra, axis=0)
        lag_spectra_std = np.std(lag_spectra, axis=0)

        # Average coherence spectra
        coh_spectra_mean = np.mean(coh_spectra, axis=0)
        coh_spectra_std = np.std(coh_spectra, axis=0)

        freqs, freq_widths = lag_spectrum[0], lag_spectrum[1]

        return freqs, freq_widths, lag_spectra_mean, lag_spectra_std, coh_spectra_mean, coh_spectra_std

    def plot(self, **kwargs):
        """
        Plot the lag-frequency and coherence spectrum.
        Plot appearance can be customized using keyword arguments:
        
        - figsize : tuple, optional
            Size of the figure (default: (8, 6)).
        - xlabel : str, optional
            Label for the x-axis (default: "Frequency").
        - ylabel : str, optional
            Label for the y-axis of the lag panel (default: "Time Lag").
        - xscale : str, optional
            Scale for the x-axis ("log" or "linear", default: "log").
        - yscale : str, optional
            Scale for the y-axis of the lag panel ("linear" or "log", default: "linear").
        """
        figsize = kwargs.get('figsize', (8, 6))
        xlabel = kwargs.get('xlabel', 'Frequency')
        ylabel = kwargs.get('ylabel', 'Time Lag')
        xscale = kwargs.get('xscale', 'log')
        yscale = kwargs.get('yscale', 'linear')

        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=figsize, sharex=True)
        plt.subplots_adjust(hspace=0.05)

        # Lag-frequency spectrum
        ax1.errorbar(self.freqs, self.lags, xerr=self.freq_widths, yerr=self.lag_errors,
                    fmt='o', color='black', ms=3, lw=1.5)
        ax1.set_xscale(xscale)
        ax1.set_yscale(yscale)
        ax1.set_ylabel(ylabel, fontsize=12)
        ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax1.tick_params(which='both', direction='in', length=6, width=1, top=True, right=True, labelsize=12)

        # Coherence spectrum
        if self.cohs is not None and self.coh_errors is not None:
            ax2.errorbar(self.freqs, self.cohs, xerr=self.freq_widths, yerr=self.coh_errors,
                        fmt='o', color='black', ms=3, lw=1.5)
            ax2.set_xscale(xscale)
            ax2.set_ylabel('Coherence', fontsize=12)
            ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            ax2.tick_params(which='both', direction='in', length=6, width=1, top=True, right=True, labelsize=12)

        fig.text(0.5, 0.04, xlabel, ha='center', va='center', fontsize=12)
        plt.show()

    def count_frequencies_in_bins(self, fmin=None, fmax=None, num_bins=None, bin_type=None, bin_edges=[]):
        """
        Counts the number of frequencies in each frequency bin.
        Wrapper method to use FrequencyBinning.count_frequencies_in_bins with class attributes.
        """

        return FrequencyBinning.count_frequencies_in_bins(
            self, fmin=fmin, fmax=fmax, num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges
        )