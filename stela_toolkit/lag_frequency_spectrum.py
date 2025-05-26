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

    This class accepts either LightCurve objects (with regular sampling) or trained
    GaussianProcess models from this package. If GP models are provided, the most
    recently generated samples are used. If no samples have been created yet,
    the toolkit will automatically generate 1000 samples on a 1000-point grid.

    The sign convention is such that a **positive lag** indicates that the input provided as
    `lc_or_model1` is **lagging behind** the input provided as `lc_or_model2`.

    There are two modes for computing uncertainties:
    - If the inputs are individual light curves, lag uncertainties are propagated from
      the coherence spectrum using a theoretical error model.
    - If the inputs are GP models, the lag spectrum is computed for each sample and
      uncertainties are reported as the standard deviation across all samples.

    Parameters
    ----------
    lc_or_model1 : LightCurve or GaussianProcess
        First input time series or trained model.
    lc_or_model2 : LightCurve or GaussianProcess
        Second input time series or trained model.
    fmin : float or 'auto', optional
        Minimum frequency to include. If 'auto', uses the lowest nonzero FFT frequency.
    fmax : float or 'auto', optional
        Maximum frequency to include. If 'auto', uses the Nyquist frequency.
    num_bins : int, optional
        Number of frequency bins.
    bin_type : str, optional
        Type of binning: 'log' or 'linear'.
    bin_edges : array-like, optional
        Custom bin edges (overrides `num_bins` and `bin_type`).
    subtract_coh_bias : bool, optional
        Whether to subtract Poisson noise bias from the coherence.
    plot_lfs : bool, optional
        Whether to generate the lag-frequency plot on initialization.

    Attributes
    ----------
    freqs : array-like
        Frequency bin centers.
    freq_widths : array-like
        Bin widths for each frequency bin.
    lags : array-like
        Computed time lags.
    lag_errors : array-like
        Uncertainties in the lag estimates.
    cohs : array-like
        Coherence values for each frequency bin.
    coh_errors : array-like
        Uncertainties in the coherence values.
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
        
        # To do: update main docstring for lag interpretation, add coherence in plotting !!
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
        computed either from coherence (light curves) or from GP sampling (if stacked mode).

        Parameters
        ----------
        times1, rates1 : array-like, optional
            Input time and rates for the first time series.
        times2, rates2 : array-like, optional
            Input time and rates for the second time series.
        subtract_coh_bias : bool, optional
            Whether to subtract noise bias from coherence.

        Returns
        -------
        freqs : array-like
            Frequency bin centers.
        freq_widths : array-like
            Frequency bin widths.
        lags : array-like
            Time lags at each frequency.
        lag_errors : array-like
            Uncertainty in the lag values.
        cohs : array-like
            Coherence values.
        coh_errors : array-like
            Uncertainties in the coherence values.
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

    def plot(self, freqs=None, freq_widths=None, lags=None, lag_errors=None, cohs=None, coh_errors=None, **kwargs):
        """
        Plot the lag-frequency and coherence spectrum.

        Parameters
        ----------
        **kwargs : dict
            Custom plotting arguments (xlabel, xscale, yscale, etc.).
        """

        freqs = self.freqs if freqs is None else freqs
        freq_widths = self.freq_widths if freq_widths is None else freq_widths
        lags = self.lags if lags is None else lags
        lag_errors = self.lag_errors if lag_errors is None else lag_errors
        cohs = self.cohs if cohs is None else cohs
        coh_errors = self.coh_errors if coh_errors is None else coh_errors

        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(8, 6), sharex=True)

        # Lag-frequency spectrum
        kwargs.setdefault('xlabel', 'Frequency')
        kwargs.setdefault('ylabel', 'Time Lags')
        kwargs.setdefault('xscale', 'log')
        kwargs.setdefault('yscale', 'linear')
        ax1.errorbar(
            freqs, lags, xerr=freq_widths, yerr=lag_errors, fmt='o',
        )
        ax1.set_xscale(kwargs['xscale'])
        ax1.set_yscale(kwargs['yscale'])
        ax1.set_ylabel(kwargs['ylabel'])
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Coherence spectrum
        if cohs is not None and coh_errors is not None:
            ax2.errorbar(
                freqs, cohs, xerr=freq_widths, yerr=coh_errors, fmt='o', color='orange',
            )
            ax2.set_xscale(kwargs['xscale'])
            ax2.set_ylabel('Coherence')
            ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

        fig.text(0.5, 0.04, kwargs['xlabel'], ha='center', va='center')
        plt.show()

    def count_frequencies_in_bins(self, fmin=None, fmax=None, num_bins=None, bin_type=None, bin_edges=[]):
        """
        Counts the number of frequencies in each frequency bin.
        Wrapper method to use FrequencyBinning.count_frequencies_in_bins with class attributes.
        """

        return FrequencyBinning.count_frequencies_in_bins(
            self, fmin=fmin, fmax=fmax, num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges
        )