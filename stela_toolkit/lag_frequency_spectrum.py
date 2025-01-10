import numpy as np

from ._check_inputs import _CheckInputs
from .coherence import Coherence
from .cross_spectrum import CrossSpectrum
from .data_loader import LightCurve
from .frequency_binning import FrequencyBinning
from .plot import Plotter


class LagFrequencySpectrum():
    """
    """

    def __init__(self,
                 lightcurve_or_model1,
                 lightcurve_or_model2,
                 fmin='auto',
                 fmax='auto',
                 num_bins=None,
                 bin_type="log",
                 bin_edges=[],
                 subtract_coh_bias=True,
                 poisson_stats=False,
                 plot_lfs=False,
                 ):
        # To do: update main docstring for lag interpretation, add coherence in plotting !!
        input_data = _CheckInputs._check_lightcurve_or_model(lightcurve_or_model1)
        if input_data['type'] == 'model':
            self.times1, self.rates1 = input_data['data']
        else:
            self.times1, self.rates1, _ = input_data['data']

        input_data = _CheckInputs._check_lightcurve_or_model(lightcurve_or_model2)
        if input_data['type'] == 'model':
            self.times2, self.rates2 = input_data['data']
        else:
            self.times2, self.rates2, _ = input_data['data']

        _CheckInputs._check_input_bins(num_bins, bin_type, bin_edges)

        if not np.allclose(self.times1, self.times2):
            raise ValueError("The time arrays of the two light curves must be identical.")

        # Use absolute min and max frequencies if set to 'auto'
        self.dt = np.diff(self.times1)[0]
        # Account for floating point discrepancies
        self.fmin = 1 / (max(self.times1) - min(self.times1)) - 1e-10 if fmin == 'auto' else fmin - 1e-10
        self.fmax = 1 / (2 * self.dt) if fmax == 'auto' else fmax  # nyquist frequency

        self.num_bins = num_bins
        self.bin_type = bin_type
        self.bin_edges = bin_edges

        if len(self.rates1.shape) == 2 and len(self.rates2.shape) == 2:
            lag_spectrum = self.compute_stacked_lag_spectrum()
        else:
            lag_spectrum = self.compute_lag_spectrum(subtract_coh_bias=subtract_coh_bias,
                                                     poisson_stats=poisson_stats
                                                     )

        self.freqs, self.freq_widths, self.lags, self.lag_errors = lag_spectrum

        if plot_lfs:
            self.plot()

    def compute_lag_spectrum(self, times1=None, rates1=None, times2=None, rates2=None,
                             compute_errors=True, subtract_coh_bias=True, poisson_stats=False):
        """
        Computes the lag spectrum for the given light curves.

        Parameters:
        - times1, rates1 (array-like, optional): Time and rates for the first light curve.
        - times2, rates2 (array-like, optional): Time and rates for the second light curve.
        - fmin (float or 'auto', optional): Minimum frequency for computation.
        - fmax (float or 'auto', optional): Maximum frequency for computation.
        - num_bins (int, optional): Number of bins for frequency binning.
        - bin_type (str, optional): Type of binning ('log' or 'linear').
        - bin_edges (array-like, optional): Predefined edges for frequency bins.
        - subtract_noise_bias (bool, optional): Whether to subtract noise bias.
        - poisson_stats (bool, optional): Whether to assume Poisson noise statistics.
        - compute_errors (bool, optional): Whether to compute uncertainties for lag values.

        Returns:
        - freqs (array-like): Frequencies of the lag spectrum.
        - freq_widths (array-like): Bin widths of the frequencies.
        - lags (array-like): Computed lag values.
        - lag_errors (array-like): Uncertainty of the lag values.
        """
        times1 = self.times1 if times1 is None else times1
        rates1 = self.rates1 if rates1 is None else rates1
        times2 = self.times2 if times2 is None else times2
        rates2 = self.rates2 if rates2 is None else rates2

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

        if compute_errors:
            coherence = Coherence(lc1, lc2,
                                  fmin=self.fmin, fmax=self.fmax,
                                  num_bins=self.num_bins, bin_type=self.bin_type, bin_edges=self.bin_edges,
                                  subtract_noise_bias=subtract_coh_bias, poisson_stats=poisson_stats
                                  )

            phase_errors = np.sqrt(
                (1 - coherence.cohs) / (2 * coherence.cohs)
            )

            lag_errors = phase_errors / (2 * np.pi * cross_spectrum.freqs)
        else:
            lag_errors = np.zeros_like(lags)

        return cross_spectrum.freqs, cross_spectrum.freq_widths, lags, lag_errors

    def compute_stacked_lag_spectrum(self):
        """
        Computes the lag spectrum for multiple realizations.

        For multiple realizations (e.g., Gaussian process samples), this method 
        computes the lag spectrum for each realization pair, averages the results, 
        and calculates the standard deviation.

        Parameters:
        - fmin (float or 'auto', optional): Minimum frequency for computation.
        - fmax (float or 'auto', optional): Maximum frequency for computation.
        - num_bins (int, optional): Number of bins for frequency binning.
        - bin_type (str, optional): Type of binning ('log' or 'linear').
        - bin_edges (array-like, optional): Predefined edges for frequency bins.

        Returns:
        - freqs (array-like): Frequencies of the lag spectrum.
        - freq_widths (array-like): Bin widths of the frequencies.
        - lags_mean (array-like): Mean lag values for each frequency bin.
        - lags_std (array-like): Standard deviation of lag values.
        """
        # Compute lag spectrum for each pair of realizations
        lag_spectra = []
        for i in range(self.rates1.shape[0]):
            lag_spectrum = self.compute_lag_spectrum(times1=self.times1, rates1=self.rates1[i],
                                                     times2=self.times2, rates2=self.rates2[i],
                                                     compute_errors=False
                                                     )
            lag_spectra.append(lag_spectrum[2])

        # Stack lag spectra
        lag_spectra = np.mean(lag_spectra, axis=0)
        lag_spectra_mean = np.mean(lag_spectra, axis=0)
        lag_spectra_std = np.std(lag_spectra, axis=0)

        freqs, freq_widths = lag_spectrum[0], lag_spectrum[1]

        return freqs, freq_widths, lag_spectra_mean, lag_spectra_std

    def plot(self, freqs=None, freq_widths=None, lags=None, lag_errors=None, **kwargs):
        """
        Plots the cross-spectrum.

        Parameters:
        - **kwargs: Keyword arguments for customizing the plot.
        """
        freqs = self.freqs if freqs is None else freqs
        freq_widths = self.freq_widths if freq_widths is None else freq_widths
        lags = self.lags if lags is None else lags
        lag_errors = self.lag_errors if lag_errors is None else lag_errors

        kwargs.setdefault('xlabel', 'Frequency')
        kwargs.setdefault('ylabel', 'Time Lags')
        kwargs.setdefault('xscale', 'log')
        kwargs.setdefault('yscale', 'log')
        Plotter.plot(
            x=freqs, y=lags, xerr=freq_widths, yerr=lag_errors, **kwargs
        )

    def count_frequencies_in_bins(self, fmin=None, fmax=None, num_bins=None, bin_type="log", bin_edges=[]):
        """
        Counts the number of frequencies in each bin for the power spectrum.
        Wrapper method to use FrequencyBinning.count_frequencies_in_bins with class attributes.

        If 
        Parameters:
        - fmin (float): Minimum frequency (optional).
        - fmax (float): Maximum frequency (optional).
            *** Class attributes will be used if not specified.
        - num_bins (int): Number of bins to create (if bin_edges is not provided).
        - bin_type (str): Type of binning ("log" or "linear").
        - bin_edges (array-like): Custom array of bin edges (optional).
            *** Class attributes will be used if not specified.

        Returns:
        - bin_counts (list): List of counts of frequencies in each bin.
        """
        return FrequencyBinning.count_frequencies_in_bins(
            parent=self, fmin=fmin, fmax=fmax, num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges
        )
