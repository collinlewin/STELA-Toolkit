import numpy as np

from .check_inputs import _CheckInputs
from .coherence import Coherence
from .cross_spectrum import CrossSpectrum
from .frequency_binning import FrequencyBinning
from .plotter import Plotter


class LagFrequencySpectrum():
    """
    Computes the lag-frequency spectrum for time series data.

    This class calculates the lag-frequency spectrum for one or more realizations of
    time series data. It supports single and stacked realizations, binning of 
    frequencies, and optional plotting of results.

    Parameters:
    - times1 (array-like, optional): Time points for the first time series.
    - values1 (array-like, optional): Values for the first time series.
    - times2 (array-like, optional): Time points for the second time series.
    - values2 (array-like, optional): Values for the second time series.
    - timeseries1 (object, optional): First time series object (overrides times1/values1).
    - timeseries2 (object, optional): Second time series object (overrides times2/values2).
    - fmin (float or 'auto', optional): Minimum frequency for computation.
    - fmax (float or 'auto', optional): Maximum frequency for computation.
    - num_bins (int, optional): Number of bins for frequency binning.
    - bin_type (str, optional): Type of binning ('log' or 'linear').
    - bin_edges (array-like, optional): Predefined edges for frequency bins.
    - subtract_coherence_bias (bool, optional): Whether to subtract the coherence bias.
    - poisson_stats (bool, optional): Whether to assume Poisson noise statistics.
    - plot_lfs (bool, optional): Whether to automatically plot the lag-frequency spectrum.

    Key Attributes:
    - freqs (array-like): Frequencies of the lag spectrum.
    - freq_widths (array-like): Bin widths of the frequencies.
    - lags (array-like): Computed lag values for each frequency bin.
    - lag_sigmas (array-like): Uncertainty of the lag values.
    """
    def __init__(self,
                 times1=[],
                 values1=[],
                 times2=[],
                 values2=[],
                 timeseries1=None,
                 timeseries2=None,
                 fmin='auto',
                 fmax='auto',
                 num_bins=None,
                 bin_type="log",
                 bin_edges=[],
                 subtract_coherence_bias=True,
                 poisson_stats=False,
                 plot_lfs=False
                 ):
        # To do: update main docstring for lag interpretation
        self.times1, self.values1 = self._check_input(timeseries1, times1, values1)
        self.times2, self.values2 = self._check_input(timeseries2, times2, values2)
        _CheckInputs._check_input_bins(num_bins, bin_type, bin_edges)

        if not np.allclose(self.times1, self.times2):
            raise ValueError("The time arrays of the two time series must be identical.")

        # Use absolute min and max frequencies if set to 'auto'
        self.dt = np.diff(self.times)[0]
        self.fmin = 1 / (self.times.max() - self.times.min()) if fmin == 'auto' else fmin
        self.fmax = 1 / (2 * self.dt) if fmax == 'auto' else fmax # nyquist frequency

        self.num_bins = num_bins
        self.bin_type = bin_type
        self.bin_edges = bin_edges

        if len(self.values1.shape) == 2 and len(self.values2.shape) == 2:
            lag_spectrum = self.compute_stacked_lag_spectrum(
                fmin=self.fmin, fmax=self.fmax, num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges
            )
        else:
            lag_spectrum = self.compute_lag_spectrum(
                fmin=self.fmin, fmax=self.fmax, num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges,
                subtract_coherence_bias=subtract_coherence_bias, poisson_stats=poisson_stats
            )

        self.freqs, self.freq_widths, self.lags, self.lag_sigmas = lag_spectrum

        if plot_lfs:
            self.plot()

    def compute_lag_spectrum(self, times1=None, values1=None, times2=None, values2=None,
                             fmin='auto', fmax='auto', num_bins=None, bin_type="log", 
                             bin_edges=[], subtract_noise_bias=True, poisson_stats=False, 
                             compute_sigmas=True):
        """
        Computes the lag spectrum for the given time series.

        Parameters:
        - times1, values1 (array-like, optional): Time and values for the first time series.
        - times2, values2 (array-like, optional): Time and values for the second time series.
        - fmin (float or 'auto', optional): Minimum frequency for computation.
        - fmax (float or 'auto', optional): Maximum frequency for computation.
        - num_bins (int, optional): Number of bins for frequency binning.
        - bin_type (str, optional): Type of binning ('log' or 'linear').
        - bin_edges (array-like, optional): Predefined edges for frequency bins.
        - subtract_noise_bias (bool, optional): Whether to subtract noise bias.
        - poisson_stats (bool, optional): Whether to assume Poisson noise statistics.
        - compute_sigmas (bool, optional): Whether to compute uncertainties for lag values.

        Returns:
        - freqs (array-like): Frequencies of the lag spectrum.
        - freq_widths (array-like): Bin widths of the frequencies.
        - lags (array-like): Computed lag values.
        - lag_sigmas (array-like): Uncertainty of the lag values.
        """
        times1 = self.times1 if times1 is None else times1
        values1 = self.values1 if values1 is None else values1
        times2 = self.times2 if times2 is None else times2
        values2 = self.values2 if values2 is None else values2

        # Compute the cross spectrum
        cross_spectrum = CrossSpectrum(
            times1=times1, values1=values1, times2=times2, values2=values2
        )
        lags = cross_spectrum.cs / (2 * np.pi * cross_spectrum.freqs)

        if compute_sigmas:
            coherence = Coherence(
                times1=times1, values1=values1, times2=times2, values2=values2,
                fmin=fmin, fmax=fmax, num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges,
                subtract_noise_bias=subtract_noise_bias, poisson_stats=poisson_stats
            )
            
            phase_sigmas = np.sqrt(
                (1 - coherence.cohs) / (2 * coherence.cohs)
            )
            lag_sigmas = phase_sigmas / (2 * np.pi * cross_spectrum.freqs)
        else:
            lag_sigmas = np.zeros_like(lags)

        return cross_spectrum.freqs, cross_spectrum.freq_widths, lags, lag_sigmas

    def compute_stacked_lag_spectrum(self, fmin='auto', fmax='auto', num_bins=None, 
                                     bin_type="log", bin_edges=[]):
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
        for i in range(self.values1.shape[0]):
            lag_spectrum = self.compute_lag_spectrum(
                times1=self.times1, values1=self.values1[i], times2=self.times2, values2=self.values2[i],
                fmin=fmin, fmax=fmax, num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges,
                compute_sigmas=False
            )
            lag_spectra.append(lag_spectrum[2])

        # Stack lag spectra
        lag_spectra = np.mean(lag_spectra, axis=0)
        lag_spectra_mean = np.mean(lag_spectra, axis=0)
        lag_spectra_std = np.std(lag_spectra, axis=0)

        freqs, freq_widths = lag_spectrum[0], lag_spectrum[1]

        return freqs, freq_widths, lag_spectra_mean, lag_spectra_std

    def plot(self, freqs=None, freq_widths=None, lags=None, lag_sigmas=None, **kwargs):
        """
        Plots the cross-spectrum.

        Parameters:
        - **kwargs: Keyword arguments for customizing the plot.
        """
        freqs = self.freqs if freqs is None else freqs
        freq_widths = self.freq_widths if freq_widths is None else freq_widths
        lags = self.lags if lags is None else lags
        lag_sigmas = self.lag_sigmas if lag_sigmas is None else lag_sigmas

        kwargs.setdefault('xlabel', 'Frequency')
        kwargs.setdefault('ylabel', 'Time Lags')
        kwargs.setdefault('xscale', 'log')
        kwargs.setdefault('yscale', 'log')
        Plotter.plot(
            x=freqs, y=lags, xerr=freq_widths, yerr=lag_sigmas, **kwargs
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