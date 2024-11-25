import numpy as np

from .check_inputs import _CheckInputs
from .coherence import Coherence
from .cross_spectrum import CrossSpectrum
from .frequency_binning import FrequencyBinning
from .plotter import Plotter


class LagFrequencySpectrum():
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
            lag_spectrum = self.compute_stacked_lag_spectrum()
        else:
            lag_spectrum = self.compute_lag_spectrum()
        self.freqs, self.freq_widths, self.lags, self.lag_sigmas = lag_spectrum

        if plot_lfs:
            self.plot()

    def compute_lag_spectrum(self, fmin='auto', fmax='auto', num_bins=None, 
                                bin_type="log", bin_edges=[]):
        """
        """
        # Compute the cross spectrum
        cross_spectrum = CrossSpectrum(
            times1=self.times1, values1=self.values1, times2=self.times2, values2=self.values2
        )
        lags = cross_spectrum.cs / (2 * np.pi * cross_spectrum.freqs)
        #lag_sigmas = # need coherence to compute this

        return cross_spectrum.freqs, cross_spectrum.freq_widths, lags, lag_sigmas

    def compute_stacked_lag_spectrum(self, fmin='auto', fmax='auto', num_bins=None,
                                        bin_type="log", bin_edges=[]):
        """
        """
        # Compute lag spectrum for each pair of realizations
        lag_spectra = []
        for i in range(self.values1.shape[0]):
            cross_spectrum = CrossSpectrum(
                times1=self.times1, values1=self.values1[i], times2=self.times2, values2=self.values2[i],
                fmin=fmin, fmax=fmax, num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges
            )
            lags = cross_spectrum.cs / (2 * np.pi * cross_spectrum.freqs)
            lag_spectra.append(lags)

        # Stack lag spectra
        lag_spectra = np.mean(lag_spectra, axis=0)
        lag_spectra_mean = np.mean(lag_spectra, axis=0)
        lag_spectra_std = np.std(lag_spectra, axis=0)

        return cross_spectrum.freqs, cross_spectrum.freq_widths, lag_spectra_mean, lag_spectra_std

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
        kwargs.setdefault('ylabel', 'Lags')
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