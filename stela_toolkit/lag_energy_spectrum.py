import numpy as np

from ._check_inputs import _CheckInputs
from .coherence import Coherence
from .cross_spectrum import CrossSpectrum
from .frequency_binning import FrequencyBinning
from .plot import Plotter


class LagEnergySpectrum():
    """
    """
    def __init__(self,
                 lightcurves_or_models1,
                 lightcurves_or_models2,
                 fmin,
                 fmax,
                 bin_edges=[],
                 subtract_coh_bias=True,
                 poisson_stats=False,
                 plot_les=False,
                 ):
        # To do: update main docstring for lag interpretation, coherence plotting
        # To do: ensure that input is a list of arrays or lists (lcs) or models (objects).

        # leave main input check to LagFrequencySpectrum. Just check same input dimensions for now.
        if len(lightcurves_or_models1) != len(lightcurves_or_models2):
            raise ValueError("The lightcurves_or_models arrays must contain the sane number of lightcurve/model objects.")

        self.energies = [np.mean(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges[:-1]))]
        self.energy_widths = np.diff(bin_edges) / 2

        self.fmin, self.fmax = fmin, fmax
        self.lags, self.lag_errors = self.compute_lag_spectrum(subtract_coh_bias=subtract_coh_bias, 
                                                               poisson_stats=poisson_stats
                                                               )

        if plot_les:
            self.plot()

    def compute_lag_spectrum(self):
        


        return lags, lag_errors

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