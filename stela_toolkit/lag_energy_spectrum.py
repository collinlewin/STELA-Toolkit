import numpy as np
import matplotlib.pyplot as plt

from ._check_inputs import _CheckInputs
from .coherence import Coherence
from .cross_spectrum import CrossSpectrum
from .frequency_binning import FrequencyBinning
from .lag_frequency_spectrum import LagFrequencySpectrum


class LagEnergySpectrum():
    """
    Computes the time lag as a function of energy for multiple energy bands using 
    light curves or trained models. If models are provided, it will detect and use 
    the most recently generated samples for each energy band.

    A positive lag indicates that the variability in lightcurve/s 1 is lagging 
    that in lightcurve/s 2.

    Parameters:
    - lightcurves_or_models1, lightcurves_or_models2 (list of LightCurve or model objects): 
        Lists of input light curves or models corresponding to different energy bands. 
        Models must have been trained. The most recently generated realizations/samples 
        will be used; if none have been generated, 1000 samples at 1000 time values 
        will be generated.
    - fmin, fmax (float): Minimum and maximum frequencies to integrate over when 
        computing the lag.
    - bin_edges (array-like): Energy bin edges used to define the energy bands.
    - subtract_coh_bias (bool, optional): Whether to subtract the coherence bias. 
    Defaults to True.
    - poisson_stats (bool, optional): Whether to assume Poisson noise statistics. 
    Defaults to False.
    - plot_les (bool, optional): Whether to plot the resulting lag energy spectrum. 
    Defaults to False.

    Attributes:
    - energies (array-like): Mean energies for each energy bin.
    - energy_widths (array-like): Widths of the energy bins.
    - lags (array-like): Computed time lags for each energy bin.
    - lag_errors (array-like): Uncertainties in the time lag values.
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
        # leave main input check to LagFrequencySpectrum, check same input dimensions for now.
        if len(lightcurves_or_models1) != len(lightcurves_or_models2):
            raise ValueError("The lightcurves_or_models arrays must contain the sane number of lightcurve/model objects.")

        self.data_models1 = lightcurves_or_models1
        self.data_models2 = lightcurves_or_models2

        self.energies = [np.mean(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges[:-1]))]
        self.energy_widths = np.diff(bin_edges) / 2

        self.fmin, self.fmax = fmin, fmax
        lag_spectrum = self.compute_lag_spectrum(subtract_coh_bias=subtract_coh_bias, 
                                                 poisson_stats=poisson_stats
                                                 )
        self.lags, self.lag_errors, self.cohs, self. coh_errors = lag_spectrum

        if plot_les:
            self.plot()

    def compute_lag_spectrum(self, subtract_coh_bias, poisson_stats):
        lags, lag_errors, cohs, coh_errors = [], [], [], []
        for i in range(len(self.data_models1)):
            lfs = LagFrequencySpectrum(self.data_models1[i],
                                       self.data_models2[i],
                                       fmin=self.fmin,
                                       fmax=self.fmax,
                                       num_bins=1,
                                       subtract_coh_bias=subtract_coh_bias,
                                       poisson_stats=poisson_stats
                                       )
            lags.append(lfs.lags)
            lag_errors.append(lfs.lag_errors)
            cohs.append(lfs.cohs)
            coh_errors.append(lfs.coh_errors)

        return lags, lag_errors, cohs, coh_errors

    def plot(self, energies=None, energy_widths=None, lags=None, lag_errors=None, cohs=None, coh_errors=None, **kwargs):
        """
        Plots the lag-energy spectrum and coherence in two subplots.

        Parameters:
        - **kwargs: Keyword arguments for customizing the plots.
        """
        energies = self.energies if energies is None else energies
        energy_widths = self.energy_widths if energy_widths is None else energy_widths
        lags = self.lags if lags is None else lags
        lag_errors = self.lag_errors if lag_errors is None else lag_errors
        cohs = self.cohs if cohs is None else cohs
        coh_errors = self.coh_errors if coh_errors is None else coh_errors

        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(8, 6), sharex=True)

        # Lag-energy spectrum
        kwargs.setdefault('xlabel', 'Energy')
        kwargs.setdefault('ylabel', 'Time Lags')
        kwargs.setdefault('xscale', 'log')
        kwargs.setdefault('yscale', 'log')
        ax1.errorbar(
            energies, lags, xerr=energy_widths, yerr=lag_errors, fmt='o', label='Lag-Energy Spectrum'
        )
        ax1.set_xscale(kwargs['xscale'])
        ax1.set_yscale(kwargs['yscale'])
        ax1.set_ylabel(kwargs['ylabel'])
        ax1.legend()
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Coherence spectrum
        if cohs is not None and coh_errors is not None:
            ax2.errorbar(
                energies, cohs, xerr=energy_widths, yerr=coh_errors, fmt='o', color='orange', label='Coherence'
            )
            ax2.set_xscale(kwargs['xscale'])
            ax2.set_ylabel('Coherence')
            ax2.legend()
            ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

        fig.text(0.5, 0.04, kwargs['xlabel'], ha='center', va='center')
        plt.tight_layout()
        plt.show()

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