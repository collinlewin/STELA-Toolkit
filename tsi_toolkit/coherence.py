import numpy as np

from ._check_inputs import _CheckInputs
from .power_spectrum import PowerSpectrum
from .cross_spectrum import CrossSpectrum
from .plot import Plotter
from .frequency_binning import FrequencyBinning

class Coherence:
    """
    """
    def __init__(self,
                 times1=[],
                 values1=[],
                 sigmas1=[],
                 times2=[],
                 values2=[],
                 sigmas2=[],
                 timeseries1=None,
                 timeseries2=None,
                 fmin='auto',
                 fmax='auto',
                 num_bins=None,
                 bin_type="log",
                 bin_edges=[],
                 subtract_noise_bias=True,
                 poisson_stats=False,
                 plot_coh=False
                 ):
        # To do: determine if or if not Poisson statistics for the user
        # To do: decrease number of parameters
        self.times1, self.values1 = self._check_input(timeseries1, times1, values1, sigmas1)
        self.times2, self.values2 = self._check_input(timeseries2, times2, values2, sigmas2)
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

        # Check if the input values are for multiple realizations
        # this needs to be corrected for handling different shapes and dim val1 != dim val2
        if len(self.values1.shape) == 2 and len(self.values2.shape) == 2:
            coherence_spectrum = self.compute_stacked_coherence(fmin=self.fmin, fmax=self.fmax,
                                                             num_bins=self.num_bins, bin_type=self.bin_type, 
                                                             bin_edges=bin_edges, subtract_noise_bias=subtract_noise_bias
                                                             )
        else:
            coherence_spectrum = self.compute_coherence(fmin=self.fmin, fmax=self.fmax,
                                                     num_bins=self.num_bins, bin_type=self.bin_type, 
                                                     bin_edges=bin_edges, subtract_noise_bias=subtract_noise_bias
                                                     )
        
        self.freqs, self.freq_widths, self.cohs, self.coh_sigmas = coherence_spectrum

        if plot_coh:
            self.plot()

    def compute_coherence(self, times1=None, values1=None, sigmas1=None,
                          times2=None, values2=None, sigmas2=None,
                          fmin='auto', fmax='auto', num_bins=None, bin_type="log",
                          bin_edges=[], subtract_noise_bias=True, poisson_stats=False):
        """
        """
        times1 = self.times1 if times1 is None else times1
        values1 = self.values1 if values1 is None else values1
        sigmas1 = self.sigmas1 if sigmas1 is None else sigmas1
        times2 = self.times2 if times2 is None else times2
        values2 = self.values2 if values2 is None else values2
        sigmas2 = self.sigmas2 if sigmas2 is None else sigmas2
        
        cross_spectrum = CrossSpectrum(
            times1=times1, values1=values1, times2=times2, values2=values2, 
            fmin=fmin, fmax=fmax, num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges
        )
        power_spectrum1 = PowerSpectrum(
            times=times1, values=values1, fmin=fmin, fmax=fmax,
            num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges
        )
        power_spectrum2 = PowerSpectrum(
            times=times2, values=values2, fmin=fmin, fmax=fmax,
            num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges
        )

        ps1 = power_spectrum1.powers
        ps2 = power_spectrum2.powers
        cs = cross_spectrum.cs

        if subtract_noise_bias:
            mean1 = np.mean(self.values1)
            mean2 = np.mean(self.values2)

            if poisson_stats:
                pnoise1 = 2 * (mean1 + self.bkg1) / mean1 ** 2
                pnoise2 = 2 * (mean2 + self.bkg2) / mean2 ** 2
            else:
                # compute the noise bias using errors
                if sigmas1 is None or sigmas2 is None:
                    raise ValueError("Sigmas must be provided to compute the noise bias.")
                mean_sigma1 = np.mean(sigmas1)
                mean_sigma2 = np.mean(sigmas2)
                nyquist_freq = 1 / (2 * np.diff(times1)[0])
                pnoise1 = mean_sigma1 ** 2 / ( nyquist_freq * mean1 ** 2 )
                pnoise2 = mean_sigma2 ** 2 / ( nyquist_freq * mean2 ** 2 )

            bias = (pnoise2 * (ps1 - pnoise1) + pnoise1 * (ps2 - pnoise2) + pnoise1 * pnoise2)
        else:
            bias = 0

        coherence = ( np.abs(cs) ** 2 - bias ) / ps1 * ps2
        return power_spectrum1.freqs, power_spectrum1.freq_widths, coherence, None

    def compute_stacked_coherence(self, fmin='auto', fmax='auto', num_bins=None, bin_type="log", 
                                       bin_edges=[], subtract_noise_bias=True, poisson_stats=False):
        """
        """
        coherences = []
        for i in range(self.values1.shape[0]):
            coherence_spectrum = self.compute_coherence(
                times1=self.times1, values1=self.values1[i], sigmas1=self.sigmas1,
                times2=self.times2, values2=self.values2[i], sigmas2=self.sigmas2,
                fmin=fmin, fmax=fmax, num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges,
                subtract_noise_bias=False # subtract noise bias only once
            )
            freqs, freq_widths, coherence, _ = coherence_spectrum
            coherences.append(coherence)

        if subtract_noise_bias:
            # decide how to do this next, could we just use GP fit noise?
            # test how the GP noise level fit compares to the noise approximated above
            bias = (pnoise2 * (ps1 - pnoise1) + pnoise1 * (ps2 - pnoise2) + pnoise1 * pnoise2)
        else:
            bias = 0

        coherences = np.vstack(coherences)
        coherences_mean = np.mean(coherences, axis=0)
        coherences_std = np.std(coherences, axis=0)

        return freqs, freq_widths, coherences_mean, coherences_std

    def plot(self, freqs=None, freq_widths=None, cohs=None, coh_sigmas=None, **kwargs):
        """
        """
        freqs = self.freqs if freqs is None else freqs
        freq_widths = self.freq_widths if freq_widths is None else freq_widths
        cohs = self.cohs if cohs is None else cohs
        coh_sigmas = self.coh_sigmas if coh_sigmas is None else coh_sigmas

        kwargs.setdefault('xlabel', 'Frequency')
        kwargs.setdefault('ylabel', 'Cross-Spectrum')
        kwargs.setdefault('xscale', 'log')
        kwargs.setdefault('yscale', 'log')
        Plotter.plot(
            x=freqs, y=cohs, xerr=freq_widths, yerr=coh_sigmas, **kwargs
        )

    def count_frequencies_in_bins(self, fmin=None, fmax=None, num_bins=None, bin_type="log", bin_edges=[]):
        """
        Counts the number of frequencies in each frequency bin.
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