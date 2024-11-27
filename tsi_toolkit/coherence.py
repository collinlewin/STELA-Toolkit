import numpy as np

from ._check_inputs import _CheckInputs
from .power_spectrum import PowerSpectrum
from .cross_spectrum import CrossSpectrum
from .plot import Plotter
from .frequency_binning import FrequencyBinning

class Coherence:
    """
    Computes the coherence between two time series.

    This class calculates the coherence spectrum, which measures the degree of linear 
    correlation between two time series in the frequency domain. It supports stacked 
    realizations, optional noise bias subtraction, and Poisson statistics.

    Parameters:
    - times1 (array-like, optional): Time values for the first time series.
    - values1 (array-like, optional): Measurement values for the first time series.
    - sigmas1 (array-like, optional): Errors for the first time series.
    - times2 (array-like, optional): Time values for the second time series.
    - values2 (array-like, optional): Measurement values for the second time series.
    - sigmas2 (array-like, optional): Errors for the second time series.
    - timeseries1 (object, optional): A time series object (overrides times1/values1/sigmas1).
    - timeseries2 (object, optional): A time series object (overrides times2/values2/sigmas2).
    - fmin (float or 'auto', optional): Minimum frequency for computation.
    - fmax (float or 'auto', optional): Maximum frequency for computation.
    - num_bins (int, optional): Number of bins for frequency binning.
    - bin_type (str, optional): Type of binning ('log' or 'linear').
    - bin_edges (array-like, optional): Custom bin edges for binning.
    - subtract_noise_bias (bool, optional): Whether to subtract the noise bias from coherence.
    - poisson_stats (bool, optional): Whether to use Poisson statistics for noise computation.
    - plot_coh (bool, optional): Whether to automatically plot the coherence spectrum.

    Raises:
    - ValueError: If the time arrays of the two time series are not identical.

    Attributes:
    - freqs (array-like): Frequencies of the coherence spectrum.
    - freq_widths (array-like): Bin widths of the frequencies.
    - cohs (array-like): Coherence values for each frequency bin.
    - coh_sigmas (array-like): Uncertainties in the coherence values.
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
                 bkg1=0,
                 bkg2=0,
                 plot_coh=False
                 ):
        # To do: determine if or if not Poisson statistics for the user
        # To do: decrease number of parameters
        self.times1, self.values1, self.sigmas1 = _CheckInputs._check_input_data(timeseries1, times1, values1, sigmas1)
        self.times2, self.values2, self.sigmas2 = _CheckInputs._check_input_data(timeseries2, times2, values2, sigmas2)
        _CheckInputs._check_input_bins(num_bins, bin_type, bin_edges)

        if not np.allclose(self.times1, self.times2):
            raise ValueError("The time arrays of the two time series must be identical.")

        # Use absolute min and max frequencies if set to 'auto'
        self.dt = np.diff(self.times1)[0]
        self.fmin = 1e-8 if fmin == 'auto' else fmin
        self.fmax = 1 / (2 * self.dt) if fmax == 'auto' else fmax # nyquist frequency

        self.num_bins = num_bins
        self.bin_type = bin_type
        self.bin_edges = bin_edges

        self.bkg1 = bkg1
        self.bkg2 = bkg2

        # Check if the input values are for multiple realizations
        # this needs to be corrected for handling different shapes and dim val1 != dim val2
        if len(self.values1.shape) == 2 and len(self.values2.shape) == 2:
            coherence_spectrum = self.compute_stacked_coherence(subtract_noise_bias=subtract_noise_bias,
                                                                poisson_stats=poisson_stats
                                                                )
        else:
            coherence_spectrum = self.compute_coherence(subtract_noise_bias=subtract_noise_bias,
                                                        poisson_stats=poisson_stats
                                                        )
        
        self.freqs, self.freq_widths, self.cohs, self.coh_sigmas = coherence_spectrum

        if plot_coh:
            self.plot()

    def compute_coherence(self, times1=None, values1=None, sigmas1=None,
                          times2=None, values2=None, sigmas2=None,
                          subtract_noise_bias=True, poisson_stats=False):
        """
        Computes the coherence between two time series.

        Parameters:
        - times1, values1, sigmas1 (array-like, optional): Data for the first time series.
        - times2, values2, sigmas2 (array-like, optional): Data for the second time series.
        - fmin (float or 'auto', optional): Minimum frequency for computation.
        - fmax (float or 'auto', optional): Maximum frequency for computation.
        - num_bins (int, optional): Number of bins for frequency binning.
        - bin_type (str, optional): Type of binning ('log' or 'linear').
        - bin_edges (array-like, optional): Custom bin edges for binning.
        - subtract_noise_bias (bool, optional): Whether to subtract the noise bias.
        - poisson_stats (bool, optional): Whether to use Poisson statistics for noise computation.

        Returns:
        - freqs (array-like): Frequencies of the coherence spectrum.
        - freq_widths (array-like): Bin widths of the frequencies.
        - coherence (array-like): Coherence values.
        - None (NoneType): Placeholder for compatibility with other methods.
        """
        times1 = self.times1 if times1 is None else times1
        values1 = self.values1 if values1 is None else values1
        sigmas1 = self.sigmas1 if sigmas1 is None else sigmas1
        times2 = self.times2 if times2 is None else times2
        values2 = self.values2 if values2 is None else values2
        sigmas2 = self.sigmas2 if sigmas2 is None else sigmas2
        
        cross_spectrum = CrossSpectrum(
            times1=times1, values1=values1,
            times2=times2, values2=values2, 
            fmin=self.fmin, fmax=self.fmax,
            num_bins=self.num_bins, bin_type=self.bin_type, bin_edges=self.bin_edges
        )
        power_spectrum1 = PowerSpectrum(
            times=times1, values=values1,
            fmin=self.fmin, fmax=self.fmax,
            num_bins=self.num_bins, bin_type=self.bin_type, bin_edges=self.bin_edges
        )
        power_spectrum2 = PowerSpectrum(
            times=times2, values=values2,
            fmin=self.fmin, fmax=self.fmax,
            num_bins=self.num_bins, bin_type=self.bin_type, bin_edges=self.bin_edges
        )

        ps1 = power_spectrum1.powers
        ps2 = power_spectrum2.powers
        cs = cross_spectrum.cs

        if subtract_noise_bias:
            bias = self.compute_bias(ps1, ps2, poisson_stats=poisson_stats)
        else:
            bias = 0

        coherence = ( np.abs(cs) ** 2 - bias ) / ps1 * ps2
        return power_spectrum1.freqs, power_spectrum1.freq_widths, coherence, None

    def compute_stacked_coherence(self, subtract_noise_bias=True, poisson_stats=False):
        """
        Computes the coherence spectrum for stacked realizations.

        This method calculates the coherence for multiple realizations of two time series 
        and averages the results to obtain the mean and standard deviation.

        Parameters:
        - fmin (float or 'auto', optional): Minimum frequency for computation.
        - fmax (float or 'auto', optional): Maximum frequency for computation.
        - num_bins (int, optional): Number of bins for frequency binning.
        - bin_type (str, optional): Type of binning ('log' or 'linear').
        - bin_edges (array-like, optional): Custom bin edges for binning.
        - subtract_noise_bias (bool, optional): Whether to subtract the noise bias.
        - poisson_stats (bool, optional): Whether to use Poisson statistics for noise computation.

        Returns:
        - freqs (array-like): Frequencies of the coherence spectrum.
        - freq_widths (array-like): Bin widths of the frequencies.
        - coherence_mean (array-like): Mean coherence values across realizations.
        - coherence_std (array-like): Standard deviation of coherence values.
        """
        coherences = []
        for i in range(self.values1.shape[0]):
            coherence_spectrum = self.compute_coherence(
                times1=self.times1, values1=self.values1[i], sigmas1=self.sigmas1,
                times2=self.times2, values2=self.values2[i], sigmas2=self.sigmas2,
                subtract_noise_bias=subtract_noise_bias, poisson_stats=poisson_stats
            )
            freqs, freq_widths, coherence, _ = coherence_spectrum
            coherences.append(coherence)

        coherences = np.vstack(coherences)
        coherences_mean = np.mean(coherences, axis=0)
        coherences_std = np.std(coherences, axis=0)

        return freqs, freq_widths, coherences_mean, coherences_std

    def compute_bias(self, power_spectrum1, power_spectrum2, poisson_stats=False):
        mean1 = np.mean(self.values1)
        mean2 = np.mean(self.values2)

        if poisson_stats:
            pnoise1 = 2 * (mean1 + self.bkg1) / mean1 ** 2
            pnoise2 = 2 * (mean2 + self.bkg2) / mean2 ** 2
        else:
            # compute the noise bias using errors
            if self.sigmas1 is None or self.sigmas2 is None:
                raise ValueError("Sigmas must be provided to compute the noise bias.")
            mean_sigma1 = np.mean(self.sigmas1)
            mean_sigma2 = np.mean(self.sigmas2)
            nyquist_freq = 1 / (2 * self.dt)
            pnoise1 = mean_sigma1 ** 2 / ( nyquist_freq * mean1 ** 2 )
            pnoise2 = mean_sigma2 ** 2 / ( nyquist_freq * mean2 ** 2 )

        bias = (
            pnoise2 * (power_spectrum1 - pnoise1) 
            + pnoise1 * (power_spectrum2 - pnoise2)
            + pnoise1 * pnoise2
        )
        return bias

    def plot(self, freqs=None, freq_widths=None, cohs=None, coh_sigmas=None, **kwargs):
        """
        Plots the coherence spectrum.

        Parameters:
        - **kwargs: Keyword arguments for customizing the plot.
        """
        freqs = self.freqs if freqs is None else freqs
        freq_widths = self.freq_widths if freq_widths is None else freq_widths
        cohs = self.cohs if cohs is None else cohs
        coh_sigmas = self.coh_sigmas if coh_sigmas is None else coh_sigmas

        kwargs.setdefault('xlabel', 'Frequency')
        kwargs.setdefault('ylabel', 'Coherence')
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