import numpy as np

from ._check_inputs import _CheckInputs
from .cross_spectrum import CrossSpectrum
from .data_loader import LightCurve
from .frequency_binning import FrequencyBinning
from .plot import Plotter
from .power_spectrum import PowerSpectrum


class Coherence:
    """
    Computes the coherence between two light curves.

    This class calculates the coherence spectrum, which measures the degree of linear 
    correlation between two light curves in the frequency domain. It supports stacked 
    realizations, optional noise bias subtraction, and Poisson statistics.

    Parameters:
    - times1 (array-like, optional): Time values for the first light curve.
    - rates1 (array-like, optional): Measurement rates for the first light curve.
    - errors1 (array-like, optional): Errors for the first light curve.
    - times2 (array-like, optional): Time values for the second light curve.
    - rates2 (array-like, optional): Measurement rates for the second light curve.
    - errors2 (array-like, optional): Errors for the second light curve.
    - lightcurve1 (object, optional): A LightCurve object (overrides times1/rates1/errors1).
    - lightcurve2 (object, optional): A LightCurve object (overrides times2/rates2/errors2).
    - fmin (float or 'auto', optional): Minimum frequency for computation.
    - fmax (float or 'auto', optional): Maximum frequency for computation.
    - num_bins (int, optional): Number of bins for frequency binning.
    - bin_type (str, optional): Type of binning ('log' or 'linear').
    - bin_edges (array-like, optional): Custom bin edges for binning.
    - subtract_noise_bias (bool, optional): Whether to subtract the noise bias from coherence.
    - plot_coh (bool, optional): Whether to automatically plot the coherence spectrum.

    Attributes:
    - freqs (array-like): Frequencies of the coherence spectrum.
    - freq_widths (array-like): Bin widths of the frequencies.
    - cohs (array-like): Coherence values for each frequency bin.
    - coh_errors (array-like): Uncertainties in the coherence values.
    """

    def __init__(self,
                 lightcurve_or_model1,
                 lightcurve_or_model2,
                 fmin='auto',
                 fmax='auto',
                 num_bins=None,
                 bin_type="log",
                 bin_edges=[],
                 subtract_noise_bias=True,
                 bkg1=0,
                 bkg2=0,
                 plot_coh=False
                 ):
        # To do: determine if or if not Poisson statistics for the user
        input_data = _CheckInputs._check_lightcurve_or_model(lightcurve_or_model1)
        if input_data['type'] == 'model':
            self.times1, self.rates1 = input_data['data']
        else:
            self.times1, self.rates1, self.errors1 = input_data['data']

        input_data = _CheckInputs._check_lightcurve_or_model(lightcurve_or_model2)
        if input_data['type'] == 'model':
            self.times2, self.rates2 = input_data['data']
        else:
            self.times2, self.rates2, self.errors2 = input_data['data']
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

        self.bkg1 = bkg1
        self.bkg2 = bkg2

        # Check if the input rates are for multiple realizations
        # this needs to be corrected for handling different shapes and dim val1 != dim val2
        # namely for multiple observations
        if len(self.rates1.shape) == 2 and len(self.rates2.shape) == 2:
            coherence_spectrum = self.compute_stacked_coherence(subtract_noise_bias=subtract_noise_bias)
        else:
            coherence_spectrum = self.compute_coherence(subtract_noise_bias=subtract_noise_bias)

        self.freqs, self.freq_widths, self.cohs, self.coh_errors = coherence_spectrum

        if plot_coh:
            self.plot()

    def compute_coherence(self, times1=None, rates1=None, times2=None, rates2=None, subtract_noise_bias=True):
        """
        Computes the coherence between two light curves.

        Parameters:
        - times1, rates1, errors1 (array-like, optional): Data for the first light curve.
        - times2, rates2, errors2 (array-like, optional): Data for the second light curve.
        - fmin (float or 'auto', optional): Minimum frequency for computation.
        - fmax (float or 'auto', optional): Maximum frequency for computation.
        - num_bins (int, optional): Number of bins for frequency binning.
        - bin_type (str, optional): Type of binning ('log' or 'linear').
        - bin_edges (array-like, optional): Custom bin edges for binning.
        - subtract_noise_bias (bool, optional): Whether to subtract the noise bias.

        Returns:
        - freqs (array-like): Frequencies of the coherence spectrum.
        - freq_widths (array-like): Bin widths of the frequencies.
        - coherence (array-like): Coherence values.
        - None (NoneType): Placeholder for compatibility with other methods.
        """
        times1 = self.times1 if times1 is None else times1
        rates1 = self.rates1 if rates1 is None else rates1
        times2 = self.times2 if times2 is None else times2
        rates2 = self.rates2 if rates2 is None else rates2

        lc1 = LightCurve(times=times1, rates=rates1)
        lc2 = LightCurve(times=times2, rates=rates2)
        cross_spectrum = CrossSpectrum(
            lc1, lc2,
            fmin=self.fmin, fmax=self.fmax,
            num_bins=self.num_bins, bin_type=self.bin_type, bin_edges=self.bin_edges
        )
        power_spectrum1 = PowerSpectrum(
            lc1,
            fmin=self.fmin, fmax=self.fmax,
            num_bins=self.num_bins, bin_type=self.bin_type, bin_edges=self.bin_edges
        )
        power_spectrum2 = PowerSpectrum(
            lc2,
            fmin=self.fmin, fmax=self.fmax,
            num_bins=self.num_bins, bin_type=self.bin_type, bin_edges=self.bin_edges
        )

        ps1 = power_spectrum1.powers
        ps2 = power_spectrum2.powers
        cs = cross_spectrum.cs

        if subtract_noise_bias:
            bias = self.compute_bias(ps1, ps2)
        else:
            bias = 0

        coherence = (np.abs(cs) ** 2 - bias) / ps1 * ps2
        return power_spectrum1.freqs, power_spectrum1.freq_widths, coherence, None

    def compute_stacked_coherence(self, subtract_noise_bias=True):
        """
        Computes the coherence spectrum for stacked realizations.

        This method calculates the coherence for multiple realizations of two light curve 
        and averages the results to obtain the mean and standard deviation.

        Parameters:
        - fmin (float or 'auto', optional): Minimum frequency for computation.
        - fmax (float or 'auto', optional): Maximum frequency for computation.
        - num_bins (int, optional): Number of bins for frequency binning.
        - bin_type (str, optional): Type of binning ('log' or 'linear').
        - bin_edges (array-like, optional): Custom bin edges for binning.
        - subtract_noise_bias (bool, optional): Whether to subtract the noise bias.

        Returns:
        - freqs (array-like): Frequencies of the coherence spectrum.
        - freq_widths (array-like): Bin widths of the frequencies.
        - coherence_mean (array-like): Mean coherence values across realizations.
        - coherence_std (array-like): Standard deviation of coherence values.
        """
        coherences = []
        for i in range(self.rates1.shape[0]):
            coherence_spectrum = self.compute_coherence(
                times1=self.times1, rates1=self.rates1[i], errors1=self.errors1,
                times2=self.times2, rates2=self.rates2[i], errors2=self.errors2,
                subtract_noise_bias=subtract_noise_bias
            )
            freqs, freq_widths, coherence, _ = coherence_spectrum
            coherences.append(coherence)

        coherences = np.vstack(coherences)
        coherences_mean = np.mean(coherences, axis=0)
        coherences_std = np.std(coherences, axis=0)

        return freqs, freq_widths, coherences_mean, coherences_std

    def compute_bias(self, power_spectrum1, power_spectrum2):
        """
        """
        mean1 = np.mean(self.rates1)
        mean2 = np.mean(self.rates2)

        pnoise1 = 2 * (mean1 + self.bkg1) / mean1 ** 2
        pnoise2 = 2 * (mean2 + self.bkg2) / mean2 ** 2

        bias = (
            pnoise2 * (power_spectrum1 - pnoise1)
            + pnoise1 * (power_spectrum2 - pnoise2)
            + pnoise1 * pnoise2
        )

        num_freq = self.count_frequencies_in_bins()
        bias /= num_freq
        return bias

    def plot(self, freqs=None, freq_widths=None, cohs=None, coh_errors=None, **kwargs):
        """
        Plots the coherence spectrum.

        Parameters:
        - **kwargs: Keyword arguments for customizing the plot.
        """
        freqs = self.freqs if freqs is None else freqs
        freq_widths = self.freq_widths if freq_widths is None else freq_widths
        cohs = self.cohs if cohs is None else cohs
        coh_errors = self.coh_errors if coh_errors is None else coh_errors

        kwargs.setdefault('xlabel', 'Frequency')
        kwargs.setdefault('ylabel', 'Coherence')
        kwargs.setdefault('xscale', 'log')
        kwargs.setdefault('yscale', 'log')
        Plotter.plot(
            x=freqs, y=cohs, xerr=freq_widths, yerr=coh_errors, **kwargs
        )

    def count_frequencies_in_bins(self, fmin=None, fmax=None, num_bins=None, bin_type=None, bin_edges=[]):
        """
        Counts the number of frequencies in each frequency bin.
        Wrapper method to use FrequencyBinning.count_frequencies_in_bins with class attributes.

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
            self, fmin=fmin, fmax=fmax, num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges
        )
