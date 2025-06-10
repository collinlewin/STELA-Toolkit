import numpy as np
from ._check_inputs import _CheckInputs
from .cross_spectrum import CrossSpectrum
from .data_loader import LightCurve
from .frequency_binning import FrequencyBinning
from .plot import Plotter
from .power_spectrum import PowerSpectrum


class Coherence:
    """
    Compute the frequency-dependent coherence between two light curves or GP models.

    This class estimates the coherence spectrum, which quantifies the degree of linear correlation
    between two time series as a function of frequency. Coherence values range from 0 to 1,
    with values near 1 indicating a strong linear relationship at that frequency.

    Inputs can be either LightCurve objects or trained GaussianProcess models from this package.
    If GP models are provided and posterior samples already exist, those are used.
    If no samples exist, 1000 GP realizations will be generated automatically on a 1000-point grid.

    If both inputs are GP models, the coherence is computed for each sample pair and the
    mean and standard deviation across samples are returned. Otherwise, coherence is computed
    on the raw input light curves.

    Poisson noise bias correction is supported and may be enabled to correct for uncorrelated noise.

    Parameters
    ----------
    lc_or_model1 : LightCurve or GaussianProcess
        First input light curve or trained GP model.
    
    lc_or_model2 : LightCurve or GaussianProcess
        Second input light curve or trained GP model.
    
    fmin : float or 'auto', optional
        Minimum frequency for the coherence spectrum. If 'auto', uses the lowest nonzero FFT frequency.
    
    fmax : float or 'auto', optional
        Maximum frequency. If 'auto', uses the Nyquist frequency.
    
    num_bins : int, optional
        Number of frequency bins.
    
    bin_type : str, optional
        Type of frequency binning ('log' or 'linear').
    
    bin_edges : array-like, optional
        Custom frequency bin edges.
    
    subtract_noise_bias : bool, optional
        Whether to subtract Poisson noise bias from the coherence spectrum.
    
    bkg1 : float, optional
        Background count rate for lightcurve 1 (used in noise bias correction).
    
    bkg2 : float, optional
        Background count rate for lightcurve 2.

    Attributes
    ----------
    freqs : array-like
        Frequency bin centers.
    
    freq_widths : array-like
        Widths of each frequency bin.
    
    cohs : array-like
        Coherence values.
    
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
                 subtract_noise_bias=True,
                 bkg1=0,
                 bkg2=0):
        
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

        self.bkg1 = bkg1
        self.bkg2 = bkg2

        # Check if the input rates are for multiple realizations
        # this needs to be corrected for handling different shapes and dim val1 != dim val2
        # namely for multiple observations
        if len(self.rates1.shape) == 2 and len(self.rates2.shape) == 2:
            coherence_spectrum = self.compute_stacked_coherence()
        else:
            coherence_spectrum = self.compute_coherence(subtract_noise_bias=subtract_noise_bias)

        self.freqs, self.freq_widths, self.cohs, self.coh_errors = coherence_spectrum

    def compute_coherence(self, times1=None, rates1=None, times2=None, rates2=None, subtract_noise_bias=True):
        """
        Compute the coherence spectrum between two light curves.

        Parameters
        ----------
        times1, rates1 : array-like, optional
            Time and rate values for the first light curve. Defaults to object attributes.
        times2, rates2 : array-like, optional
            Time and rate values for the second light curve. Defaults to object attributes.
        subtract_noise_bias : bool, optional
            Whether to subtract the estimated noise bias.

        Returns
        -------
        freqs : array-like
            Frequency bin centers.
        freq_widths : array-like
            Frequency bin widths.
        coherence : array-like
            Coherence spectrum.
        None
            Reserved for compatibility (with the stacked method).
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

        cohs = (np.abs(cs) ** 2 - bias) / (ps1 * ps2)

        M = self.count_frequencies_in_bins()
        coh_errors = np.sqrt((1 - cohs) / (2 * cohs * M))

        return power_spectrum1.freqs, power_spectrum1.freq_widths, cohs, coh_errors

    def compute_stacked_coherence(self):
        """
        Compute the coherence from stacked realizations of the light curves.

        For multiple realizations (GP samples), this method computes the
        coherence for each pair of realizations and returns the mean and standard deviation.

        Returns
        -------
        freqs : array-like
            Frequency bin centers.
        freq_widths : array-like
            Frequency bin widths.
        coherence_mean : array-like
            Mean coherence spectrum across realizations.
        coherence_std : array-like
            Standard deviation of the coherence across realizations.
        """

        coherences = []
        for i in range(self.rates1.shape[0]):
            coherence_spectrum = self.compute_coherence(times1=self.times1, rates1=self.rates1[i],
                                                        times2=self.times2, rates2=self.rates2[i],
                                                        subtract_noise_bias=False
                                                    )
            freqs, freq_widths, coherence, _ = coherence_spectrum
            coherences.append(coherence)

        coherences = np.vstack(coherences)
        coherences_mean = np.mean(coherences, axis=0)
        coherences_std = np.std(coherences, axis=0)

        return freqs, freq_widths, coherences_mean, coherences_std

    def compute_bias(self, power_spectrum1, power_spectrum2):
        """
        Estimate the Poisson noise bias for the coherence calculation. 

        Parameters
        ----------
        power_spectrum1 : array-like
            Power spectrum of the first light curve.
        power_spectrum2 : array-like
            Power spectrum of the second light curve.

        Returns
        -------
        bias : array-like
            Estimated noise bias per frequency bin.
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
        Plot the coherence spectrum.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for plot customization (e.g., xlabel, xscale).
        """

        freqs = self.freqs if freqs is None else freqs
        freq_widths = self.freq_widths if freq_widths is None else freq_widths
        cohs = self.cohs if cohs is None else cohs
        coh_errors = self.coh_errors if coh_errors is None else coh_errors

        kwargs.setdefault('xlabel', 'Frequency')
        kwargs.setdefault('ylabel', 'Coherence')
        kwargs.setdefault('xscale', 'log')
        Plotter.plot(
            x=freqs, y=cohs, xerr=freq_widths, yerr=coh_errors, **kwargs
        )

    def count_frequencies_in_bins(self, fmin=None, fmax=None, num_bins=None, bin_type=None, bin_edges=[]):
        """
        Counts the number of frequencies in each frequency bin.
        Wrapper method to use FrequencyBinning.count_frequencies_in_bins with class attributes.
        """
        
        return FrequencyBinning.count_frequencies_in_bins(
            self, fmin=fmin, fmax=fmax, num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges
        )
