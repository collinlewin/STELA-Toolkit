import numpy as np

from ._check_inputs import _CheckInputs
from .plot import Plotter
from .frequency_binning import FrequencyBinning
from .data_loader import LightCurve


class CrossSpectrum:
    """
    Computes the cross-spectrum between two light curves.

    This class calculates the cross-spectrum for single or multiple realizations
    of two light curves. It supports frequency binning and optional normalization
    to be in units consistent with the power spectral density (PSD).

    Parameters:
    - times1 (array-like, optional): Time points for the first light curve.
    - rates1 (array-like, optional): Rates for the first light curve.
    - times2 (array-like, optional): Time points for the second light curve.
    - rates2 (array-like, optional): Rates for the second light curve.
    - lightcurve1 (object, optional): First light curve object (overrides times1/rates1).
    - lightcurve2 (object, optional): Second light curve object (overrides times2/rates2).
    - fmin (float or 'auto', optional): Minimum frequency for computation.
    - fmax (float or 'auto', optional): Maximum frequency for computation.
    - num_bins (int, optional): Number of bins for frequency binning.
    - bin_type (str, optional): Type of binning ('log' or 'linear').
    - bin_edges (array-like, optional): Predefined edges for frequency bins.
    - norm (bool, optional): Whether to normalize the spectrum.
    - plot_cs (bool, optional): Whether to automatically plot the cross-spectrum.

    Key Attributes:
    - freqs (array-like): Frequencies of the cross-spectrum.
    - freq_widths (array-like): Bin widths of the frequencies.
    - cs (array-like): Cross-spectrum values.
    - cs_errors (array-like): Uncertainty of the cross-spectrum values.
    """

    def __init__(self,
                 lightcurve_or_model1,
                 lightcurve_or_model2,
                 fmin='auto',
                 fmax='auto',
                 num_bins=None,
                 bin_type="log",
                 bin_edges=[],
                 norm=True,
                 plot_cs=False
                 ):
        # To do: update main docstring
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
        self.fmin = np.fft.rfftfreq(len(self.rates1), d=self.dt)[0] if fmin == 'auto' else fmin
        self.fmax = np.fft.rfftfreq(len(self.rates1), d=self.dt)[-1] if fmax == 'auto' else fmax  # nyquist frequency

        self.num_bins = num_bins
        self.bin_type = bin_type
        self.bin_edges = bin_edges

        # Check if the input rates are for multiple realizations
        # this needs to be corrected for handling different shapes and dim val1 != dim val2
        if len(self.rates1.shape) == 2 and len(self.rates2.shape) == 2:
            cross_spectrum = self.compute_stacked_cross_spectrum(norm=norm)
        else:
            cross_spectrum = self.compute_cross_spectrum(norm=norm)

        self.freqs, self.freq_widths, self.cs, self.cs_errors = cross_spectrum

        if plot_cs:
            self.plot()

    def compute_cross_spectrum(self, times1=None, rates1=None, times2=None, rates2=None, norm=True):
        """
        Computes the cross-spectrum between two light curves.

        Parameters:
        - times1, rates1 (array-like, optional): Time and rates for the first light curve.
        - times2, rates2 (array-like, optional): Time and rates for the second light curve.
        - fmin (float or 'auto', optional): Minimum frequency for computation.
        - fmax (float or 'auto', optional): Maximum frequency for computation.
        - num_bins (int, optional): Number of bins for frequency binning.
        - bin_type (str, optional): Type of binning ('log' or 'linear').
        - bin_edges (array-like, optional): Predefined edges for frequency bins.
        - norm (bool, optional): Whether to normalize the spectrum.

        Returns:
        - freqs (array-like): Frequencies of the cross-spectrum.
        - freq_widths (array-like): Bin widths of the frequencies.
        - cross_spectrum (array-like): Cross-spectrum values.
        - None (NoneType): Placeholder for compatibility with other methods.
        """
        times1 = self.times1 if times1 is None else times1
        rates1 = self.rates1 if rates1 is None else rates1
        times2 = self.times2 if times2 is None else times2
        rates2 = self.rates2 if rates2 is None else rates2

        freqs, fft1 = LightCurve(times=times1, rates=rates1).fft()
        _, fft2 = LightCurve(times=times2, rates=rates2).fft()

        cross_spectrum = np.conj(fft1) * fft2

        # Filter frequencies within [fmin, fmax]
        valid_mask = (freqs >= self.fmin) & (freqs <= self.fmax)
        freqs = freqs[valid_mask]
        cross_spectrum = cross_spectrum[valid_mask]

        # Normalize power spectrum to units of variance (PSD)
        if norm:
            length = len(rates1)
            norm_factor = length * np.mean(rates1) * np.mean(rates2) / (2 * self.dt)
            cross_spectrum /= norm_factor

            # negative norm factor shifts the phase by pi
            if norm_factor < 0:
                phase = np.angle(cross_spectrum)
                cross_spectrum = np.abs(cross_spectrum) * np.exp(1j * phase)

        # Apply binning
        if self.num_bins or self.bin_edges:
            if self.bin_edges:
                # use custom bin edges
                bin_edges = FrequencyBinning.define_bins(
                    self.fmin, self.fmax, num_bins=self.num_bins,
                    bin_type=self.bin_type, bin_edges=self.bin_edges
                )
            elif self.num_bins:
                # use equal-width bins in log or linear space
                bin_edges = FrequencyBinning.define_bins(
                    self.fmin, self.fmax, num_bins=self.num_bins, bin_type=self.bin_type
                )
            else:
                raise ValueError("Either num_bins or bin_edges must be provided.\n"
                                 "In other words, you must specify the number of bins or the bin edges.")

            binned_cross_spectrum = FrequencyBinning.bin_data(freqs, cross_spectrum, bin_edges)
            freqs, freq_widths, cross_spectrum, cross_spectrum_errors = binned_cross_spectrum
        else:
            freq_widths, cross_spectrum_errors = None, None

        return freqs, freq_widths, cross_spectrum, cross_spectrum_errors

    def compute_stacked_cross_spectrum(self, norm=True):
        """
        Computes the cross-spectrum for multiple realizations.

        For multiple realizations (e.g., GP samples), this method computes the
        cross-spectrum for each realization pair. The resulting cross-spectra are
        averaged to compute the mean and standard deviation for each frequency bin.

        Parameters:
        - fmin (float or 'auto', optional): Minimum frequency for computation.
        - fmax (float or 'auto', optional): Maximum frequency for computation.
        - num_bins (int, optional): Number of bins for frequency binning.
        - bin_type (str, optional): Type of binning ('log' or 'linear').
        - bin_edges (array-like, optional): Predefined edges for frequency bins.
        - norm (bool, optional): Whether to normalize the spectrum.

        Returns:
        - freqs (array-like): Frequencies of the cross-spectrum.
        - freq_widths (array-like): Bin widths of the frequencies.
        - cross_spectra_mean (array-like): Mean cross-spectrum values.
        - cross_spectra_std (array-like): Standard deviation of cross-spectrum values.
        """
        cross_spectra = []
        for i in range(self.rates1.shape[0]):
            cross_spectrum = self.compute_cross_spectrum(
                times1=self.times1, rates1=self.rates1[i],
                times2=self.times2, rates2=self.rates2[i],
                norm=norm
            )
            cross_spectra.append(cross_spectrum[2])

        cross_spectra = np.vstack(cross_spectra)
        cross_spectra_mean = np.mean(cross_spectra, axis=0)
        cross_spectra_std = np.std(cross_spectra, axis=0)

        freqs, freq_widths = cross_spectrum[0], cross_spectrum[1]

        return freqs, freq_widths, cross_spectra_mean, cross_spectra_std

    def plot(self, freqs=None, freq_widths=None, cs=None, cs_errors=None, **kwargs):
        """
        Plots the cross-spectrum.

        Parameters:
        - **kwargs: Keyword arguments for customizing the plot.
        """
        freqs = self.freqs if freqs is None else freqs
        freq_widths = self.freq_widths if freq_widths is None else freq_widths
        cs = self.cs if cs is None else cs
        cs_errors = self.cs_errors if cs_errors is None else cs_errors

        kwargs.setdefault('xlabel', 'Frequency')
        kwargs.setdefault('ylabel', 'Cross-Spectrum')
        kwargs.setdefault('xscale', 'log')
        kwargs.setdefault('yscale', 'log')
        Plotter.plot(
            x=freqs, y=cs, xerr=freq_widths, yerr=cs_errors, **kwargs
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