import numpy as np

from ._check_inputs import _CheckInputs
from .frequency_binning import FrequencyBinning
from .plot import Plotter
from .data_loader import LightCurve


class PowerSpectrum:
    """
    Computes the power spectrum for light curve data.

    This class calculates the power spectrum for single or multiple realizations
    of light curve data. It supports frequency binning and optional normalization.

    Parameters:
    - lightcurve (object, optional): A LightCurve object (overrides times/rates). Must have regular sampling, otherwise use model.
    - model (object, optional): A model-class (e.g., GaussianProcess) object. 
    - fmin (float or 'auto', optional): Minimum frequency for computation.
    - fmax (float or 'auto', optional): Maximum frequency for computation.
    - num_bins (int, optional): Number of bins for frequency binning.
    - bin_type (str, optional): Type of binning ('log' or 'linear').
    - bin_edges (array-like, optional): Predefined edges for frequency bins.
    - norm (bool, optional): Whether to normalize the spectrum to variance units.
    - plot_fft (bool, optional): Whether to automatically plot the power spectrum.

    Attributes:
    - freqs (array-like): Frequencies of the power spectrum.
    - freq_widths (array-like): Bin widths of the frequencies.
    - powers (array-like): Power spectrum values.
    - power_errors (array-like): Uncertainty of the power spectrum values.
    """

    def __init__(self,
                 lightcurve_or_model,
                 fmin='auto',
                 fmax='auto',
                 num_bins=None,
                 bin_type="log",
                 bin_edges=[],
                 norm=True,
                 ):
        # To do: ValueError for norm=True acting on mean=0 (standardized data)
        input_data = _CheckInputs._check_lightcurve_or_model(lightcurve_or_model)
        if input_data['type'] == 'model':
            self.times, self.rates = input_data['data']
        else:
            self.times, self.rates, _ = input_data['data']
        _CheckInputs._check_input_bins(num_bins, bin_type, bin_edges)

        # Use absolute min and max frequencies if set to 'auto'
        self.dt = np.diff(self.times)[0]
        self.fmin = np.fft.rfftfreq(len(self.rates), d=self.dt)[1] if fmin == 'auto' else fmin
        self.fmax = np.fft.rfftfreq(len(self.rates), d=self.dt)[-1] if fmax == 'auto' else fmax  # nyquist frequency

        self.num_bins = num_bins
        self.bin_type = bin_type
        self.bin_edges = bin_edges

        # if multiple light curve are provided, compute the stacked power spectrum
        if len(self.rates.shape) == 2:
            power_spectrum = self.compute_stacked_power_spectrum(norm=norm)
        else:
            power_spectrum = self.compute_power_spectrum(norm=norm)

        self.freqs, self.freq_widths, self.powers, self.power_errors = power_spectrum

    def compute_power_spectrum(self, times=None, rates=None, norm=True):
        """
        Computes the power spectrum for a single light curve.

        Parameters:
        - times (array-like, optional): Time values for the light curve.
        - rates (array-like, optional): Measurement rates for the light curve.
        - fmin (float or 'auto', optional): Minimum frequency for the power spectrum.
        - fmax (float or 'auto', optional): Maximum frequency for the power spectrum.
        - norm (bool, optional): Whether to normalize the spectrum to variance units.
        - num_bins (int, optional): Number of bins for frequency binning.
        - bin_type (str, optional): Type of binning ('log' or 'linear').
        - bin_edges (array-like, optional): Custom array of bin edges.

        Returns:
        - freqs (array-like): Frequencies of the power spectrum.
        - freq_widths (array-like or None): Bin widths of the frequencies.
        - powers (array-like): Power spectrum values.
        - power_errors (array-like or None): Uncertainties in power values.
        """
        times = self.times if times is None else times
        rates = self.rates if rates is None else rates
        length = len(rates)

        freqs, fft = LightCurve(times=times, rates=rates).fft()
        powers = np.abs(fft) ** 2

        # Filter frequencies within [fmin, fmax]
        valid_mask = (freqs >= self.fmin) & (freqs <= self.fmax)
        freqs = freqs[valid_mask]
        powers = powers[valid_mask]

        if norm:
            powers /= length * np.mean(rates) ** 2 / (2 * self.dt)

        # Apply binning
        if self.num_bins or self.bin_edges:
            
            if self.bin_edges:
                bin_edges = FrequencyBinning.define_bins(self.fmin, self.fmax, num_bins=self.num_bins, 
                                                         bin_type=self.bin_type, bin_edges=self.bin_edges
                                                        )

            elif self.num_bins:
                bin_edges = FrequencyBinning.define_bins(self.fmin, self.fmax, num_bins=self.num_bins, bin_type=self.bin_type)

            else:
                raise ValueError("Either num_bins or bin_edges must be provided.\n"
                                 "In other words, you must specify the number of bins or the bin edges.")

            binned_power = FrequencyBinning.bin_data(freqs, powers, bin_edges)
            freqs, freq_widths, powers, power_errors = binned_power
        else:
            freq_widths, power_errors = None, None

        return freqs, freq_widths, powers, power_errors

    def compute_stacked_power_spectrum(self, norm=True):
        """
        Computes the power spectrum for multiple realizations of a light curve.

        For multiple realizations, this method calculates the power spectrum for each
        realization and averages the results to compute the mean and standard deviation
        for each frequency bin.

        Parameters:
        - fmin (float or 'auto', optional): Minimum frequency for the power spectrum.
        - fmax (float or 'auto', optional): Maximum frequency for the power spectrum.
        - num_bins (int, optional): Number of bins for frequency binning.
        - bin_type (str, optional): Type of binning ('log' or 'linear').
        - bin_edges (array-like, optional): Custom array of bin edges.
        - norm (bool, optional): Whether to normalize the spectrum to variance units.

        Returns:
        - freqs (array-like): Frequencies of the power spectrum.
        - freq_widths (array-like): Bin widths of the frequencies.
        - power_mean (array-like): Mean power spectrum values across realizations.
        - power_std (array-like): Standard deviation of power values across realizations.
        """
        powers = []
        for i in range(self.rates.shape[0]):
            power_spectrum = self.compute_power_spectrum(self.times, self.rates[i], norm=norm)
            freqs, freq_widths, power, _ = power_spectrum
            powers.append(power)

        # Stack the collected powers and errors
        powers = np.vstack(powers)
        power_mean = np.mean(powers, axis=0)
        power_std = np.std(powers, axis=0)

        return freqs, freq_widths, power_mean, power_std

    def plot(self, freqs=None, freq_widths=None, powers=None, power_errors=None, **kwargs):
        """
        Plots the power spectrum.

        Parameters:
        - freqs (array-like): Frequencies to plot (optional, defaults to internal data).
        - freq_widths (array-like): Frequency bin widths (optional, defaults to internal data).
        - powers (array-like): Power values to plot (optional, defaults to internal data).
        - power_errors (array-like): Uncertainties in power values (optional, defaults to internal data).
        - **kwargs: Additional keyword arguments for plot customization.
        """
        freqs = self.freqs if freqs is None else freqs
        freq_widths = self.freq_widths if freq_widths is None else freq_widths
        powers = self.powers if powers is None else powers
        power_errors = self.power_errors if power_errors is None else power_errors

        kwargs.setdefault('xlabel', 'Frequency')
        kwargs.setdefault('ylabel', 'Power')
        kwargs.setdefault('xscale', 'log')
        kwargs.setdefault('yscale', 'log')
        Plotter.plot(x=freqs, y=powers, xerr=freq_widths, yerr=power_errors, **kwargs)

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