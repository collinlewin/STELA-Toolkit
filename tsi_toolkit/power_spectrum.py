import numpy as np

from ._check_inputs import _CheckInputs
from .data_loader import TimeSeries
from .frequency_binning import FrequencyBinning
from .plot import Plotter


class PowerSpectrum:
    """
    Computes the power spectrum for time series data.

    This class calculates the power spectrum for single or multiple realizations
    of time series data. It supports frequency binning and optional normalization.

    Parameters:
    - times (array-like, optional): Time values for the time series.
    - values (array-like, optional): Measurement values for the time series.
    - timeseries (object, optional): A TimeSeries object (overrides times/values).
    - fmin (float or 'auto', optional): Minimum frequency for computation.
    - fmax (float or 'auto', optional): Maximum frequency for computation.
    - num_bins (int, optional): Number of bins for frequency binning.
    - bin_type (str, optional): Type of binning ('log' or 'linear').
    - bin_edges (array-like, optional): Predefined edges for frequency bins.
    - norm (bool, optional): Whether to normalize the spectrum to variance units.
    - plot_fft (bool, optional): Whether to automatically plot the power spectrum.

    Raises:
    - ValueError: If the time series is not evenly sampled.

    Attributes:
    - freqs (array-like): Frequencies of the power spectrum.
    - freq_widths (array-like): Bin widths of the frequencies.
    - powers (array-like): Power spectrum values.
    - power_sigmas (array-like): Uncertainty of the power spectrum values.
    """
    def __init__(self,
                 times=[],
                 values=[],
                 timeseries=None,
                 fmin='auto',
                 fmax='auto',
                 num_bins=None, 
                 bin_type="log",
                 bin_edges=[],
                 norm=True,
                 plot_fft=False
                 ):
        self.times, self.values = _CheckInputs._check_input_data(timeseries, times, values)
        _CheckInputs._check_input_bins(num_bins, bin_type, bin_edges)

        # Use absolute min and max frequencies if set to 'auto'
        self.dt = np.diff(self.times)[0]
        self.fmin = 1 / (self.times.max() - self.times.min()) if fmin == 'auto' else fmin
        self.fmax = 1 / (2 * self.dt) if fmax == 'auto' else fmax # nyquist frequency

        self.num_bins = num_bins
        self.bin_type = bin_type
        self.bin_edges = bin_edges

        # if multiple time series are provided, compute the stacked power spectrum
        if len(values.shape) == 2:
            power_spectrum  = self.compute_stacked_power_spectrum(fmin=self.fmin, fmax=self.fmax, 
                                                                  num_bins=num_bins, bin_type=bin_type, 
                                                                  bin_edges=bin_edges, norm=norm
                                                                  )
        else:
            power_spectrum = self.compute_power_spectrum(fmin=self.fmin, fmax=self.fmax,
                                                         num_bins=num_bins, bin_type=bin_type, 
                                                         bin_edges=bin_edges, norm=norm
                                                         )
        
        self.freqs, self.freq_widths, self.powers, self.power_sigmas = power_spectrum

        if plot_fft:
            self.plot()

    def compute_power_spectrum(self, times=None, values=None, fmin='auto', fmax='auto',
                               num_bins=None, bin_type="log", bin_edges=None, norm=True):        
        """
        Computes the power spectrum for a single time series.

        Parameters:
        - times (array-like, optional): Time values for the time series.
        - values (array-like, optional): Measurement values for the time series.
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
        - power_sigmas (array-like or None): Uncertainties in power values.
        """
        if times is None:
            times = self.times
        if values is None:
            values = self.values

        time_diffs = np.round(np.diff(times),10)
        if np.unique(time_diffs).size > 1:
            raise ValueError("Time series must have a uniform sampling interval.\n"
                            "Interpolate the data to a uniform grid first."
                        )
        length = len(values)

        fft = np.fft.fft(values)
        freqs = np.fft.fftfreq(length, d=self.dt)
        powers = np.abs(fft) ** 2

        # Filter frequencies within [fmin, fmax]
        valid_mask = (freqs >= self.fmin) & (freqs <= self.fmax)
        freqs = freqs[valid_mask]
        powers = powers[valid_mask]

        if num_bins or bin_edges:
            if bin_edges:
                # use custom bin edges
                bin_edges = FrequencyBinning.define_bins(
                    self.fmin, self.fmax, num_bins=num_bins, bins=bin_edges, bin_type=bin_type
                )
            elif num_bins:
                # use equal-width bins in log or linear space
                bin_edges = FrequencyBinning.define_bins(
                    self.fmin, self.fmax, num_bins=num_bins, bin_type=bin_type
                )
            else:
                raise ValueError("Either num_bins or bin_edges must be provided.\n"
                                 "In other words, you must specify the number of bins or the bin edges.")
            
            freqs, freq_widths, powers, power_sigmas = FrequencyBinning.bin_data(freqs, powers, bin_edges)
        else:
            freq_widths, power_sigmas = None, None

        # Normalize power spectrum to units of variance
        if norm:
            powers /= length * np.mean(values) ** 2 / (2 * self.dt)
            
        return freqs, freq_widths, powers, power_sigmas
    
    def compute_stacked_power_spectrum(self, fmin='auto', fmax='auto', num_bins=None, 
                                       bin_type="log", bin_edges=None, norm=True):
        """
        Computes the power spectrum for multiple realizations of a time series.

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
        for i in range(self.values.shape[0]):
            power_spectrum = self.compute_power_spectrum(self.times, self.values[i], fmin=fmin, fmax=fmax,
                                                         num_bins=num_bins, bin_type=bin_type, 
                                                         bin_edges=bin_edges, norm=norm
                                                         )
            freqs, freq_widths, power, _ = power_spectrum
            powers.append(power)

        # Stack the collected powers and sigmas
        powers = np.vstack(powers)
        power_mean = np.mean(powers, axis=0)
        power_std = np.std(powers, axis=0)

        return freqs, freq_widths, power_mean, power_std
    
    def plot(self, freqs=None, freq_widths=None, powers=None, power_sigmas=None, **kwargs):
        """
        Plots the power spectrum.

        Parameters:
        - freqs (array-like): Frequencies to plot (optional, defaults to internal data).
        - freq_widths (array-like): Frequency bin widths (optional, defaults to internal data).
        - powers (array-like): Power values to plot (optional, defaults to internal data).
        - power_sigmas (array-like): Uncertainties in power values (optional, defaults to internal data).
        - **kwargs: Additional keyword arguments for plot customization.
        """
        freqs = self.freqs if freqs is None else freqs
        freq_widths = self.freq_widths if freq_widths is None else freq_widths
        powers = self.powers if powers is None else powers
        power_sigmas = self.power_sigmas if power_sigmas is None else power_sigmas

        kwargs.setdefault('xlabel', 'Frequency')
        kwargs.setdefault('ylabel', 'Power')
        kwargs.setdefault('xscale', 'log')
        kwargs.setdefault('yscale', 'log')
        Plotter.plot(x=freqs, y=powers, xerr=freq_widths, yerr=power_sigmas, **kwargs)

    def count_frequencies_in_bins(self, fmin=None, fmax=None, num_bins=None, bin_type="log", bin_edges=[]):
        """
        Counts the number of frequencies in each bin for the power spectrum.
        Wrapper method to use FrequencyBinning.count_frequencies_in_bins with class attributes.

        If 
        Parameters:
        - fmin (float): Minimum frequency (optional).
        - fmax (float): Maximum frequency (optional).
            Class attributes will be used if not specified.
        - num_bins (int): Number of bins to create (if bin_edges is not provided).
        - bin_type (str): Type of binning ("log" or "linear").
        - bin_edges (array-like): Custom array of bin edges (optional).

        Returns:
        - bin_counts (list): List of counts of frequencies in each bin.
        """
        return FrequencyBinning.count_frequencies_in_bins(
            parent=self, fmin=fmin, fmax=fmax, num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges
        )