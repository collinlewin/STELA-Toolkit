import numpy as np

from ._check_inputs import _CheckInputs
from .data_loader import TimeSeries
from .frequency_binning import FrequencyBinning
from .plot import Plotter


class PowerSpectrum:
    def __init__(self,
                 times=[],
                 values=[],
                 timeseries=None,
                 fmin='auto',
                 fmax='auto',
                 num_bins=None, 
                 bin_type="log",
                 bin_edges=None,
                 norm=True,
                 plot_fft=False
                 ):
        """
        Initializes the PowerSpectrum object and computes the power spectrum.

        The method processes the input time series data (either as arrays or
        TimeSeries objects), determines whether the input represents a single
        or multiple time series, and computes the corresponding power spectrum
        using FFT. If requested, it plots the spectrum.

        Parameters:
        - times (array-like): Time values for the input time series.
        - values (array-like): Measurement values for the input time series (e.g., flux, counts).
        - timeseries (TimeSeries): TimeSeries object encapsulating time and values (optional).
        - fmin (float or 'auto'): Minimum frequency for the power spectrum.
        - fmax (float or 'auto'): Maximum frequency for the power spectrum.
        - num_bins (int): Number of bins for frequency binning.
        - norm (bool): Whether to normalize the spectrum to variance units.
        - plot_fft (bool): Whether to plot the computed power spectrum.
        """
        self.times, self.values = _CheckInputs._check_input_data(timeseries, times, values)
        _CheckInputs._check_input_bins(num_bins, bin_type, bin_edges)
        self.fmin = fmin
        self.fmax = fmax

        # if multiple time series are provided, compute the stacked power spectrum
        if len(values.shape) == 2:
            self.freqs, self.freq_widths, self.powers, self.power_sigmas = self.compute_stacked_power_spectrum(
                fmin=self.fmin, fmax=self.fmax, num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges, norm=norm
            )
        else:
            self.freqs, self.freq_widths, self.powers, self.power_sigmas = self.compute_power_spectrum(
                fmin=self.fmin, fmax=self.fmax, num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges, norm=norm
            )

        if plot_fft:
            self.plot()

    def compute_power_spectrum(self, times=None, values=None, fmin='auto', fmax='auto',
                               num_bins=None, bin_type="log", bin_edges=None, norm=True):        
        """
        Computes the power spectrum for a single time series.

        Parameters:
        - times (array-like): Time values for the time series (optional).
        - values (array-like): Measurement values for the time series (optional).
        - fmin (float or 'auto'): Minimum frequency for the power spectrum.
        - fmax (float or 'auto'): Maximum frequency for the power spectrum.
        - norm (bool): Whether to normalize the spectrum to variance units.
        - num_bins (int): Number of bins for frequency binning.
        - bin_type (str): Type of binning ('log' or 'linear').
        - bin_edges (array-like): Custom array of bin edges (optional).

        Returns:
        - freqs (array-like): Frequencies of the power spectrum.
        - freq_widths (array-like): Bin widths of the frequencies (if binned).
        - powers (array-like): Power spectrum values.
        - power_sigmas (array-like or None): Uncertainties in power values (if binned).
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
        dt = time_diffs[0] 
        length = len(values)

        # Use absolute min and max frequencies if set to 'auto'
        fmin = 1 / (times.max() - times.min()) if fmin == 'auto' else fmin
        fmax = 1 / (2 * dt) if fmax == 'auto' else fmax # Nyquist frequency

        fft = np.fft.fft(values)
        freqs = np.fft.fftfreq(length, d=dt)
        powers = np.abs(fft) ** 2

        # Filter frequencies within [fmin, fmax]
        valid_mask = (freqs >= fmin) & (freqs <= fmax)
        freqs = freqs[valid_mask]
        powers = powers[valid_mask]

        if num_bins or bin_edges:
            if bin_edges:
                # use custom bin edges
                bin_edges = FrequencyBinning.define_bins(freqs, num_bins=num_bins, bins=bin_edges, bin_type=bin_type)
            elif num_bins:
                # use equal-width bins in log or linear space
                bin_edges = FrequencyBinning.define_bins(freqs, num_bins=num_bins, bin_type=bin_type)
            else:
                raise ValueError("Either num_bins or bin_edges must be provided.\n"
                                 "In other words, you must specify the number of bins or the bin edges.")
            
            freqs, freq_widths, powers, power_sigmas = FrequencyBinning.bin_data(freqs, powers, bin_edges)
        else:
            freq_widths, power_sigmas = None, None

        # Normalize power spectrum to units of variance
        if norm:
            powers /= length * np.mean(values) ** 2 / (2 * dt)
            
        return freqs, freq_widths, powers, power_sigmas
    
    def compute_stacked_power_spectrum(self, fmin='auto', fmax='auto', num_bins=None, 
                                       bin_type="log", bin_edges=None, norm=True):
        """
        Computes the power spectrum for multiple realizations of a time series.

        The method iterates over multiple realizations of the input data,
        calculates the power spectrum for each realization, and averages
        the results to compute the mean and standard deviation of the power
        values for each frequency bin.

        Parameters:
        - fmin (float or 'auto'): Minimum frequency for the power spectrum.
        - fmax (float or 'auto'): Maximum frequency for the power spectrum.
        - norm (bool): Whether to normalize the spectrum to variance units.
        - num_bins (int): Number of bins for frequency binning.
        - bin_type (str): Type of binning ('log' or 'linear').
        - bin_edges (array-like): Custom array of bin edges (optional).

        Returns:
        - freqs (array-like): Frequencies of the power spectrum.
        - freq_widths (array-like): Bin widths of the frequencies.
        - power_mean (array-like): Mean power spectrum values across realizations.
        - power_std (array-like): Standard deviation of power values across realizations.
        """
        powers = []
        power_sigmas = []
        for i in range(self.values.shape[0]):
            freqs_oneseries, freq_widths, powers_oneseries, power_sigmas_oneseries = self.compute_power_spectrum(
                self.times, self.values[i], fmin=fmin, fmax=fmax, norm=norm, 
                num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges
            )
            powers.append(powers_oneseries)
            if num_bins:
                power_sigmas.append(power_sigmas_oneseries)

        # Stack the collected powers and sigmas
        powers = np.vstack(powers)

        if num_bins:
            power_sigmas = np.vstack(power_sigmas)

        freqs = freqs_oneseries
        freq_widths = freq_widths
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

    def bin(self, num_bins=None, bin_type="log",  bin_edges=None, plot=False, save=True, verbose=True):
        """
        Bins the power spectrum data, either using a specified number of bins or custom bin edges.

        Parameters:
        - num_bins: Number of bins (if `bins` is not provided).
        - bins: Custom array of bin edges (optional).
        - bin_type: Type of binning ("log", "linear", or custom).
        - plot: If True, plots the binned data.
        - save: If True, updates the internal attributes with binned data.
        - verbose: If True, prints information about binning.
        """
        if bin_edges is None:
            try:
                bin_edges = FrequencyBinning.define_bins(self.freqs, num_bins=num_bins, bins=bin_edges, bin_type=bin_type)
            except ValueError:
                raise ValueError("Either num_bins or bin_edges must be provided.\n"
                                 "In other words, you must specify the number of bins or the bin edges.")

        if verbose:
            num_freqs_in_bins = FrequencyBinning.count_frequencies_in_bins(self.freqs, bin_edges)
            print(f"Number of frequencies in each bin: {num_freqs_in_bins}")

        freqs, freq_widths, powers, power_sigmas = FrequencyBinning.bin_data(self.freqs, self.powers, bin_edges)

        if plot:
            self.plot(freqs=freqs, freq_widths=freq_widths, powers=powers, power_sigmas=power_sigmas)

        if save:
            self.freqs, self.freq_widths, self.powers, self.power_sigmas = freqs, freq_widths, powers, power_sigmas
    
    def count_frequencies_in_bins(self, bin_edges=None, num_bins=None, bin_type="log"):
        """
        Counts the number of frequencies in each bin for the power spectrum.

        Uses the frequency data to count the number of entries in each bin defined
        by the provided edges or calculated based on the number of bins and bin type.

        Parameters:
        - bin_edges (array-like): Custom array of bin edges (optional).
        - num_bins (int): Number of bins to create (if bin_edges is not provided).
        - bin_type (str): Type of binning ("log" or "linear").

        Returns:
        - bin_counts (list): List of counts of frequencies in each bin.

        Raises:
        - ValueError: If neither bin_edges nor num_bins is provided.
        """
        if bin_edges is None:
            if num_bins is None:
                raise ValueError("Either bin_edges or num_bins must be provided to count frequencies in bins.")
            bin_edges = FrequencyBinning.define_bins(self.freqs, num_bins=num_bins, bin_type=bin_type)

        bin_counts = FrequencyBinning.count_frequencies_in_bins(self.freqs, bin_edges)
        return bin_counts