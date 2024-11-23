import numpy as np
import matplotlib.pyplot as plt

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
                 norm=True,
                 plot_fft=False
                 ):
        """

        """
        self.times, self.values = self._check_input(timeseries, times, values)
        self.fmin = fmin
        self.fmax = fmax

        # if multiple time series are provided, compute the stacked power spectrum
        if len(values.shape) == 2:
            self.freqs, self.freq_widths, self.powers, self.power_sigmas = self.compute_stacked_power_spectrum(
                fmin=self.fmin, fmax=self.fmax, num_bins=num_bins, norm=norm
            )
        else:
            self.freqs, self.freq_widths, self.powers, self.power_sigmas = self.compute_power_spectrum(
                fmin=self.fmin, fmax=self.fmax, num_bins=num_bins, norm=norm
            )

        if plot_fft:
            self.plot()

    def compute_power_spectrum(self, times=None, values=None, fmin='auto', fmax='auto', norm=True,
                               num_bins=None, bin_type="log", bin_edges=None):        
        """
        Computes the FFT of the values data and bins the output in frequency space.

        Returns:
        - binned_freqs (array-like): Binned frequencies.
        - binned_power (array-like): Binned power spectrum.
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
    
    def compute_stacked_power_spectrum(self, fmin='auto', fmax='auto', norm=True, num_bins=None, bin_type="log", bin_edges=None):
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
        Bins the power spectrum data.

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
    
    def _check_input(self, timeseries, times, values):
        if timeseries:
            if not isinstance(timeseries, TimeSeries):
                raise TypeError("timeseries must be an instance of the TimeSeries class.")
            times = timeseries.times
            values = timeseries.values
            
        elif len(times) > 0 and len(values) > 0:
            times = np.array(times)
            values = np.array(values)
            if len(values.shape) == 1 and len(times) != len(values):
                raise ValueError("Times and values must have the same length.")
            
            elif len(values.shape) == 2 and values.shape[1] != len(times):
                raise ValueError("Times and values must have the same length for each time series.\n"
                                 "Check the shape of the values array: expecting (n_series, n_times)."
                            )
        else:
            raise ValueError("Either provide a TimeSeries object or times and values arrays.")
        
        return times, values