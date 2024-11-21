import numpy as np
import matplotlib.pyplot as plt

from .data_loader import TimeSeries


class PowerSpectrum:
    def __init__(self,
                 times=[],
                 values=[],
                 timeseries=None,
                 fmin='auto',
                 fmax='auto',
                 num_freq_bins=None,
                 norm=True,
                 plot_fft=False
                 ):
        """

        """
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
            
        unique_time_diffs = np.unique(np.diff(times))
        if unique_time_diffs.size > 1:
            raise ValueError("Time series must have a uniform sampling interval.\n"
                            "Interpolate the data to a uniform grid first."
                        )
        self.dt = unique_time_diffs[0]
        
        if len(values.shape) == 2:
            self.freq, self.power, self.power_std = self.compute_stacked_power_spectrum(
                times, values, fmin=fmin, fmax=fmax, num_bins=num_freq_bins, norm=norm
            )
        else:
            self.freq, self.power, self.power_std = self.compute_power_spectrum(
                times, values, fmin=fmin, fmax=fmax, num_bins=num_freq_bins, norm=norm
            )

        if plot_fft:
            self.plot()

    def compute_power_spectrum(self, times, values, fmin='auto', fmax='auto', num_freq_bins=None, norm=True):
        """
        Computes the FFT of the values data and bins the output in frequency space.

        Returns:
        - binned_freqs (array-like): Binned frequencies.
        - binned_power (array-like): Binned power spectrum.
        """
        length = len(values)

        # Use absolute min and max frequencies if set to 'auto'
        fmin = 1 / (times.max() - times.min()) if self.fmin == 'auto' else self.fmin
        fmax = 1 / (2 * self.dt)if self.fmax == 'auto' else self.fmax # Nyquist frequency

        fft_vals = np.fft.fft(self.values)
        freqs = np.fft.fftfreq(length, d=self.dt)
        power = np.abs(fft_vals) ** 2

        # Filter frequencies within [fmin, fmax]
        valid_mask = (freqs >= fmin) & (freqs <= fmax)
        freqs = freqs[valid_mask]
        power = power[valid_mask]

        if num_freq_bins:
            freqs, power, power_error = self._bin_frequency_logspace(freqs, power, num_freq_bins)
        else:
            power_error = None

        # Normalize power spectrum to units of variance
        if norm:
            power /= length * np.mean(self.values) ** 2 / (2 * self.dt)
            
        return freqs, power, power_error
    
    def compute_stacked_power_spectrum(self, times, values, fmin='auto', fmax='auto', num_freq_bins=None, norm=True):
        for i in range(values.shape[0]):
            freqs, power, power_error = self.compute_power_spectrum(
                times, values[i], fmin=fmin, fmax=fmax, num_freq_bins=num_freq_bins, norm=norm
                )
            if i == 0:
                stacked_power = power
            else:
                stacked_power = np.vstack((stacked_power, power))

            # Weight final power spectrum by inverse error squared
            weights = 1 / power_error ** 2
            weights[np.isinf(weights)] = 0  # Handle cases where uncertainty is 0
            power_mean = np.average(stacked_power, axis=0, weights=weights)
            power_std = np.sum(weights * (stacked_power - power_mean)**2, axis=0) / np.sum(weights, axis=0)

        return freqs, power_mean, power_std
    
    def plot(self, **kwargs):
        """
        Plots the FFT power spectrum.

        Parameters:
        - freqs (array-like): Frequencies.
        - power (array-like): Power spectrum.
        - power_error (array-like, optional): Errors in power spectrum.
        - kwargs: Plot customization arguments.

        Keyword arguments:
        - figsize (tuple): Figure size (width, height).
        - title (str): Title of the plot.
        - xlabel (str): Label for the x-axis.
        - ylabel (str): Label for the y-axis.
        - xlim (tuple): Limits for the x-axis.
        - ylim (tuple): Limits for the y-axis.
        - fig_kwargs (dict): Additional keyword arguments for the figure function.
        - plot_kwargs (dict): Additional keyword arguments for the plot function.
        - major_tick_kwargs (dict): Additional keyword arguments for the tick_params function.
        - minor_tick_kwargs (dict): Additional keyword arguments for the tick_params function.
        """
        title = kwargs.get('title', None)

        # Default plotting settings
        if self.power_error is None:
            default_plot_kwargs = {'color': 'black', 'fmt': 'o', 'ms': 2, 'lw': 1, 'label': None}
        else:
            default_plot_kwargs = {'color': 'black', 's': 2, 'label': None}

        figsize = kwargs.get('figsize', (8, 5))
        fig_kwargs = {'figsize': figsize, **kwargs.pop('fig_kwargs', {})}
        plot_kwargs = {**default_plot_kwargs, **kwargs.pop('plot_kwargs', {})}
        major_tick_kwargs = {'which': 'major', **kwargs.pop('major_tick_kwargs', {})}
        minor_tick_kwargs = {'which': 'minor', **kwargs.pop('minor_tick_kwargs', {})}

        plt.figure(**fig_kwargs)

        if self.power_error is not None:
            plt.errorbar(self.freqs, self.power, yerr=self.power_error, **plot_kwargs)
        else:
            plt.scatter(self.freqs, self.power, **plot_kwargs)

        # Set labels and title
        plt.xlabel(kwargs.get('xlabel', 'Frequency (Hz)'))
        plt.ylabel(kwargs.get('ylabel', 'Power'))
        plt.xlim(kwargs.get('xlim', None))
        plt.ylim(kwargs.get('ylim', None))

        # Show legend if label is provided
        if plot_kwargs.get('label') is not None:
            plt.legend()

        if title is not None:
            plt.title(title)

        plt.tick_params(**major_tick_kwargs)
        if len(minor_tick_kwargs) > 1:
            plt.minorticks_on()
            plt.tick_params(**minor_tick_kwargs)

        plt.show()

    def bin(self, num_bins):
        """
        Bins the FFT output in frequency space.

        Parameters:
        - num_bins (int): Number of bins.

        Returns:
        - binned_freqs (array-like): Binned frequencies.
        - binned_power (array-like): Binned power spectrum.
        """
        self.freqs, self.power, self.power_error = self._bin_frequency_logspace(self.freqs, self.power, num_bins)
    
    def _bin_frequency_logspace(self, freqs, power, num_bins):
        """
        Bins the FFT output in frequency space.

        Parameters:
        - freqs (array-like): Frequencies.
        - power (array-like): Power spectrum.
        - num_bins (int): Number of bins.

        Returns:
        - binned_freqs (array-like): Binned frequencies.
        - binned_power (array-like): Binned power spectrum.
        """
        log_bins = np.logspace(np.log10(freqs.min()), np.log10(freqs.max()), num_bins + 1)
        binned_freqs = []
        binned_power = []
        binned_power_error = []

        for i in range(len(log_bins) - 1):
            mask = (freqs >= log_bins[i]) & (freqs < log_bins[i + 1])
            if mask.any():
                binned_freqs.append(freqs[mask].mean())
                binned_power.append(power[mask].mean())
                binned_power_error.append(power[mask].std())

        return np.array(binned_freqs), np.array(binned_power), np.array(binned_power_error)