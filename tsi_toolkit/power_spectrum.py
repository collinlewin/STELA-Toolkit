import numpy as np
import matplotlib.pyplot as plt

from .data_loader import TimeSeries


class PowerSpectrum:
    def __init__(self,
                 times=[],
                 values=[],
                 timeseries=None,
                 gp_samples=None,
                 norm=True,
                 fmin=None,
                 fmax=None,
                 num_bins=None,
                 plot_fft=True
                 ):
        """

        """
        if timeseries:
            if not isinstance(timeseries, TimeSeries):
                raise TypeError("timeseries must be an instance of the TimeSeries class.")
            times = timeseries.times
            values = timeseries.values
            errors = timeseries.errors
            time_diffs = np.diff(times)
        elif gp_samples:
            # need to implement this
            # self.dt needs to be defined properly for GPs
            # find out how time information will be entered for GP samples
        else:
            raise ValueError("Either provide a TimeSeries object, times and values arrays, or array of GP samples.")
            
        if list(set(time_diffs)).size > 1:
            raise ValueError("Time series must have a uniform sampling interval."
                            "Interpolate the data to a uniform grid first."
                        )
        self.dt = time_diffs[0]
        self.norm = norm

        if gp_samples:
            self.freq, self.power, self.power_error = self.compute_gp_fft(gp_samples, fmin=fmin, fmax=fmax, num_bins=num_bins)
        else:
            self.freq, self.power, self.power_error = self.compute_fft(times, values, fmin=fmin, fmax=fmax, num_bins=num_bins)

        if plot_fft:
            self.plot()

    def compute_fft(self, times, values, fmin='auto', fmax='auto', num_bins=None):
        """
        Computes the FFT of the values data and bins the output in frequency space.

        Returns:
        - binned_freqs (array-like): Binned frequencies.
        - binned_power (array-like): Binned power spectrum.
        """
        n = len(values)

        # Calculate default fmin and fmax if set to 'auto'
        fmin = 1 / (times.max() - times.min()) if self.fmin == 'auto' else self.fmin
        fmax = 1 / (2 * self.dt) if self.fmax == 'auto' else self.fmax

        # Compute FFT
        fft_vals = np.fft.fft(self.values)
        freqs = np.fft.fftfreq(n, d=self.dt)
        power = np.abs(fft_vals) ** 2

        # Filter frequencies within [fmin, fmax]
        valid_mask = (freqs >= fmin) & (freqs <= fmax)
        freqs = freqs[valid_mask]
        power = power[valid_mask]

        #if self.norm:
        #    power /= n ** 2 # need to do this

        if num_bins:
            freqs, power, power_error = self._bin_frequency_logspace(freqs, power, num_bins)
        else:
            power_error = None
            
        return freqs, power, power_error
    
    def compute_gp_fft(self, gp_samples, fmin=None, fmax=None, num_bins=None):
        for sample in gp_samples:


    
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