import numpy as np
from .power_spectrum import PowerSpectrum
from .spectrum_plotter import SpectrumPlotter
from .frequency_binning import FrequencyBinning


class CrossSpectrum(PowerSpectrum):
    def __init__(self,
                 times1=[],
                 values1=[],
                 times2=[],
                 values2=[],
                 timeseries1=None,
                 timeseries2=None,
                 fmin='auto',
                 fmax='auto',
                 num_bins=None,
                 bin_type="log",
                 norm=True,
                 plot_fft=False):
        """
        Initializes the CrossSpectrum class.

        Parameters:
        - times1, values1: Time and value arrays for the first time series.
        - times2, values2: Time and value arrays for the second time series.
        - timeseries1, timeseries2: TimeSeries objects for the two time series.
        - fmin, fmax: Frequency range for the cross-spectrum analysis.
        - num_bins: Number of frequency bins.
        - bin_type: Type of binning ("log" or "linear").
        - norm: Normalize the cross-spectrum to the variance.
        - plot_fft: Whether to plot the cross-spectrum.
        """
        self.times1, self.values1 = self._check_input(timeseries1, times1, values1)
        self.times2, self.values2 = self._check_input(timeseries2, times2, values2)

        if not np.allclose(self.times1, self.times2):
            raise ValueError("The time arrays of the two time series must be identical.")

        self.fmin = fmin
        self.fmax = fmax
        self.num_bins = num_bins
        self.bin_type = bin_type

        # Check if the input values are for multiple realizations
        if len(self.values1.shape) == 2 and len(self.values2.shape) == 2:
            self.freqs, self.freq_widths, self.cross_powers, self.cross_power_sigmas = self.compute_stacked_cross_spectrum(
                fmin=self.fmin, fmax=self.fmax, num_bins=self.num_bins, bin_type=self.bin_type, norm=norm
            )
        else:
            self.freqs, self.freq_widths, self.cross_powers, self.cross_power_sigmas = self.compute_cross_spectrum(
                fmin=self.fmin, fmax=self.fmax, num_bins=self.num_bins, bin_type=self.bin_type, norm=norm
            )

        if plot_fft:
            self.plot()

    def _check_input(self, timeseries, times, values):
        """
        Validates and extracts time and value arrays from input or TimeSeries objects.
        """
        if timeseries:
            if not isinstance(timeseries, PowerSpectrum.TimeSeries):
                raise TypeError("timeseries must be an instance of the TimeSeries class.")
            times = timeseries.times
            values = timeseries.values
        elif len(times) > 0 and len(values) > 0:
            times = np.array(times)
            values = np.array(values)
            if len(values.shape) == 1 and len(times) != len(values):
                raise ValueError("Times and values must have the same length.")
            elif len(values.shape) == 2 and values.shape[1] != len(times):
                raise ValueError(
                    "Times and values must have the same length for each time series.\n"
                    "Check the shape of the values array: expecting (n_series, n_times)."
                )
        else:
            raise ValueError("Either provide a TimeSeries object or times and values arrays.")
        return times, values

    def compute_cross_spectrum(self, fmin='auto', fmax='auto', num_bins=None, bin_type="log", norm=True):
        """
        Computes the cross-spectrum between two time series.

        Returns:
        - freqs: Frequencies of the cross-spectrum.
        - freq_widths: Widths of the frequency bins (if binning is applied).
        - cross_power: Cross power spectrum values.
        - cross_power_sigma: Uncertainties in cross power spectrum (if binning is applied).
        """
        # Compute individual power spectra with binning via PowerSpectrum
        ps1 = PowerSpectrum(times=self.times1, values=self.values1, fmin=fmin, fmax=fmax, num_bins=num_bins, bin_type=bin_type, norm=norm)
        ps2 = PowerSpectrum(times=self.times2, values=self.values2, fmin=fmin, fmax=fmax, num_bins=num_bins, bin_type=bin_type, norm=norm)

        # Cross-spectrum computation
        cross_power = np.real(ps1.powers * np.conj(ps2.powers))
        return ps1.freqs, ps1.freq_widths, cross_power, None

    def compute_stacked_cross_spectrum(self, fmin='auto', fmax='auto', num_bins=None, bin_type="log", norm=True):
        """
        Computes the stacked cross-spectrum across multiple GP realizations.

        Returns:
        - freqs: Frequencies of the cross-spectrum.
        - freq_widths: Widths of the frequency bins.
        - cross_power_mean: Mean cross power across realizations.
        - cross_power_std: Standard deviation of the cross power across realizations.
        """
        cross_powers = []

        for i in range(self.values1.shape[0]):
            ps1 = PowerSpectrum(times=self.times1, values=self.values1[i], fmin=fmin, fmax=fmax, num_bins=num_bins, bin_type=bin_type, norm=norm)
            ps2 = PowerSpectrum(times=self.times2, values=self.values2[i], fmin=fmin, fmax=fmax, num_bins=num_bins, bin_type=bin_type, norm=norm)

            # Compute cross-spectrum for each realization
            cross_power = np.real(ps1.powers * np.conj(ps2.powers))
            cross_powers.append(cross_power)

        # Stack the collected cross powers
        cross_powers = np.vstack(cross_powers)

        # Compute mean and std of cross-powers
        cross_power_mean = np.mean(cross_powers, axis=0)
        cross_power_std = np.std(cross_powers, axis=0)

        return ps1.freqs, ps1.freq_widths, cross_power_mean, cross_power_std

    def bin(self, num_bins=None, bin_type="log", bin_edges=None, plot=False, save=True, verbose=True):
        """
        Bins the cross-spectrum data.

        Parameters:
        - num_bins: Number of bins (if `bins` is not provided).
        - bin_type: Type of binning ("log" or "linear").
        - bin_edges: Custom array of bin edges (optional).
        - plot: If True, plots the binned data.
        - save: If True, updates the internal attributes with binned data.
        - verbose: If True, prints information about binning.
        """
        if bin_edges is None:
            try:
                bin_edges = FrequencyBinning.define_bins(self.freqs, num_bins=num_bins, bins=bin_edges, bin_type=bin_type)
            except ValueError:
                raise ValueError("Either num_bins or bin_edges must be provided.")

        if verbose:
            num_freqs_in_bins = FrequencyBinning.count_frequencies_in_bins(self.freqs, bin_edges)
            print(f"Number of frequencies in each bin: {num_freqs_in_bins}")

        freqs, freq_widths, cross_powers, cross_power_sigmas = FrequencyBinning.bin_data(self.freqs, self.cross_powers, bin_edges)

        if plot:
            self.plot(x=freqs, y=cross_powers, xerr=freq_widths, yerr=cross_power_sigmas)

        if save:
            self.freqs, self.freq_widths, self.cross_powers, self.cross_power_sigmas = freqs, freq_widths, cross_powers, cross_power_sigmas

    def plot(self, **kwargs):
        """
        Plots the cross-spectrum using the SpectrumPlotter.

        Parameters:
        - **kwargs: Keyword arguments for customizing the plot.
        """
        kwargs.setdefault('xlabel', 'Frequency')
        kwargs.setdefault('ylabel', 'Cross Power')
        kwargs.setdefault('xscale', 'log')
        kwargs.setdefault('yscale', 'log')
        SpectrumPlotter.plot(x=self.freqs, y=self.cross_powers, xerr=self.freq_widths, yerr=self.cross_power_sigmas, **kwargs)
