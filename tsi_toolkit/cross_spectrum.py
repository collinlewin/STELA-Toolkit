import numpy as np

from ._check_inputs import _CheckInputs
from .power_spectrum import PowerSpectrum
from .plot import Plotter
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
                 bin_edges = [],
                 norm=True,
                 plot_cs=False):
        """
        Initializes the CrossSpectrum object and computes the cross-spectrum. It determines whether the input represents
        individual time series or multiple realizations and computes the corresponding
        cross-spectrum using FFT.

        Parameters:
        - times1 (array-like): Time values for the first time series.
        - values1 (array-like): Measurement values for the first time series (e.g., flux, counts).
        - times2 (array-like): Time values for the second time series.
        - values2 (array-like): Measurement values for the second time series (e.g., flux, counts).
        - timeseries1 (TimeSeries): First time series object (optional).
        - timeseries2 (TimeSeries): Second time series object (optional).
        - fmin (float or 'auto'): Minimum frequency for the cross-spectrum.
        - fmax (float or 'auto'): Maximum frequency for the cross-spectrum.
        - num_bins (int): Number of bins for frequency binning.
        - bin_type (str): Binning type ('log' or 'linear').
        - norm (bool): Whether to normalize the spectrum.
        - plot_cs (bool): Whether to plot the cross-spectrum after creation.
        """
        self.times1, self.values1 = self._check_input(timeseries1, times1, values1)
        self.times2, self.values2 = self._check_input(timeseries2, times2, values2)
        _CheckInputs._check_input_bins(num_bins, bin_type, bin_edges)

        if not np.allclose(self.times1, self.times2):
            raise ValueError("The time arrays of the two time series must be identical.")

        self.fmin = fmin
        self.fmax = fmax
        self.num_bins = num_bins
        self.bin_type = bin_type

        # Check if the input values are for multiple realizations
        if len(self.values1.shape) == 2 and len(self.values2.shape) == 2:
            cross_spec = self.compute_stacked_cross_spectrum(fmin=self.fmin, fmax=self.fmax, num_bins=self.num_bins,
                                                             bin_type=self.bin_type, bin_edges=bin_edges, norm=norm
                                                             )
        else:
            cross_spec = self.compute_cross_spectrum(fmin=self.fmin, fmax=self.fmax, num_bins=self.num_bins, 
                                                     bin_type=self.bin_type, bin_edges=bin_edges, norm=norm
                                                     )
        self.freqs, self.freq_widths, self.cross_powers, self.cross_power_sigmas = cross_spec

        if plot_cs:
            self.plot()

    def compute_cross_spectrum(self, fmin='auto', fmax='auto', num_bins=None, bin_type="log",
                               bin_edges=[], norm=True):
        """
        Computes the cross-spectrum between two time series.
        The method calculates the cross-spectrum as the real part of the product
        of the FFT of the first time series and the conjugate FFT of the second.

        Parameters:
        - fmin (float or 'auto'): Minimum frequency for computation.
        - fmax (float or 'auto'): Maximum frequency for computation.
        - num_bins (int): Number of bins for frequency binning.
        - bin_type (str): Type of binning ('log' or 'linear').
        - norm (bool): Whether to normalize the spectrum.

        Returns:
        - freqs (array-like): Frequencies of the cross-spectrum.
        - freq_widths (array-like): Bin widths of the frequencies.
        - cross_power (array-like): Cross-power spectrum values.
        - cross_power_sigma (array-like or None): Uncertainties in cross-power values.
        """
        ps1 = PowerSpectrum(
            times=self.times1, values=self.values1, fmin=fmin, fmax=fmax,
            num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges, norm=norm
        )
        ps2 = PowerSpectrum(
            times=self.times2, values=self.values2, fmin=fmin, fmax=fmax,
            num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges, norm=norm
        )

        cross_power = np.real(ps1.powers * np.conj(ps2.powers))
        return ps1.freqs, ps1.freq_widths, cross_power, None

    def compute_stacked_cross_spectrum(self, fmin='auto', fmax='auto', num_bins=None, bin_type="log", 
                                       bin_edges=[], norm=True):
        """
        Computes the cross-spectrum for multiple realizations.
        For multiple realizations (e.g., GP samples), this method computes the
        cross-spectrum for each realization pair. The resulting cross-spectra are
        averaged to compute the mean and standard deviation for each frequency bin.

        Parameters:
        - fmin (float or 'auto'): Minimum frequency for computation.
        - fmax (float or 'auto'): Maximum frequency for computation.
        - num_bins (int): Number of bins for frequency binning.
        - bin_type (str): Type of binning ('log' or 'linear').
        - norm (bool): Whether to normalize the spectrum.

        Returns:
        - freqs (array-like): Frequencies of the cross-spectrum.
        - freq_widths (array-like): Bin widths of the frequencies.
        - cross_power_mean (array-like): Mean cross-power spectrum values.
        - cross_power_std (array-like): Standard deviation of cross-power values.
        """
        cross_powers = []
        for i in range(self.values1.shape[0]):
            ps1 = PowerSpectrum(
                times=self.times1, values=self.values1[i], fmin=fmin, fmax=fmax, 
                num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges, norm=norm
            )
            ps2 = PowerSpectrum(
                times=self.times2, values=self.values2[i], fmin=fmin, fmax=fmax,
                num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges, norm=norm
            )

            # Compute cross-spectrum for each pair of realizations
            cross_power = np.real(ps1.powers * np.conj(ps2.powers))
            cross_powers.append(cross_power)

        cross_powers = np.vstack(cross_powers)
        cross_power_mean = np.mean(cross_powers, axis=0)
        cross_power_std = np.std(cross_powers, axis=0)

        return ps1.freqs, ps1.freq_widths, cross_power_mean, cross_power_std

    def bin(self, num_bins=None, bin_type="log", bin_edges=None, plot=False, save=True, verbose=True):
        """
        Bins the cross-spectrum data into specified bins.
        Optionally, saves the binned data back to the object and plots
        the binned spectrum.

        Parameters:
        - num_bins (int): Number of bins (if `bin_edges` is not provided).
        - bin_type (str): Type of binning ('log' or 'linear').
        - bin_edges (array-like): Custom bin edges (optional).
        - plot (bool): Whether to plot the binned data.
        - save (bool): Whether to save the binned data back to the object.
        - verbose (bool): Whether to print details about the binning process.
        """
        if bin_edges is None:
            try:
                bin_edges = FrequencyBinning.define_bins(
                    self.freqs, num_bins=num_bins, bins=bin_edges, bin_type=bin_type
                )
            except ValueError:
                raise ValueError("Either num_bins or bin_edges must be provided.")

        if verbose:
            num_freqs_in_bins = FrequencyBinning.count_frequencies_in_bins(self.freqs, bin_edges)
            print(f"Number of frequencies in each bin: {num_freqs_in_bins}")

        freqs, freq_widths, cross_powers, cross_power_sigmas = FrequencyBinning.bin_data(
            self.freqs, self.cross_powers, bin_edges
        )

        if plot:
            self.plot(x=freqs, y=cross_powers, xerr=freq_widths, yerr=cross_power_sigmas)

        if save:
            self.freqs = freqs
            self.freq_widths = freq_widths
            self.cross_powers = cross_powers
            self.cross_power_sigmas = cross_power_sigmas

    def plot(self, **kwargs):
        """
        Plots the cross-spectrum.

        Parameters:
        - **kwargs: Keyword arguments for customizing the plot.
        """
        kwargs.setdefault('xlabel', 'Frequency')
        kwargs.setdefault('ylabel', 'Cross Power')
        kwargs.setdefault('xscale', 'log')
        kwargs.setdefault('yscale', 'log')
        Plotter.plot(
            x=self.freqs, y=self.cross_powers, xerr=self.freq_widths, yerr=self.cross_power_sigmas, **kwargs
        )

    def count_frequencies_in_bins(self, bin_edges=None, num_bins=None, bin_type="log"):
        """
        Counts the number of frequencies in each bin for the cross-spectrum.

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