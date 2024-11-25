import numpy as np
import matplotlib.pyplot as plt


class FrequencyBinning:
    """
    A utility class for binning data over frequency space.

    Provides methods for defining bins (linear or logarithmic, or user-defined), binning frequency
    data and corresponding values, and calculating statistics for binned data.
    """
    # To do: Modify count_frequencies_in_bins for already binned data
    @staticmethod
    def define_bins(fmin, fmax, num_bins=None, bin_type="log", bin_edges=[]):
        """
        Defines bin edges for the given frequencies based on the specified binning type.

        If custom bins are provided, they are used directly. Otherwise, bins are computed
        either logarithmically or linearly based on the specified bin type.

        Parameters:
        - freqs (array-like): Array of frequencies to define bins for.
        - num_bins (int): Number of bins to create (ignored if `bins` is provided).
        - bins (array-like): Custom array of bin edges (optional).
        - bin_type (str): Type of binning ("log" for logarithmic, "linear" for linear).

        Returns:
        - bin_edges (array-like): Array of bin edges for the specified binning type.
        """
        if len(bin_edges) > 0:
            # Use custom bins
            bin_edges = np.array(bin_edges)
        elif bin_type == "log":
            # Define logarithmic bins
            bin_edges = np.logspace(np.log10(fmin), np.log10(fmax), num_bins + 1)
        elif bin_type == "linear":
            # Define linear bins
            bin_edges = np.linspace(fmin, fmax, num_bins + 1)
        else:
            raise ValueError(f"Unsupported bin_type '{bin_type}'. Choose 'log', 'linear', or provide custom bins.")

        return bin_edges

    @staticmethod
    def bin_data(freqs, values, bin_edges):
        """
        Bins frequencies and corresponding values into specified bins.

        Parameters:
        - freqs (array-like): Array of frequencies to be binned.
        - values (array-like): Array of values corresponding to the frequencies.
        - bin_edges (array-like): Array of bin edges defining the bins.

        Returns:
        - binned_freqs (array-like): Mean frequency for each bin.
        - binned_freq_widths (array-like): Half-widths of the frequency bins (for error bars).
        - binned_values (array-like): Mean value for each bin.
        - binned_value_sigmas (array-like): Standard deviation of the values in each bin.
        """
        binned_freqs = []
        binned_freq_widths = []
        binned_values = []
        binned_value_sigmas = []
        for i in range(len(bin_edges) - 1):
            mask = (freqs >= bin_edges[i]) & (freqs < bin_edges[i + 1])
            if mask.any():
                binned_freqs.append(freqs[mask].mean())
                binned_values.append(values[mask].mean())
                binned_value_sigmas.append(values[mask].std())

                # Calculate bin half-widths for error bars
                lower_bound = bin_edges[i]
                upper_bound = bin_edges[i + 1]
                binned_freq_widths.append((upper_bound - lower_bound) / 2)

        return (
            np.array(binned_freqs),
            np.array(binned_freq_widths),
            np.array(binned_values),
            np.array(binned_value_sigmas),
        )
    
    @staticmethod
    def count_frequencies_in_bins(spectrum, fmin=None, fmax=None, num_bins=None, bin_type="log", bin_edges=[]):
        """
        Counts the number of frequencies in each bin for the power spectrum.

        Parameters:
        - spectrum: The object containing attributes like `times`, `fmin`, and `fmax`.
        - fmin: Minimum frequency (optional, falls back to spectrum's attribute).
        - fmax: Maximum frequency (optional, falls back to spectrum's attribute).
        - num_bins: Number of bins to create (if bin_edges is not provided).
        - bin_type: Type of binning ("log" or "linear").
        - bin_edges: Custom array of bin edges (optional).

        Returns:
        - bin_counts: List of counts of frequencies in each bin.
        """
        # Use spectrum's attributes if not provided
        if fmin is None:
            fmin = spectrum.fmin
        if fmax is None:
            fmax = spectrum.fmax

        # Check if bin_edges or num_bins provided
        if len(bin_edges) == 0 and num_bins is None:
            bin_edges = FrequencyBinning.define_bins(
                spectrum.fmin, spectrum.fmax, num_bins=spectrum.num_bins, 
                bin_type=spectrum.bin_type, bin_edges=spectrum.bin_edges
            )
        elif num_bins is not None:
            bin_edges = FrequencyBinning.define_bins(
                fmin, fmax, num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges
            )
        elif len(bin_edges) > 0:
            bin_edges = np.array(bin_edges)
        else:
            raise ValueError("Either num_bins or bin_edges must be provided.")

        # Use spectrum's times attribute
        length = len(spectrum.times)
        dt = np.diff(spectrum.times)[0]
        freqs = np.fft.fftfreq(length, d=dt)

        # Count frequencies in bins
        bin_counts = np.histogram(freqs, bins=bin_edges)[0]
        return bin_counts