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
    def define_bins(freqs, num_bins=None, bins=None, bin_type="log"):
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
        if bins is not None:
            # Use custom bins
            bin_edges = np.array(bins)
        elif bin_type == "log":
            # Define logarithmic bins
            bin_edges = np.logspace(np.log10(freqs.min()), np.log10(freqs.max()), num_bins + 1)
        elif bin_type == "linear":
            # Define linear bins
            bin_edges = np.linspace(freqs.min(), freqs.max(), num_bins + 1)
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
    def count_frequencies_in_bins(freqs, bin_edges):
        """
        Counts the number of frequencies in each bin defined by the provided edges.

        Parameters:
        - freqs (array-like): Array of frequencies to count.
        - bin_edges (array-like): Array of bin edges defining the bins.

        Returns:
        - n_freqs_in_bin (list): List of counts of frequencies in each bin.
        """
        n_freqs_in_bin = []
        for i in range(len(bin_edges) - 1):
            in_bin = (freqs >= bin_edges[i]) & (freqs < bin_edges[i + 1])
            count = int(np.sum(in_bin))
            n_freqs_in_bin.append(count)

        return n_freqs_in_bin