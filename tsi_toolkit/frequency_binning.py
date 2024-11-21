import numpy as np
import matplotlib.pyplot as plt


class FrequencyBinning:
    """
    A utility class for binning data over frequency space, with methods for logarithmic
    binning and calculating statistics for binned data.
    """
    # To do: Modify count_frequencies_in_bins for already binned data
    @staticmethod
    def define_bins(freqs, num_bins=None, bins=None, bin_type="log"):
        """
        Defines bins based on the specified binning type or user-defined edges.

        Parameters:
        - freqs: Array of frequencies to define bins for.
        - num_bins: Number of bins to create (ignored if `bins` is provided).
        - bins: Custom array of bin edges (optional).
        - bin_type: Type of binning ("log" for logarithmic, "linear" for linear).

        Returns:
        - bin_edges: Array of bin edges.
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
        Bins frequencies and associated values based on provided bin edges.

        Parameters:
        - freqs: Array of frequencies to be binned.
        - values: Array of values corresponding to the frequencies.
        - bin_edges: Array of bin edges.

        Returns:
        - binned_freqs: Mean frequency for each bin.
        - binned_freq_widths: Half-widths of frequency bins (for error bars).
        - binned_values: Mean value for each bin.
        - binned_value_sigmas: Standard deviation of values in each bin.
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
        Computes the number of frequencies in each bin based on the provided bin edges.

        Parameters:
        - freqs: Array of frequencies to be binned.
        - bin_edges: Array of bin edges defining the bins.

        Returns:
        - n_freqs_in_bin: List of counts of frequencies in each bin.
        """
        n_freqs_in_bin = []
        for i in range(len(bin_edges) - 1):
            in_bin = (freqs >= bin_edges[i]) & (freqs < bin_edges[i + 1])
            count = int(np.sum(in_bin))
            n_freqs_in_bin.append(count)

        return n_freqs_in_bin