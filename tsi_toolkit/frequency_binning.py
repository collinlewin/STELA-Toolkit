import numpy as np
import matplotlib.pyplot as plt


class FrequencyBinning:
    """
    A utility class for binning data over frequency space, with methods for logarithmic
    binning and calculating statistics for binned data.
    """
    @staticmethod
    def number_frequencies_in_bin(freqs, num_bins):
        """
        Computes the number of frequencies in each bin for a logarithmic binning scheme.

        Parameters:
        - freqs: Array of frequencies to be binned.
        - num_bins: Number of logarithmic bins.

        Returns:
        - n_freqs_in_bin: List of counts of frequencies in each bin.
        """
        log_bins = np.logspace(np.log10(freqs.min()), np.log10(freqs.max()), num_bins + 1)
        n_freqs_in_bin = [
            int(np.sum((freqs >= log_bins[i]) & (freqs < log_bins[i + 1])))
            for i in range(len(log_bins) - 1)
        ]
        return n_freqs_in_bin

    @staticmethod
    def bin_frequency_logspace(freqs, values, num_bins):
        """
        Bins frequencies and associated values (e.g., power spectra) in logarithmic space.

        Parameters:
        - freqs: Array of frequencies to be binned.
        - values: Array of values (e.g., power, flux) corresponding to the frequencies.
        - num_bins: Number of logarithmic bins.

        Returns:
        - binned_freqs: Mean frequency for each bin.
        - binned_freq_widths: Half-widths of frequency bins for error bars (xerr).
        - binned_values: Mean value for each bin.
        - binned_value_sigmas: Standard deviation of values in each bin.
        """
        log_bins = np.logspace(np.log10(freqs.min()), np.log10(freqs.max()), num_bins + 1)
        binned_freqs = []
        binned_freq_widths = []
        binned_values = []
        binned_value_sigmas = []

        for i in range(len(log_bins) - 1):
            mask = (freqs >= log_bins[i]) & (freqs < log_bins[i + 1])
            if mask.any():
                binned_freqs.append(freqs[mask].mean())
                binned_values.append(values[mask].mean())
                binned_value_sigmas.append(values[mask].std())

                # Calculate bin half-widths for error bars
                lower_bound = log_bins[i]
                upper_bound = log_bins[i + 1]
                binned_freq_widths.append((upper_bound - lower_bound) / 2)

        return (
            np.array(binned_freqs),
            np.array(binned_freq_widths),
            np.array(binned_values),
            np.array(binned_value_sigmas),
        )
