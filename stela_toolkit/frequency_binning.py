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
        else:

            if bin_type == "log":
                # Define logarithmic bins
                bin_edges = np.logspace(np.log10(fmin), np.log10(fmax), num_bins + 1)

            elif bin_type == "linear":
                # Define linear bins
                bin_edges = np.linspace(fmin, fmax, num_bins + 1)

            else:
                raise ValueError(
                    f"Unsupported bin_type '{bin_type}'. Choose 'log', 'linear', or provide custom bins.")

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
        - binned_value_errors (array-like): Standard deviation of the values in each bin.
        """
        binned_freqs = []
        binned_freq_widths = []
        binned_values = []
        binned_value_errors = []

        for i in range(len(bin_edges) - 1):
            mask = (freqs >= bin_edges[i]) & (freqs < bin_edges[i + 1])
            num_freqs = np.sum(mask)

            if mask.any():
                lower_bound = bin_edges[i]
                upper_bound = bin_edges[i + 1]
                bin_cent = (upper_bound + lower_bound) / 2

                binned_freqs.append(bin_cent)
                binned_freq_widths.append(bin_cent - lower_bound)
                binned_values.append(
                    np.mean(values[mask])
                )
                binned_value_errors.append(
                    np.std(values[mask]) / np.sqrt(num_freqs)
                )

        return (
            np.array(binned_freqs),
            np.array(binned_freq_widths),
            np.array(binned_values),
            np.array(binned_value_errors),
        )

    @staticmethod
    def count_frequencies_in_bins(spectrum, fmin=None, fmax=None, num_bins=None, bin_type=None, bin_edges=[]):
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
        fmin = spectrum.fmin if fmin is None else fmin
        fmax = spectrum.fmax if fmax is None else fmax
        num_bins = spectrum.num_bins if num_bins is None else num_bins
        bin_type = spectrum.bin_type if bin_type is None else bin_type
        bin_edges = spectrum.bin_edges if bin_edges is None else bin_edges

        # Define time array from input class
        if hasattr(spectrum, 'times'):  
            times = spectrum.times
        elif hasattr(spectrum, 'times1'):
            times = spectrum.times1
        else:
            raise AttributeError('Input class object for frequency binning does not have a time array properly defined.')

        length = len(times) 
        dt = np.diff(times)[0]
        freqs = np.fft.rfftfreq(length, d=dt)
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        freqs = freqs[freq_mask]

        # if neither num_bins nor bin_edges have been provided, no binning
        if not any([num_bins, bin_edges]):
            return np.ones(len(freqs))
            
        # Check if bin_edges or num_bins provided
        if len(bin_edges) == 0 and fmin and fmax and num_bins:
            bin_edges = FrequencyBinning.define_bins(fmin, fmax, num_bins=num_bins, 
                                                     bin_type=bin_type, bin_edges=bin_edges
                                                    )
        elif len(bin_edges) > 0:
            bin_edges = np.array(bin_edges)
        else:
            raise ValueError(
                "Frequency binning requires either 1) defined bin edges, 2) num_bins + fmin + fmax., \
                3) all defined as none to leave products unbinned."
            )

        # Count frequencies in bins
        bin_counts = np.histogram(freqs, bins=bin_edges)[0]
        return bin_counts
