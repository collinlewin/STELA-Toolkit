import numpy as np

from .timeseries import TimeSeries


class _CheckInputs:
    """
    A utility class for checking and validating input data and binning.
    """
    @staticmethod
    def _check_input_data(timeseries, times, values):
        """
        Validates and extracts time and value arrays from input or TimeSeries objects.
        """
        if timeseries:
            if not isinstance(timeseries, TimeSeries):
                raise TypeError("timeseries must be an instance of the TimeSeries class.")
            times = timeseries.times
            values = timeseries.values

        # check input arrays if not timeseries object
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
        
        # check for regular sampling
        time_sampling = np.round(np.diff(times),10)
        if np.unique(time_sampling).size > 1:
            raise ValueError("Time series must have a uniform sampling interval.\n"
                            "Interpolate the data to a uniform grid first."
                        )
        
        return times, values
    
    def _check_input_bins(num_bins, bin_type, bin_edges):
        """
        Validates and returns bin edges for frequency binning.
        """
        if bin_edges is not None:
            # Use custom bins
            if np.diff(bin_edges) <= 0:
                raise ValueError("Custom bin edges must be monotonically increasing.")
            if num_bins is not None:
                print("Custom bin_edges detected: num_bins is ignored when custom bins are provided.")

        else:
            if num_bins is None:
                raise ValueError("Number of bins (num_bins) must be provided if custom bins are not used.")
            elif not isinstance(num_bins, int) or num_bins < 1:
                raise ValueError("Number of bins (num_bins) must be a positive integer.")
            else:
                if bin_type is None:
                    raise ValueError("bin_type must be provided if num_bins is used.")

        if bin_type not in ["log", "linear"]:
            raise ValueError(f"Unsupported bin_type '{bin_type}'. Choose 'log', 'linear', or provide custom bins.")  