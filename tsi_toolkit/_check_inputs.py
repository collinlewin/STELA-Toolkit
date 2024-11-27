import numpy as np

from .data_loader import TimeSeries


class _CheckInputs:
    """
    A utility class for checking and validating input data and binning.
    """
    @staticmethod
    def _check_input_data(timeseries, times, values, sigmas=None):
        """
        Validates and extracts time and value arrays from input or TimeSeries objects.
        """
        if timeseries:
            # using methods to allow flexible import of TimeSeries objects
            if not all(callable(getattr(timeseries, method, None)) for method in ["load_file", "load_fits"]):
                raise TypeError("timeseries must be an instance of the TimeSeries class.")
            times = timeseries.times
            values = timeseries.values
            sigmas = timeseries.errors

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
            
            if sigmas: 
                if np.min(sigmas) <= 0:
                    raise ValueError("Uncertainties of the input data must be positive.")
        else:
            raise ValueError("Either provide a TimeSeries object or times and values arrays.")
        
        # check for regular sampling
        time_sampling = np.round(np.diff(times),10)
        if np.unique(time_sampling).size > 1:
            raise ValueError("Time series must have a uniform sampling interval.\n"
                            "Interpolate the data to a uniform grid first."
                        )
        
        return times, values, sigmas
    
    def _check_input_bins(num_bins, bin_type, bin_edges):
        """
        Validates and returns bin edges for frequency binning.
        """
        if len(bin_edges) > 0:
            # Use custom bins
            if np.diff(bin_edges) <= 0:
                raise ValueError("Custom bin edges must be monotonically increasing.")
            if num_bins is not None:
                print("Custom bin_edges detected: num_bins is ignored when custom bins are provided.")

        elif num_bins is not None:
            if not isinstance(num_bins, int) or num_bins < 1:
                raise ValueError("Number of bins (num_bins) must be a positive integer.")
            if bin_type is None:
                raise ValueError("bin_type must be provided if num_bins is used.")

        if bin_type not in ["log", "linear"]:
            raise ValueError(f"Unsupported bin_type '{bin_type}'. Choose 'log', 'linear', or provide custom bins.")  