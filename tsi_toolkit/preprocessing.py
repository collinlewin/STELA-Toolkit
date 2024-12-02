import numpy as np
import matplotlib.pyplot as plt


class Preprocessing:
    """
    Provides utility functions for preprocessing time series data.

    This class includes methods for standardizing and unstandardizing data,
    trimming time segments, removing NaN values and outliers, and detrending
    time series using polynomial fitting. Designed to operate directly on
    TimeSeries objects.
    """
    @staticmethod
    def standardize(timeseries):
        """
        Standardizes the time series data. 
        Stores the original mean and standard deviation for future unstandardization.
        """
        ts = timeseries
        if np.isclose(ts.mean, 0, atol=1e-10) and np.isclose(ts.std, 1, atol=1e-10):
            if not hasattr(ts, "unstandard_mean") and not hasattr(ts, "unstandard_std"):
                ts.unstandard_mean = 0
                ts.unstandard_std = 1
            print("The data is already standardized.")
        else:
            ts.unstandard_mean = ts.mean
            ts.unstandard_std = ts.std
            ts.values = (ts.values - ts.unstandard_mean) / ts.unstandard_std
            if ts.sigmas.size > 0:
                ts.sigmas = ts.sigmas / ts.unstandard_std

    @staticmethod
    def unstandardize(timeseries):
        """
        Unstandardizes the time series data.
        Restores the values and sigmas of the input TimeSeries object to their
        unstandardized form using the previously stored mean and standard deviation.
        """
        ts = timeseries
        try:
            ts.values = (ts.values * ts.unstandard_std) + ts.unstandard_mean
        except AttributeError:
            raise AttributeError(
                "The data has not been standardized yet. "
                "Please call the 'standardize' method first."
            )
        if ts.sigmas.size > 0:
            ts.sigmas = ts.sigmas * ts.unstandard_std

    @staticmethod
    def trim_time_segment(timeseries, start_time=None, end_time=None, plot=False, save=True):
        """
        Trims the time series data to a specified time range.

        Filters the time, value, and sigma arrays based on the provided start and
        end times. Optionally plots the data before and after trimming.

        Parameters:
        - timeseries (TimeSeries): The time series object to trim.
        - start_time (float): The starting time for the range (default: first time point).
        - end_time (float): The ending time for the range (default: last time point).
        - plot (bool): Whether to plot the data before and after trimming.
        - save (bool): Whether to modify the original TimeSeries object in place.

        Raises:
        - ValueError: If neither start_time nor end_time is provided.
        """
        ts = timeseries

        if start_time is None:
            start_time = ts.times[0]
        if end_time is None:
            end_time = ts.times[-1]
        if start_time and end_time is None:
            raise ValueError("Please specify a start and/or end time.")
        
        # Apply mask to trim data
        mask = (ts.times >= start_time) & (ts.times <= end_time)
        if plot:
            if ts.sigmas.size > 0:
                plt.errorbar(ts.times[mask], ts.values[mask], yerr=ts.sigmas[mask], fmt='o', lw=1, ms=2, color='black', label='Kept Data')
                plt.errorbar(ts.times[~mask], ts.values[~mask], yerr=ts.sigmas[~mask], fmt='o', lw=1, ms=2, color='red', label='Trimmed Data')
            else:
                plt.scatter(ts.times[mask], ts.values[mask], s=2, color="black", label="Kept Data")
                plt.scatter(ts.times[~mask], ts.values[~mask], s=2, color="red", label="Trimmed Data")

            plt.xlabel("Time")
            plt.ylabel("Values")
            plt.title("Trimming")
            plt.legend()
            plt.show()
        
        if save:
            ts.times = ts.times[mask]
            ts.values = ts.values[mask]
            if ts.sigmas.size > 0:
                ts.sigmas = ts.sigmas[mask]

    @staticmethod
    def remove_nans(timeseries, verbose=True):
        """
        Removes rows with NaN values in time, value, or sigma arrays.

        Parameters:
        - timeseries (TimeSeries): The time series object to clean.
        - verbose (bool): Whether to print the number of NaN points removed.
        """
        ts = timeseries
        if ts.sigmas.size > 0:
            nonnan_mask = ~np.isnan(ts.values) & ~np.isnan(ts.times) & ~np.isnan(ts.sigmas)
        else:
            nonnan_mask = ~np.isnan(ts.values) & ~np.isnan(ts.times)

        if verbose:
            print(f"Removed {np.sum(~nonnan_mask)} NaN points.\n"
                  f"({np.sum(np.isnan(ts.values))} NaN values, "
                  f"{np.sum(np.isnan(ts.sigmas))} NaN sigmas)")

        ts.times = ts.times[nonnan_mask]
        ts.values = ts.values[nonnan_mask]
        if ts.sigmas.size > 0:
            ts.sigmas = ts.sigmas[nonnan_mask]

    @staticmethod
    def remove_outliers(timeseries, threshold=1.5, rolling_window=None, plot=True, save=True, verbose=True):
        """
        Removes outliers from the time series data based on the Interquartile Range (IQR).

        Detects outliers using the IQR defined either 1) globally or 2) within a rolling window
        around each data point. Flags outliers for plotting and removes them from the data if save=True.

        Parameters:
        - timeseries (TimeSeries): The time series object to clean.
        - threshold (float): Multiplier for the IQR to define outlier limits (default: 1.5).
        - rolling_window (int): If specified, applies a local IQR within the rolling window.
        - plot (bool): Whether to visualize the detected outliers.
        - save (bool): Whether to remove the outliers from the original data.
        - verbose (bool): Whether to print the number of outliers removed.
        """
        def plot_outliers(outlier_mask):
            """Plots the data flagged as outliers."""
            if sigmas is not None:
                plt.errorbar(
                    times[~outlier_mask], values[~outlier_mask],
                    yerr=sigmas[~outlier_mask], fmt='o', color='black', lw=1, ms=2
                )
                plt.errorbar(
                    times[outlier_mask], values[outlier_mask],
                    yerr=sigmas[outlier_mask], fmt='o', color='red', label='Outliers', lw=1, ms=2
                )
            else:
                plt.scatter(times[~outlier_mask], values[~outlier_mask], s=2)
                plt.scatter(
                    times[outlier_mask], values[outlier_mask],
                    color='red', label='Outliers', s=2
                )
            plt.xlabel('Time')
            plt.ylabel('Values')
            plt.title('Outlier Detection')
            plt.legend()
            plt.show()

        def detect_outliers(values, threshold, rolling_window):
            if rolling_window:
                outlier_mask = np.zeros_like(values, dtype=bool)
                half_window = rolling_window // 2
                for i in range(len(values)):
                    start = max(0, i - half_window)
                    end = min(len(values), i + half_window + 1)
                    local_data = values[start:end]
                    q1, q3 = np.percentile(local_data, [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    if values[i] < lower_bound or values[i] > upper_bound:
                        outlier_mask[i] = True
            else:
                q1, q3 = np.percentile(values, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                outlier_mask = (values < lower_bound) | (values > upper_bound)
            return outlier_mask

        ts = timeseries
        times = ts.times
        values = ts.values
        sigmas = ts.sigmas

        outlier_mask = detect_outliers(values, threshold=threshold, rolling_window=rolling_window)

        if verbose:
            print(f"Removed {np.sum(outlier_mask)} outliers "
                f"({np.sum(outlier_mask) / len(values) * 100:.2f}% of data).")

        if plot:
            plot_outliers(outlier_mask)

        # Save results back to the original timeseries if save=True
        if save:
            ts.times = times[~outlier_mask]
            ts.values =  values[~outlier_mask]
            if sigmas.size > 0:
                ts.sigmas = sigmas[~outlier_mask]

    @staticmethod
    def polynomial_detrend(timeseries, degree=1, plot=False, save=True):
        """
        Removes a polynomial trend from the data.

        Fits a polynomial of the specified degree to the data and subtracts it
        to produce a detrended time series. Optionally visualizes the original
        data, fitted trend, and detrended data.

        Parameters:
        - timeseries (TimeSeries): The time series object to detrend.
        - degree (int): Degree of the polynomial to fit (default: 1, linear).
        - plot (bool): Whether to plot the original, trend, and detrended data.
        - save (bool): Whether to modify the original TimeSeries object or return the detrended data.

        Returns:
        - detrended_values (array-like): The detrended values (if save=False).
        """
        ts = timeseries

        # Fit polynomial to the data
        if ts.sigmas.size > 0:
            coefficients = np.polyfit(ts.times, ts.values, degree, w=1/ts.sigmas)
        else:
            coefficients = np.polyfit(ts.times, ts.values, degree)
        polynomial = np.poly1d(coefficients)
        trend = polynomial(ts.times)

        detrended_values = ts.values - trend
        if plot:
            if ts.sigmas.size > 0:
                plt.errorbar(ts.times, ts.values, yerr=ts.sigmas, fmt='o', color='black', lw=1, ms=2, label='Original Data')
                plt.errorbar(ts.times, detrended_values, yerr=ts.sigmas, fmt='o', color='dodgerblue', lw=1, ms=2, label='Detrended Data')
            else:
                plt.plot(ts.times, ts.values, label="Original Data", color="black", alpha=0.6)
                plt.plot(ts.times, detrended_values, label="Detrended Data", color="dodgerblue")

            plt.plot(ts.times, trend, color='orange', linestyle='--', label=f'Fitted Polynomial (degree={degree})')

            plt.xlabel("Time")
            plt.ylabel("Values")
            plt.title("Polynomial Detrending")
            plt.legend()
            plt.show()

        if save:
            ts.values = detrended_values
