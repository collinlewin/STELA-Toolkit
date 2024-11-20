
import numpy as np
import matplotlib.pyplot as plt
from .data_loader import TimeSeries


class Preprocessing:
    @staticmethod
    def clean_data(timeseries, outlier_threshold=1.5, outlier_rolling_window=10, verbose=True):
        """Cleans the time series data by removing NaNs and outliers and standardizing."""
        ts = timeseries
        Preprocessing.remove_nans(ts, verbose=verbose)
        Preprocessing.remove_outliers(ts, threshold=outlier_threshold, rolling_window=outlier_rolling_window, verbose=verbose)
        Preprocessing.standardize(ts)

    @staticmethod
    def standardize(timeseries):
        """Standardizes the time series data."""
        ts = timeseries
        ts.unstandard_mean = np.mean(ts.values)
        ts.unstandard_std = np.std(ts.values)
        ts.values = (ts.values - ts.unstandard_mean) / ts.unstandard_std
        if ts.errors.size > 0:
            ts.errors /= ts.unstandard_std

    @staticmethod
    def unstandardize(timeseries):
        """Unstandardizes the time series data."""
        ts = timeseries
        try:
            ts.values = (ts.values * ts.unstandard_std) + ts.unstandard_mean
            if ts.errors.size > 0:
                ts.errors *= ts.unstandard_std
        except AttributeError:
            raise AttributeError("The data has not been standardized yet. Please call the 'standardize' method first.")

    @staticmethod
    def trim_time_segment(timeseries, start_time, end_time):
        """Trims the time series data to a specific time range."""
        ts = timeseries
        mask = (ts.times >= start_time) & (ts.times <= end_time)
        ts.times = ts.times[mask]
        ts.values = ts.values[mask]
        if ts.errors.size > 0:
            ts.errors = ts.errors[mask]

    @staticmethod
    def remove_nans(timeseries, verbose=False):
        """Removes NaN values where time, value, or measurement error is NaN."""
        ts = timeseries
        if ts.errors.size > 0:
            nonnan_mask = ~np.isnan(ts.values) & ~np.isnan(ts.times) & ~np.isnan(ts.errors)
        else:
            nonnan_mask = ~np.isnan(ts.values) & ~np.isnan(ts.times)

        if verbose:
            print(f"Removed {np.sum(~nonnan_mask)} NaN points.")

        ts.times = ts.times[nonnan_mask]
        ts.values = ts.values[nonnan_mask]
        if ts.errors.size > 0:
            ts.errors = ts.errors[nonnan_mask]

    @staticmethod
    def remove_outliers(timeseries, threshold=1.5, rolling_window=None, plot=True, verbose=False):
        """Identifies and removes outliers based on the Interquartile Range (IQR)."""
        ts = timeseries

        def detect_outliers(values, threshold=1.5, rolling_window=None):
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

        outlier_mask = detect_outliers(ts.values, threshold, rolling_window)
        if verbose:
            print(f"Removed {np.sum(outlier_mask)} outliers.")

        if plot:
            plt.scatter(ts.times[~outlier_mask], ts.values[~outlier_mask], s=2, label="Inliers")
            plt.scatter(ts.times[outlier_mask], ts.values[outlier_mask], s=2, color="red", label="Outliers")
            plt.xlabel("Time")
            plt.ylabel("Values")
            plt.title("Outlier Detection")
            plt.legend()
            plt.show()

        ts.times = ts.times[~outlier_mask]
        ts.values = ts.values[~outlier_mask]
        if ts.errors.size > 0:
            ts.errors = ts.errors[~outlier_mask]