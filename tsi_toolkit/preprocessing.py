
import numpy as np
import matplotlib.pyplot as plt


class Preprocessing:
    @staticmethod
    def standardize(timeseries):
        """Standardizes the time series data."""
        ts = timeseries
        ts.unstandard_mean = ts.mean
        ts.unstandard_std = ts.std
        ts.values = (ts.values - ts.unstandard_mean) / ts.unstandard_std
        if ts.errors.size > 0:
            ts.errors = ts.errors / ts.unstandard_std

    @staticmethod
    def unstandardize(timeseries):
        """Unstandardizes the time series data."""
        ts = timeseries
        try:
            ts.values = (ts.values * ts.unstandard_std) + ts.unstandard_mean
        except AttributeError:
            raise AttributeError(
                "The data has not been standardized yet. "
                "Please call the 'standardize' method first."
            )
        if ts.errors.size > 0:
            ts.errors = ts.errors * ts.unstandard_std

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
    def remove_nans(timeseries, verbose=True):
        """Removes NaN values where time, value, or measurement error is NaN."""
        ts = timeseries
        if ts.errors.size > 0:
            nonnan_mask = ~np.isnan(ts.values) & ~np.isnan(ts.times) & ~np.isnan(ts.errors)
        else:
            nonnan_mask = ~np.isnan(ts.values) & ~np.isnan(ts.times)

        if verbose:
            print(f"Removed {np.sum(~nonnan_mask)} NaN points.\n"
                  f"({np.sum(np.isnan(ts.values))} NaN values, "
                  f"{np.sum(np.isnan(ts.errors))} NaN errors)")

        ts.times = ts.times[nonnan_mask]
        ts.values = ts.values[nonnan_mask]
        if ts.errors.size > 0:
            ts.errors = ts.errors[nonnan_mask]

    @staticmethod
    def remove_outliers(timeseries, threshold=1.5, rolling_window=None, plot=True, verbose=True, save=True):
        """
        Identifies and removes outliers based on the Interquartile Range (IQR).
        Stores a mask of outliers for plotting.

        Parameters:
        - threshold (float): Multiplier for the IQR to define the outlier range.
        - rolling_window (int): If specified, applies a local IQR over a rolling
            window. Otherwise, uses the global IQR.
        """
        ts = timeseries
        # Create copies if save is False
        if not save:
            times = ts.times.copy()
            values = ts.values.copy()
            errors = ts.errors.copy()
        else:
            times = ts.times
            values = ts.values
            errors = ts.errors

        def plot_outliers(outlier_mask):
            """Plots the data flagged as outliers."""
            if errors is not None:
                plt.errorbar(
                    times[~outlier_mask], values[~outlier_mask],
                    yerr=errors[~outlier_mask], fmt='o', color='black', lw=1, ms=2
                )
                plt.errorbar(
                    times[outlier_mask], values[outlier_mask],
                    yerr=errors[outlier_mask], fmt='o', color='red', label='Outliers', lw=1, ms=2
                )
            else:
                plt.scatter(times[~outlier_mask], values[~outlier_mask], s=2)
                plt.scatter(
                    times[outlier_mask], values[outlier_mask],
                    color='red', label='Outliers', s=2
                )
            plt.xlabel('Time')
            plt.ylabel('Values')
            plt.title('Outliers Detection')
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

        # Detect outliers
        outlier_mask = detect_outliers(values, threshold=threshold, rolling_window=rolling_window)

        if verbose:
            print(f"Removed {np.sum(outlier_mask)} outliers "
                f"({np.sum(outlier_mask) / len(values) * 100:.2f}% of data).")

        if plot:
            plot_outliers(outlier_mask)

        # Apply mask
        cleaned_times = times[~outlier_mask]
        cleaned_values = values[~outlier_mask]
        cleaned_errors = errors[~outlier_mask] if errors is not None else None

        # Save results back to the original timeseries if save=True
        if save:
            ts.times = cleaned_times
            ts.values = cleaned_values
            if errors is not None:
                ts.errors = cleaned_errors