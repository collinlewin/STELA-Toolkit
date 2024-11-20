
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
    def trim_time_segment(timeseries, start_time=None, end_time=None, plot=False, save=True):
        """
        Trims the time series data to a specific time range.
        
        Parameters:
        - timeseries (TimeSeries): The time series object to trim.
        - start_time (float): The starting time for the range.
        - end_time (float): The ending time for the range.
        - plot (bool): Whether to plot the data before and after trimming.
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
            if ts.errors.size > 0:
                plt.errorbar(ts.times[mask], ts.values[mask], yerr=ts.errors[mask], fmt='o', lw=1, ms=2, color='black', label='Kept Data')
                plt.errorbar(ts.times[~mask], ts.values[~mask], yerr=ts.errors[~mask], fmt='o', lw=1, ms=2, color='red', label='Trimmed Data')
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
        errors = ts.errors

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
            if errors.size > 0:
                ts.errors = errors[~outlier_mask]

    @staticmethod
    def polynomial_detrend(timeseries, degree=1, plot=False, save=True):
        """
        Removes a polynomial trend from the time series data.

        Parameters:
        - timeseries (TimeSeries): The time series object to detrend.
        - degree (int): The degree of the polynomial to fit.
        - plot (bool): Whether to plot the original data, fitted trend, and detrended data.
        - save (bool): If True, modifies the original timeseries object. If False, returns the detrended data.

        Returns:
        If save=False:
            - detrended_values (array): Values after detrending.
        """
        ts = timeseries

        # Fit polynomial to the data
        if ts.errors.size > 0:
            coefficients = np.polyfit(ts.times, ts.values, degree, w=1/ts.errors)
        else:
            coefficients = np.polyfit(ts.times, ts.values, degree)
        polynomial = np.poly1d(coefficients)
        trend = polynomial(ts.times)

        detrended_values = ts.values - trend
        if plot:
            if ts.errors.size > 0:
                plt.errorbar(ts.times, ts.values, yerr=ts.errors, fmt='o', color='black', lw=1, ms=2, label='Original Data')
                plt.errorbar(ts.times, detrended_values, yerr=ts.errors, fmt='o', color='dodgerblue', lw=1, ms=2, label='Detrended Data')
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
