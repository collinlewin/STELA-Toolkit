import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox, shapiro, probplot


class Preprocessing:
    """
    Provides utility functions for preprocessing light curve data.

    This class includes methods for standardizing and unstandardizing data,
    trimming time segments, removing NaN rates and outliers, and detrending
    light curve using polynomial fitting. Designed to operate directly on
    LightCurve objects.
    """
    @staticmethod
    def standardize(lightcurve):
        """
        Standardizes the light curve data. 
        Stores the original mean and standard deviation for future unstandardization.
        """
        lc = lightcurve

        # check for standardization
        if np.isclose(lc.mean, 0, atol=1e-10) and np.isclose(lc.std, 1, atol=1e-10) or lc.is_standard:
            if not hasattr(lc, "unstandard_mean") and not hasattr(lc, "unstandard_std"):
                lc.unstandard_mean = 0
                lc.unstandard_std = 1
            print("The data is already standardized.")

        # apply standardization
        else:
            lc.unstandard_mean = lc.mean
            lc.unstandard_std = lc.std
            lc.rates = (lc.rates - lc.unstandard_mean) / lc.unstandard_std
            if lc.errors.size > 0:
                lc.errors = lc.errors / lc.unstandard_std
            
        lc.is_standard = True # flag for detecting transformation without computation

    @staticmethod
    def unstandardize(lightcurve):
        """
        Unstandardizes the light curve data.
        Restores the rates and errors of the input LightCurve object to their
        unstandardized form using the previously stored mean and standard deviation.
        """
        lc = lightcurve
        # check that data has been standardized
        if getattr(lc, "is_standard", False):
            lc.rates = (lc.rates * lc.unstandard_std) + lc.unstandard_mean
        else:
            if np.isclose(lc.mean, 0, atol=1e-10) and np.isclose(lc.std, 1, atol=1e-10):
                raise AttributeError(
                    "The data has not been standardized by STELA. "
                    "Please call the 'standardize' method first."
                )
            else:
                raise AttributeError(
                    "The data is not standardized, and needs to be standardized first by STELA."
                    "Please call the 'standardize' method first (e.g., Preprocessing.standardize(lightcurve))."
                )
        
        if lc.errors.size > 0:
            lc.errors = lc.errors * lc.unstandard_std

        lc.is_standard = False  # reset the standardization flag
        del lc.unstandard_mean  # clean up unnecessary attributes
        del lc.unstandard_std

    @staticmethod
    def generate_qq_plot(lightcurve=None, rates=[]):
        """
        Generates a Q-Q plot to visualize the normality of the input data.

        Parameters: ** ONE of the following must be provided
        lightcurve (object): Lightcurve object.
        rates (list or array-like): Direct input of rate values.
        """
        if lightcurve:
            rates = lightcurve.rates
        elif np.array(rates).size != 0:
            pass
        else: 
            raise ValueError("Either 'lightcurve' or 'rates' must be provided.")
        
        probplot(rates, dist="norm", plot=plt)
        plt.title("Q-Q Plot")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Sample Quantiles")
        plt.grid(True)
        plt.show()

    @staticmethod
    def check_normal(lightcurve=None, rates=[], plot=True, _boxcox=False):
        """
        Check if the given light curve data or rates are normally distributed
        using the Shapiro-Wilk test.
        
        Parameters: ** ONE of the following must be provided
        lightcurve (object): Lightcurve object.
        rates (list or array-like): Direct input of rate values.
        """        
        if lightcurve:
            rates = lightcurve.rates
        elif np.array(rates).size != 0:
            pass
        else:
            raise ValueError("Either 'lightcurve' or 'rates' must be provided.")
        
        # Run Shapiro-wilks test
        pvalue = shapiro(rates).pvalue
        print(f"p-value from Shapiro-Wilks test: {pvalue:.3f}")
        
        # Compare to alpha = 0.05
        if pvalue <= 0.05:  # reject
            print(f"  => can reject null hypothesis that the data is \
                  normally distributed, assuming a 0.05 significance level.")
            
            if not _boxcox:
                print("Use check_boxcox_normal to see if boxcox transformation \
                      can sufficiently help achieve normality.")
    
        elif pvalue > 0.05:  # fail to reject
            print(f"  => unable to reject null hypothesis that the data is \
                   normally distributed, assuming a 0.05 significance level.")
            
        if plot:
            Preprocessing.generate_qq_plot(rates=rates)
    
    @staticmethod
    def boxcox_transform(lightcurve, save=True):
        """
        Apply Box-Cox transformation and adjust uncertainties using the delta method.

        y = (x**lambda - 1) / lmbda,  for lmbda != 0
            log(x),                  for lmbda = 0

        lambda optimized to minimizes negative log-likelihood function (i.e., MLE).
        """
        lc = lightcurve

        # transform rates
        rates_boxcox, lambda_opt = boxcox(lc.rates)

        # transform errors using delta method (derivative-based propagation)
        if lc.errors.size != 0:
            if lambda_opt == 0:  # for log transformation (lambda = 0)
                errors_boxcox = lc.errors / lc.rates
            else:
                errors_boxcox = (lc.rates ** (lambda_opt - 1)) * lc.errors
        else:
            errors_boxcox = None

        if save:
            lc.rates = rates_boxcox
            lc.errors = errors_boxcox
            lc.lambda_boxcox = lambda_opt  # save lambda for inverse transformation
            lc.is_boxcox_transformed = True  # flag to indicate transformation
        else:
            return rates_boxcox, errors_boxcox
        
    @staticmethod
    def reverse_boxcox_transform(lightcurve):
        """
        Reverse the Box-Cox transformation and restore original rates and uncertainties.

        Parameters:
        lightcurve (object): Light curve object that has been transformed with Box-Cox.

        Raises:
        ValueError: If the light curve has not been transformed with Box-Cox.
        """
        lc = lightcurve

        # Check if the lightcurve has been transformed
        if not getattr(lc, "is_boxcox_transformed", False):
            raise ValueError("Light curve data has not been transformed with Box-Cox.")

        # Retrieve lambda used for transformation
        lambda_opt = lc.lambda_boxcox

        # Reverse rates transformation
        if lambda_opt == 0:  # inverse log transformation
            rates_original = np.exp(lc.rates)
        else:
            rates_original = (lc.rates * lambda_opt + 1) ** (1 / lambda_opt)

        # Reverse errors transformation
        if lc.errors.size != 0:
            if lambda_opt == 0:  # inverse log transformation (lambda = 0)
                errors_original = lc.errors * rates_original
            else:
                errors_original = lc.errors / (rates_original ** (lambda_opt - 1))
        else:
            errors_original = None

        # Restore the original values
        lc.rates = rates_original
        lc.errors = errors_original
        lc.is_boxcox_transformed = False  # Reset the transformation flag
        del lc.lambda_boxcox  # Clean up lambda attribute


    @staticmethod
    def check_boxcox_normal(lightcurve):
        """
        Checks if applying a Box-Cox transormation results in normally distributed data 
        based on a Shapiro-Wilks test.
        
        Parameters: ** ONE of the following must be provided
        lightcurve (object): Lightcurve object.
        rates (list or array-like): Direct input of rate values.
        """
        rates_boxcox, _ = Preprocessing.boxcox_transform(lightcurve, save=False)
        Preprocessing.check_normal(rates_boxcox, _boxcox=True)

    @staticmethod
    def trim_time_segment(lightcurve, start_time=None, end_time=None, plot=False, save=True):
        """
        Trims the light curve data to a specified time range.

        Filters the time, rate, and error arrays based on the provided start and
        end times. Optionally plots the data before and after trimming.

        Parameters:
        - lightcurve (LightCurve): The light curve object to trim.
        - start_time (float): The starting time for the range (default: first time point).
        - end_time (float): The ending time for the range (default: last time point).
        - plot (bool): Whether to plot the data before and after trimming.
        - save (bool): Whether to modify the original LightCurve object in place.
        """
        lc = lightcurve

        if start_time is None:
            start_time = lc.times[0]
        if end_time is None:
            end_time = lc.times[-1]
        if start_time and end_time is None:
            raise ValueError("Please specify a start and/or end time.")

        # Apply mask to trim data
        mask = (lc.times >= start_time) & (lc.times <= end_time)
        if plot:
            if lc.errors.size > 0:
                plt.errorbar(lc.times[mask], lc.rates[mask], yerr=lc.errors[mask],
                             fmt='o', lw=1, ms=2, color='black', label='Kept Data')
                plt.errorbar(lc.times[~mask], lc.rates[~mask], yerr=lc.errors[~mask],
                             fmt='o', lw=1, ms=2, color='red', label='Trimmed Data')
            else:
                plt.scatter(lc.times[mask], lc.rates[mask], s=2, color="black", label="Kept Data")
                plt.scatter(lc.times[~mask], lc.rates[~mask], s=2,
                            color="red", label="Trimmed Data")

            plt.xlabel("Time")
            plt.ylabel("Rates")
            plt.title("Trimming")
            plt.legend()
            plt.show()

        if save:
            lc.times = lc.times[mask]
            lc.rates = lc.rates[mask]
            if lc.errors.size > 0:
                lc.errors = lc.errors[mask]

    @staticmethod
    def remove_nans(lightcurve, verbose=True):
        """
        Removes rows with NaN values in time, rate, or error arrays.

        Parameters:
        - lightcurve (LightCurve): The light curve object to clean.
        - verbose (bool): Whether to print the number of NaN points removed.
        """
        lc = lightcurve
        if lc.errors.size > 0:
            nonnan_mask = ~np.isnan(lc.rates) & ~np.isnan(lc.times) & ~np.isnan(lc.errors)
        else:
            nonnan_mask = ~np.isnan(lc.rates) & ~np.isnan(lc.times)

        if verbose:
            print(f"Removed {np.sum(~nonnan_mask)} NaN points.\n"
                  f"({np.sum(np.isnan(lc.rates))} NaN rates, "
                  f"{np.sum(np.isnan(lc.errors))} NaN errors)")

        lc.times = lc.times[nonnan_mask]
        lc.rates = lc.rates[nonnan_mask]
        if lc.errors.size > 0:
            lc.errors = lc.errors[nonnan_mask]

    @staticmethod
    def remove_outliers(lightcurve, threshold=1.5, rolling_window=None, plot=True, save=True, verbose=True):
        """
        Removes outliers from the light curve data based on the Interquartile Range (IQR).

        Detects outliers using the IQR defined either 1) globally or 2) within a rolling window
        around each data point. Flags outliers for plotting and removes them from the data if save=True.

        Parameters:
        - lightcurve (LightCurve): The light curve object to clean.
        - threshold (float): Multiplier for the IQR to define outlier limits (default: 1.5).
        - rolling_window (int): If specified, applies a local IQR within the rolling window.
        - plot (bool): Whether to visualize the detected outliers.
        - save (bool): Whether to remove the outliers from the original data.
        - verbose (bool): Whether to print the number of outliers removed.
        """
        def plot_outliers(outlier_mask):
            """Plots the data flagged as outliers."""
            if errors is not None:
                plt.errorbar(
                    times[~outlier_mask], rates[~outlier_mask],
                    yerr=errors[~outlier_mask], fmt='o', color='black', lw=1, ms=2
                )
                plt.errorbar(
                    times[outlier_mask], rates[outlier_mask],
                    yerr=errors[outlier_mask], fmt='o', color='red', label='Outliers', lw=1, ms=2
                )
            else:
                plt.scatter(times[~outlier_mask], rates[~outlier_mask], s=2)
                plt.scatter(
                    times[outlier_mask], rates[outlier_mask],
                    color='red', label='Outliers', s=2
                )
            plt.xlabel('Time')
            plt.ylabel('Rates')
            plt.title('Outlier Detection')
            plt.legend()
            plt.show()

        def detect_outliers(rates, threshold, rolling_window):
            if rolling_window:
                outlier_mask = np.zeros_like(rates, dtype=bool)
                half_window = rolling_window // 2
                for i in range(len(rates)):
                    start = max(0, i - half_window)
                    end = min(len(rates), i + half_window + 1)
                    local_data = rates[start:end]
                    q1, q3 = np.percentile(local_data, [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    if rates[i] < lower_bound or rates[i] > upper_bound:
                        outlier_mask[i] = True
            else:
                q1, q3 = np.percentile(rates, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                outlier_mask = (rates < lower_bound) | (rates > upper_bound)
            return outlier_mask

        lc = lightcurve
        times = lc.times
        rates = lc.rates
        errors = lc.errors

        outlier_mask = detect_outliers(rates, threshold=threshold, rolling_window=rolling_window)

        if verbose:
            print(f"Removed {np.sum(outlier_mask)} outliers "
                  f"({np.sum(outlier_mask) / len(rates) * 100:.2f}% of data).")

        if plot:
            plot_outliers(outlier_mask)

        # Save results back to the original lightcurve if save=True
        if save:
            lc.times = times[~outlier_mask]
            lc.rates = rates[~outlier_mask]
            if errors.size > 0:
                lc.errors = errors[~outlier_mask]

    @staticmethod
    def polynomial_detrend(lightcurve, degree=1, plot=False, save=True):
        """
        Removes a polynomial trend from the data.

        Fits a polynomial of the specified degree to the data and subtracts it
        to produce a detrended light curve. Optionally visualizes the original
        data, fitted trend, and detrended data.

        Parameters:
        - lightcurve (LightCurve): The light curve object to detrend.
        - degree (int): Degree of the polynomial to fit (default: 1, linear).
        - plot (bool): Whether to plot the original, trend, and detrended data.
        - save (bool): Whether to modify the original LightCurve object or return the detrended data.

        Returns:
        - detrended_rates (array-like): The detrended rates (if save=False).
        """
        lc = lightcurve

        # Fit polynomial to the data
        if lc.errors.size > 0:
            coefficients = np.polyfit(lc.times, lc.rates, degree, w=1/lc.errors)
        else:
            coefficients = np.polyfit(lc.times, lc.rates, degree)
        polynomial = np.poly1d(coefficients)
        trend = polynomial(lc.times)

        detrended_rates = lc.rates - trend
        if plot:
            if lc.errors.size > 0:
                plt.errorbar(lc.times, lc.rates, yerr=lc.errors, fmt='o',
                             color='black', lw=1, ms=2, label='Original Data')
                plt.errorbar(lc.times, detrended_rates, yerr=lc.errors, fmt='o',
                             color='dodgerblue', lw=1, ms=2, label='Detrended Data')
            else:
                plt.plot(lc.times, lc.rates, label="Original Data", color="black", alpha=0.6)
                plt.plot(lc.times, detrended_rates, label="Detrended Data", color="dodgerblue")

            plt.plot(lc.times, trend, color='orange', linestyle='--',
                     label=f'Fitted Polynomial (degree={degree})')

            plt.xlabel("Time")
            plt.ylabel("Rates")
            plt.title("Polynomial Detrending")
            plt.legend()
            plt.show()

        if save:
            lc.rates = detrended_rates
