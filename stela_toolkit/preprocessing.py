import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox, shapiro, probplot


class Preprocessing:
    """
    Utility functions for cleaning and transforming light curves.

    The static methods in this class operate on LightCurve objects directly,
    modifying them in place unless otherwise specified.

    These methods are used throughout the STELA Toolkit to prepare light curves
    for Gaussian process modeling and spectral analysis. This includes:

    - Standardizing light curve data (zero mean, unit variance)
    - Applying and reversing a Box-Cox transformation to normalize flux distributions
    - Checking for Gaussianity using the Shapiro-Wilk test and Q-Q plots
    - Trimming light curves by time range
    - Removing outliers using global or local IQR
    - Polynomial detrending
    - Handling NaNs or missing data

    Most methods automatically store relevant metadata (e.g., original mean, std, Box-Cox lambda)
    on the LightCurve object for later reversal.

    All methods are static and do not require instantiating this class.
    """
    @staticmethod
    def standardize(lightcurve):
        """
        Standardize the light curve by subtracting its mean and dividing by its std.

        Saves the original mean and std as attributes for future unstandardization.
        """
        lc = lightcurve

        # check for standardization
        if np.isclose(lc.mean, 0, atol=1e-10) and np.isclose(lc.std, 1, atol=1e-10) or getattr(lc, "is_standard", False):
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
        Restore the light curve to its original units using stored mean and std.

        This reverses a previous call to `standardize`.
        """
        lc = lightcurve
        # check that data has been standardized
        if getattr(lc, "is_standard", False):
            lc.rates = (lc.rates * lc.unstandard_std) + lc.unstandard_mean
        else:
            if np.isclose(lc.mean, 0, atol=1e-10) and np.isclose(lc.std, 1, atol=1e-10):
                raise AttributeError(
                    "The data has not been standardized by STELA.\n"
                    "Please call the 'standardize' method first."
                )
            else:
                raise AttributeError(
                    "The data is not standardized, and needs to be standardized first by STELA.\n"
                    "Please call the 'standardize' method first (e.g., Preprocessing.standardize(lightcurve))."
                )
        
        if lc.errors.size > 0:
            lc.errors = lc.errors * lc.unstandard_std

        lc.is_standard = False  # reset the standardization flag
        
    @staticmethod
    def generate_qq_plot(lightcurve=None, rates=[]):
        """
        Generate a Q-Q plot to visually assess normality.

        Parameters
        ----------
        lightcurve : LightCurve, optional
            Light curve to extract rates from.
        rates : array-like, optional
            Direct rate values if not using a LightCurve.
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
        Test for normality using the Shapiro-Wilk test.

        Parameters
        ----------
        lightcurve : LightCurve, optional
            Light curve to extract rates from.
        rates : array-like, optional
            Direct rate values if not using a LightCurve.
        plot : bool
            Whether to show a Q-Q plot.
        _boxcox : bool
            Whether this check is being called internally after Box-Cox.
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
        Apply a Box-Cox transformation to normalize the flux distribution.

        Also adjusts errors using the delta method. Stores the transformation
        parameter lambda and sets a flag for reversal.

        Parameters
        ----------
        lightcurve : LightCurve
            The input light curve.
        save : bool
            Whether to modify the light curve in place.
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
        Reverse a previously applied Box-Cox transformation.

        Parameters
        ----------
        lightcurve : LightCurve
            The transformed light curve.
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
        Apply Box-Cox and re-test for normality using Shapiro-Wilk.

        Parameters
        ----------
        lightcurve : LightCurve
            The input light curve.
        """
        rates_boxcox, _ = Preprocessing.boxcox_transform(lightcurve, save=False)
        Preprocessing.check_normal(rates_boxcox, _boxcox=True)

    @staticmethod
    def trim_time_segment(lightcurve, start_time=None, end_time=None, plot=False, save=True):
        """
        Trim the light curve to a given time range.

        Parameters
        ----------
        start_time : float, optional
            Lower time bound.
        end_time : float, optional
            Upper time bound.
        plot : bool
            Whether to plot before/after trimming.
        save : bool
            Whether to modify the light curve in place.
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
        Remove time, rate, or error entries that are NaN.

        Parameters
        ----------
        lightcurve : LightCurve
            Light curve to clean.
        verbose : bool
            Whether to print how many NaNs were removed.
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
        Remove outliers using the IQR method, globally or locally.

        Parameters
        ----------
        lightcurve : LightCurve
            The input light curve.
        threshold : float
            IQR multiplier.
        rolling_window : int, optional
            Size of local window (if local filtering is desired).
        plot : bool
            Whether to visualize removed points.
        save : bool
            Whether to modify the light curve in place.
        verbose : bool
            Whether to print how many points were removed.
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
        Remove a polynomial trend from the light curve.

        Fits and subtracts a polynomial. Optionally modifies in place.

        Parameters
        ----------
        lightcurve : LightCurve
            The input light curve.
        degree : int
            Degree of the polynomial (default is 1).
        plot : bool
            Whether to show the trend removal visually.
        save : bool
            Whether to apply the change to the light curve.

        Returns
        -------
        detrended_rates : ndarray, optional
            Only returned if `save=False`.
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
