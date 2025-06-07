import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox, shapiro, probplot
from statsmodels.stats.diagnostic import lilliefors
from copy import deepcopy


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
            rates = lightcurve.rates.copy()
        elif np.array(rates).size != 0:
            pass
        else: 
            raise ValueError("Either 'lightcurve' or 'rates' must be provided.")
        
        rates_std = (rates - np.mean(rates)) / np.std(rates)
        (osm, osr), _ = probplot(rates_std, dist="norm")

        plt.figure(figsize=(8, 4.5))
        plt.plot(osm, osr, 'o', color='black', markersize=4)
        plt.plot(osm, osm, 'g--', lw=1, label='Ideal Normal')

        plt.title("Q-Q Plot", fontsize=12)
        plt.xlabel("Theoretical Quantiles", fontsize=12)
        plt.ylabel("Sample Quantiles", fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tick_params(which='both', direction='in', length=6, width=1, top=True, right=True, labelsize=12)
        plt.show()

    @staticmethod
    def check_normal(lightcurve=None, rates=[], plot=True, _boxcox=False, verbose=True):
        """
        Test for normality using an appropriate statistical test based on sample size.

        For small samples (n < 50), this uses the Shapiro-Wilk test. For larger samples,
        it uses the Lilliefors version of the Kolmogorov-Smirnov test. Results are printed
        with an interpretation of the strength of evidence against normality.

        If `plot=True`, a Q-Q plot of the distribution is shown. This function supports either
        a full LightCurve object or a raw array of flux values.

        Parameters
        ----------
        lightcurve : LightCurve, optional
            The light curve object containing the rates to test.
        
        rates : array-like, optional
            Direct rate values if not using a LightCurve.
        
        plot : bool, optional
            Whether to display a Q-Q plot.
        
        _boxcox : bool, optional
            Whether this check is being called internally after Box-Cox (affects messaging only).
        
        verbose : bool, optional
            Whether to generate print statements.

        Returns
        -------
        is_normal : bool
            True if the data appears normally distributed (p > 0.05).
        
        pvalue : float
            The p-value from the chosen normality test.
        """

        if lightcurve:
            rates = lightcurve.rates.copy()
        elif np.array(rates).size != 0:
            rates = np.array(rates)
        else:
            raise ValueError("Either 'lightcurve' or 'rates' must be provided.")

        n = len(rates)
        if n < 50:
            if verbose:
                print("Using Shapiro-Wilk test (recommended for n < 50)")
            test_name = "Shapiro-Wilk"
            pvalue = shapiro(rates).pvalue
        else:
            if verbose:
                print("Using Lilliefors test (for n >= 50)")
            test_name = "Lilliefors (modified KS)"
            _, pvalue = lilliefors(rates, dist='norm')

        if verbose:
            print(f"{test_name} test p-value: {pvalue:.3g}")
            if pvalue <= 0.001:
                strength = "very strong"
            elif pvalue <= 0.01:
                strength = "strong"
            elif pvalue <= 0.05:
                strength = "weak"
            else:
                strength = "little to no"

            print(f"  -> {strength.capitalize()} evidence against normality (p = {pvalue:.3g})")

            if pvalue <= 0.05 and not _boxcox:
                print("     - Consider running `check_boxcox_normal()` to see if a Box-Cox transformation can help.")
                print("     - Often checking normality via a Q-Q plot (run `generate_qq_plot(lightcurve)`) is sufficient.")
            print("===================")

        if plot:
            Preprocessing.generate_qq_plot(rates=rates)
        
        return pvalue > 0.05, pvalue


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
        rates_boxcox, lambda_opt = boxcox(lc.rates)

        # transform errors using delta method (derivative-based propagation)
        if lc.errors.size != 0:
            if lambda_opt == 0:  # log transformation (lambda = 0)
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

        if not getattr(lc, "is_boxcox_transformed", False):
            raise ValueError("Light curve data has not been transformed with Box-Cox.")

        lambda_opt = lc.lambda_boxcox
        if lambda_opt == 0:  # inverse log transformation
            rates_original = np.exp(lc.rates)
        else:
            rates_original = (lc.rates * lambda_opt + 1) ** (1 / lambda_opt)

        if lc.errors.size != 0:
            if lambda_opt == 0:  # inverse log transformation (lambda = 0)
                errors_original = lc.errors * rates_original
            else:
                errors_original = lc.errors / (rates_original ** (lambda_opt - 1))
        else:
            errors_original = None

        lc.rates = rates_original
        lc.errors = errors_original
        lc.is_boxcox_transformed = False
        del lc.lambda_boxcox

    @staticmethod
    def check_boxcox_normal(lightcurve, plot=True):
        """
        Apply a Box-Cox transformation and re-test for normality using the appropriate statistical test.

        This method compares the normality of the original flux distribution to its Box-Cox transformed version,
        using either the Shapiro-Wilk or Lilliefors test depending on sample size. If `plot=True`, a Q-Q plot
        is generated showing both the original and transformed data.

        Parameters
        ----------
        lightcurve : LightCurve
            The input light curve containing flux values.
        
        plot : bool, optional
            Whether to show a Q-Q plot comparing original and Box-Cox transformed distributions.

        Returns
        -------
        is_normal : bool
            True if the Box-Cox transformed data appears normally distributed (p > 0.05).
        
        pvalue : float
            The p-value from the normality test applied to the transformed data.
        """

        rates_original = lightcurve.rates.copy()
        rates_boxcox, _ = Preprocessing.boxcox_transform(lightcurve, save=False)

        print("Before Box-Cox:")
        print("----------------")
        Preprocessing.check_normal(lightcurve=lightcurve, plot=False)

        print("After Box-Cox:")
        print("----------------")
        is_normal, pvalue = Preprocessing.check_normal(rates=rates_boxcox, plot=False, _boxcox=True)
        
        if plot:
            rates_original_std = (rates_original - np.mean(rates_original)) / np.std(rates_original)
            rates_boxcox_std = (rates_boxcox - np.mean(rates_boxcox)) / np.std(rates_boxcox)

            (osm1, osr1), _ = probplot(rates_original_std, dist="norm")
            (osm2, osr2), _ = probplot(rates_boxcox_std, dist="norm")

            plt.figure(figsize=(8, 4.5))
            plt.plot(osm1, osr1, 'o', label='Original', color='black', alpha=0.6, markersize=4)
            plt.plot(osm2, osr2, 'o', label='Transformed', color='dodgerblue', alpha=0.6, markersize=4)
            plt.plot(osm1, osm1, 'g--', label='Ideal Normal', alpha=0.5, lw=1.5)

            plt.xlabel("Theoretical Quantiles", fontsize=12)
            plt.ylabel("Sample Quantiles", fontsize=12)
            plt.title("Q-Q Plot Before and After Box-Cox", fontsize=12)
            plt.legend(loc='upper left')
            plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            plt.tick_params(which='both', direction='in', length=6, width=1, top=True, right=True, labelsize=12)
            plt.show()
        
        return is_normal, pvalue


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
            plt.figure(figsize=(8, 4.5))
            if lc.errors is not None and len(lc.errors) > 0:
                plt.errorbar(lc.times[mask], lc.rates[mask], yerr=lc.errors[mask], 
                             fmt='o', color='black', ms=3, label='Kept')
                plt.errorbar(lc.times[~mask], lc.rates[~mask], yerr=lc.errors[~mask], 
                             fmt='o', color='orange', ms=3, label='Trimmed')
            else:
                plt.scatter(lc.times[mask], lc.rates[mask], s=6, color="black", label="Kept")
                plt.scatter(lc.times[~mask], lc.rates[~mask], s=6, color="red", label="Trimmed")
            plt.xlabel("Time", fontsize=12)
            plt.ylabel("Rates", fontsize=12)
            plt.title("Trimming")
            plt.legend()
            plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            plt.tick_params(which='both', direction='in', length=6, width=1, top=True, right=True, labelsize=12)
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
            print("===================")
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
            plt.figure(figsize=(8, 4.5))
            if errors is not None:
                plt.errorbar(times[~outlier_mask], rates[~outlier_mask], yerr=errors[~outlier_mask], 
                             fmt='o', color='black', ms=3, label='Kept')
                plt.errorbar(times[outlier_mask], rates[outlier_mask], yerr=errors[outlier_mask], 
                             fmt='o', color='orange', ms=3, label='Outliers')
            else:
                plt.scatter(times[~outlier_mask], rates[~outlier_mask], 
                            s=6, color='black', label='Kept')
                plt.scatter(times[outlier_mask], rates[outlier_mask], 
                            s=6, color='orange', label='Outliers')
            plt.xlabel("Time", fontsize=12)
            plt.ylabel("Rates", fontsize=12)
            plt.title("Outlier Detection")
            plt.legend()
            plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            plt.tick_params(which='both', direction='in', length=6, width=1, top=True, right=True, labelsize=12)
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

        lc = deepcopy(lightcurve)
        times = lc.times
        rates = lc.rates
        errors = lc.errors

        outlier_mask = detect_outliers(rates, threshold=threshold, rolling_window=rolling_window)

        if verbose:
            print(f"Removed {np.sum(outlier_mask)} outliers "
                  f"({np.sum(outlier_mask) / len(rates) * 100:.2f}% of data).")
            print("===================")

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

        lc = deepcopy(lightcurve)

        # Fit polynomial to the data
        if lc.errors.size > 0:
            coefficients = np.polyfit(lc.times, lc.rates, degree, w=1/lc.errors)
        else:
            coefficients = np.polyfit(lc.times, lc.rates, degree)
        polynomial = np.poly1d(coefficients)
        trend = polynomial(lc.times)

        detrended_rates = lc.rates - trend
        if plot:
            plt.figure(figsize=(8, 4.5))
            if lc.errors is not None and len(lc.errors) > 0:
                plt.errorbar(lc.times, lc.rates, yerr=lc.errors, 
                             fmt='o', color='black', label="Original", ms=3, lw=1.5, alpha=0.6)
                plt.errorbar(lc.times, detrended_rates, yerr=lc.errors, 
                             fmt='o', color='dodgerblue', label="Detrended", ms=3, lw=1.5)
            else:
                plt.plot(lc.times, lc.rates, label="Original", color="black", alpha=0.6, ms=3, lw=1.5)
                plt.plot(lc.times, detrended_rates, label="Detrended", color="dodgerblue", ms=3, lw=1.5)
            plt.plot(lc.times, trend, color='orange', linestyle='--', label='Fitted Trend')
            plt.xlabel("Time", fontsize=12)
            plt.ylabel("Rates", fontsize=12)
            plt.title("Polynomial Detrending")
            plt.legend()
            plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            plt.tick_params(which='both', direction='in', length=6, width=1, top=True, right=True, labelsize=12)
            plt.show()

        if save:
            lc.rates = detrended_rates
