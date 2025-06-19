import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from ._check_inputs import _CheckInputs


class CrossCorrelation:
    """
    Compute the time-domain cross-correlation function (CCF) between two light curves or GP models.

    This class supports three primary use cases:

    1. **Regularly sampled `LightCurve` objects (`mode="regular"`)**  
       Computes the CCF directly using array-based shifting of flux values.
       This mode requires both light curves to be sampled on the same time grid.

    2. **Irregularly sampled `LightCurve` objects (`mode="lin_interp"`)**  
       Uses the interpolated cross-correlation function (ICCF; Gaskell & Peterson 1987),
       which linearly interpolates each light curve onto the other's time grid
       to evaluate the correlation across lags. This supports fully independent time arrays.

    3. **`GaussianProcess` models**  
       If both inputs are trained GP models with sampled realizations (via `.sample()`),
       the CCF is computed for each realization pair and averaged.
       Outputs in this case include the mean and standard deviation of the peak lag,
       centroid lag, and maximum correlation (`rmax`).
       This mode currently supports only `mode="regular"`.

    Optionally, Monte Carlo resampling can be used to estimate uncertainties on lag measurements.
    This uses:
        - Random Subset Selection (RSS) with replacement
        - Flux Randomization (FR) via Gaussian noise
    and is only available when using `mode="lin_interp"`.

    Parameters
    ----------
    lc_or_model1 : LightCurve or GaussianProcess
        First input light curve or trained GP model.
    
    lc_or_model2 : LightCurve or GaussianProcess
        Second input light curve or trained GP model.
        
    mode : {"regular", "lin_interp"}, optional
        CCF computation mode. Use "regular" for array-based shifting, or "lin_interp" for ICCF-based interpolation.
    
    monte_carlo : bool, optional
        Whether to estimate lag uncertainties using Monte Carlo resampling (RSS + FR). Only supported with lin_interp mode.
    
    n_trials : int, optional
        Number of Monte Carlo trials (default: 1000).
    
    min_lag : float or "auto", optional
        Minimum lag to evaluate. If "auto", set to `-duration / 2`.
    
    max_lag : float or "auto", optional
        Maximum lag to evaluate. If "auto", set to `+duration / 2`.
    
    dt : float or "auto", optional
        Lag grid spacing. If "auto", uses half the native time resolution in "regular" mode,
        and 1/3 the mean sampling interval in "lin_interp" mode.
    
    centroid_threshold : float, optional
        Threshold (as a fraction of peak correlation) for defining the centroid lag region.

    rmax_threshold : float, optional
        Trials with a maximum correlation (rmax) below this threshold are discarded when using Monte Carlo.

    Attributes
    ----------
    lags : ndarray
        Array of lag values evaluated.
    
    ccf : ndarray or None
        Cross-correlation coefficients. Not set when both inputs are GP models.
    
    peak_lag : float or tuple
        Peak lag of the CCF. If using GPs, returns (mean, std) across realizations.
    
    centroid_lag : float or tuple
        Centroid lag of the high-correlation region. If using GPs, returns (mean, std).
    
    rmax : float or tuple
        Maximum correlation value. If using GPs, returns (mean, std).
    
    peak_lags_mc : ndarray or None
        Peak lags from Monte Carlo trials, if enabled.
    
    centroid_lags_mc : ndarray or None
        Centroid lags from Monte Carlo trials.
    
    peak_lag_ci : tuple or None
        68% confidence interval (16th–84th percentile) on peak lag from MC trials.
    
    centroid_lag_ci : tuple or None
        68% confidence interval on centroid lag from MC trials.
    """


    def __init__(self,
                 lc_or_model1,
                 lc_or_model2,
                 mode="regular",
                 monte_carlo=False,
                 n_trials=1000,
                 min_lag="auto",
                 max_lag="auto",
                 dt="auto",
                 centroid_threshold=0.8,
                 rmax_threshold=0.0):

        req_reg_samp = True if mode=="regular" else False
        data1 = _CheckInputs._check_lightcurve_or_model(lc_or_model1, req_reg_samp=req_reg_samp)
        data2 = _CheckInputs._check_lightcurve_or_model(lc_or_model2, req_reg_samp=req_reg_samp)

        self.is_model1 = data1['type'] == 'model'
        self.is_model2 = data2['type'] == 'model'
        self.mode = mode
        self.monte_carlo = monte_carlo

        if self.is_model1:
            if not hasattr(lc_or_model1, 'samples'):
                raise ValueError("Model 1 must have generated samples via GP.sample().")
            self.times = lc_or_model1.pred_times
            self.rates1 = lc_or_model1.samples
        else:
            self.times, self.rates1, self.errors1 = data1['data']

        if self.is_model2:
            if not hasattr(lc_or_model2, 'samples'):
                raise ValueError("Model 2 must have generated samples via GP.sample().")
            self.times = lc_or_model2.pred_times
            self.rates2 = lc_or_model2.samples
        else:
            self.times, self.rates2, self.errors2 = data2['data']

        self.n_trials = n_trials
        self.centroid_threshold = centroid_threshold
        self.rmax_threshold = rmax_threshold

        duration = self.times[-1] - self.times[0]
        self.min_lag = -duration / 2 if min_lag=="auto" else min_lag
        self.max_lag = duration / 2 if max_lag=="auto" else max_lag

        if mode == "regular":
            times1 = data1['data'][0]
            times2 = data2['data'][0]

            # require the same time grid for regular
            if not np.allclose(times1, times2, rtol=1e-10, atol=1e-12):
                raise ValueError(
                    "In 'regular' mode, both light curves must have the same time array.\n"
                    "Use 'lin_interp' mode for irregular or mismatched time sampling."
                )
            
            self.dt = np.diff(self.times)[0] / 2 if dt=="auto" else dt
            self.lags = np.arange(self.min_lag, self.max_lag + self.dt, self.dt)

            if self.is_model1 and self.is_model2:
                ccfs, self.peak_lags, self.centroid_lags, self.rmaxs = [], [], [], []

                # Compute ccf and lags for each pair of realizations
                for i in range(self.rates1.shape[0]):
                    ccf = self.compute_ccf(self.rates1[i], self.rates2[i])
                    peak_lag, centroid_lag = self.find_peak_and_centroid(self.lags, ccf)
                    rmax = np.max(ccf)

                    ccfs.append(ccf)
                    self.peak_lags.append(peak_lag)
                    self.centroid_lags.append(centroid_lag)
                    self.rmaxs.append(rmax)

                # Consider use of CIs
                self.ccf = np.mean(ccfs, axis=0)
                self.peak_lag = (np.mean(self.peak_lags), np.std(self.peak_lags))
                self.centroid_lag = (np.mean(self.centroid_lags), np.std(self.centroid_lags))
                self.rmax = (np.mean(self.rmaxs), np.std(self.rmaxs))
                
        elif mode == "lin_interp":
            if self.is_model1 or self.is_model2:
                raise NotImplementedError("GP models are not yet supported with lin_interp mode.")

            self.times1 = data1['data'][0]
            self.times2 = data2['data'][0]

            self.dt = np.mean([np.mean(np.diff(self.times1)), np.mean(np.diff(self.times2))]) / 3 if dt == "auto" else dt
            self.lags = np.arange(self.min_lag, self.max_lag + self.dt, self.dt)

            self.ccf = self.compute_ccf_interp(self.times1, self.rates1, self.times2, self.rates2)

            self.rmax = np.max(self.ccf)
            self.peak_lag, self.centroid_lag = self.find_peak_and_centroid(self.lags, self.ccf)

        else:
            raise AttributeError(f"Invalid mode detected: {mode}. Valid modes are 'regular' and 'lin_interp'.")
        
        if monte_carlo:
            if np.all(self.errors1 == 0) or np.all(self.errors2 == 0):
                print("Skipping Monte Carlo: zero errors for all points in one or both light curves.")
            else:
                self.peak_lags_mc, self.centroid_lags_mc = self.run_monte_carlo()
                self.compute_confidence_intervals()

    def compute_ccf(self, rates1, rates2):
        """
        Compute the cross-correlation function (CCF) via direct shifting.

        Parameters
        ----------
        rates1 : ndarray
            First time series.
       
        rates2 : ndarray
            Second time series.

        Returns
        -------
        lags : ndarray
            Lag values.
        
        ccf : ndarray
            Pearson correlation coefficients at each lag.
        """

        ccf = []

        for lag in self.lags:
            shift = int(round(lag / self.dt))

            if shift < 0:
                x = rates1[:shift]
                y = rates2[-shift:]
            elif shift > 0:
                x = rates1[shift:]
                y = rates2[:-shift]
            else:
                x = rates1
                y = rates2

            if len(x) < 2:
                ccf.append(0.0)
            else:
                r = np.corrcoef(x, y)[0, 1]
                ccf.append(r)

        return np.array(ccf)
    

    def compute_ccf_interp(self, times1, rates1, times2, rates2):
        """
        Compute the cross-correlation function using linear interpolation on both light curves.

        Parameters
        ----------
        times1 : ndarray
            Time values for the first light curve.
        rates1 : ndarray
            Flux values for the first light curve.
        times2 : ndarray
            Time values for the second light curve.
        rates2 : ndarray
            Flux values for the second light curve.

        Returns
        -------
        ccf : ndarray
            Cross-correlation values at each lag.
        """
        interp1 = interp1d(times1, rates1, bounds_error=False, fill_value=0.0)
        interp2 = interp1d(times2, rates2, bounds_error=False, fill_value=0.0)
        ccf = []

        for lag in self.lags:
            t_shift1 = times1 + lag
            t_shift2 = times2 - lag

            mask1 = (t_shift1 >= times2[0]) & (t_shift1 <= times2[-1])
            mask2 = (t_shift2 >= times1[0]) & (t_shift2 <= times1[-1])

            if np.sum(mask1) < 2 or np.sum(mask2) < 2:
                ccf.append(0.0)
                continue

            r1 = np.corrcoef(rates1[mask1], interp2(t_shift1[mask1]))[0, 1]
            r2 = np.corrcoef(rates2[mask2], interp1(t_shift2[mask2]))[0, 1]
            ccf.append((r1 + r2) / 2)

        return np.array(ccf)


    def find_peak_and_centroid(self, lags, ccf):
        """
        Compute the peak and centroid lag of a cross-correlation function.

        The peak lag corresponds to the lag with the maximum correlation value.
        The centroid lag is computed using a weighted average of lag values
        in a contiguous region around the peak where the correlation exceeds
        a fraction of the peak value.

        Parameters
        ----------
        lags : ndarray
            Array of lag values (assumed sorted).
        
        ccf : ndarray
            Cross-correlation values at each lag.

        Returns
        -------
        peak_lag : float
            Lag corresponding to the maximum correlation.
        
        centroid_lag : float or np.nan
            Correlation-weighted centroid lag near the peak.
            Returns NaN if a valid centroid region cannot be identified.
        """
        if len(lags) != len(ccf) or len(ccf) == 0:
            raise ValueError("lags and ccf must be the same nonzero length")

        # Locate the peak correlation and corresponding lag
        peak_idx = np.nanargmax(ccf)
        peak_lag = lags[peak_idx]
        peak_val = ccf[peak_idx]

        # Define a local region around the peak above a fractional threshold
        threshold = self.centroid_threshold
        cutoff = threshold * peak_val

        # Expand to left of peak
        i_left = peak_idx
        while i_left > 0 and ccf[i_left - 1] >= cutoff:
            i_left -= 1

        # Expand to right of peak
        i_right = peak_idx
        while i_right < len(ccf) - 1 and ccf[i_right + 1] >= cutoff:
            i_right += 1

        # Compute centroid if region is valid
        if i_right >= i_left:
            lags_subset = lags[i_left:i_right + 1]
            ccf_subset = ccf[i_left:i_right + 1]
            weight_sum = np.sum(ccf_subset)
            if weight_sum > 0:
                centroid_lag = np.sum(lags_subset * ccf_subset) / weight_sum
            else:
                centroid_lag = np.nan
        else:
            centroid_lag = np.nan

        return peak_lag, centroid_lag
    

    def run_monte_carlo(self):
        """
        Run Monte Carlo simulations using RSS (only if mode='lin_interp') + FR to estimate the lag uncertainties.

        Each trial performs:

        - Random Subset Selection (RSS): Resample with replacement from each light curve
            - RSS can only be used when using linear interpolation, for which the time arrays do not need to be the same.
        - Flux Randomization (FR): Add Gaussian noise based on measurement errors
        - Discard trials with low correlation (if rmax_threshold is set)

        Returns
        -------
        peak_lags : ndarray
            Peak lag values from successful trials.

        centroid_lags : ndarray
            Centroid lag values from successful trials.
        """
        peak_lags = []
        centroid_lags = []

        n_points = len(self.times)

        for _ in range(self.n_trials):
            # Random subset selection
            if self.mode == 'lin_interp':
                idx1 = np.random.randint(0, n_points, n_points)
                unique1, counts1 = np.unique(idx1, return_counts=True)
                times1 = self.times[unique1]
                rates1 = self.rates1[unique1]
                errors1 = self.errors1[unique1] / np.sqrt(counts1) # reweighting errors based on frequency similar to bootstrap

                idx2 = np.random.randint(0, n_points, n_points)
                unique2, counts2 = np.unique(idx2, return_counts=True)
                times2 = self.times[unique2]
                rates2 = self.rates2[unique2]
                errors2 = self.errors2[unique2] / np.sqrt(counts2)

                sort1 = np.argsort(times1)
                sort2 = np.argsort(times2)
                times1, rates1, errors1 = times1[sort1], rates1[sort1], errors1[sort1]
                times2, rates2, errors2 = times2[sort2], rates2[sort2], errors2[sort2]
            else:
                rates1, errors1 = self.rates1, self.errors1
                rates2, errors2 = self.rates2, self.errors2

            # Flux randomization
            perturbed1 = np.random.normal(rates1, errors1)
            perturbed2 = np.random.normal(rates2, errors2)

            # Compute ccf
            if self.mode == "lin_interp":
                ccf = self.compute_ccf_interp(times1, perturbed1, times2, perturbed2)
            else:
                ccf = self.compute_ccf(perturbed1, perturbed2)

            # Rmax threshold filter
            peak, centroid = self.find_peak_and_centroid(self.lags, ccf)

            if np.max(ccf) < self.rmax_threshold or np.isnan(peak) or np.isnan(centroid):
                continue

            peak_lags.append(peak)
            centroid_lags.append(centroid)

        return np.array(peak_lags), np.array(centroid_lags)

    def compute_confidence_intervals(self, lower_percentile=16, upper_percentile=84):
        """
        Compute percentile-based confidence intervals for Monte Carlo lag distributions.

        Parameters
        ----------
        lower_percentile : float
            Lower percentile bound (default is 16).
        
        upper_percentile : float
            Upper percentile bound (default is 84).
        """

        if self.peak_lags_mc is None or self.centroid_lags_mc is None:
            print("No Monte Carlo results available to compute confidence intervals.")
            return

        self.peak_lag_ci = (
            np.percentile(self.peak_lags_mc, lower_percentile),
            np.percentile(self.peak_lags_mc, upper_percentile),
        )
        self.centroid_lag_ci = (
            np.percentile(self.centroid_lags_mc, lower_percentile),
            np.percentile(self.centroid_lags_mc, upper_percentile),
        )

    def plot(self, show_mc=True):
        """
        Plot the cross-correlation function and optional lag distributions.

        Parameters
        ----------
        show_mc : bool
            Whether to show lag distributions from GP samples or Monte Carlo trials.
        """
        # Plot the CCF
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.lags, self.ccf, label="CCF", color='black')

        if self.is_model1 and self.is_model2:
            peak_lag = self.peak_lag[0]
            peak_std = self.peak_lag[1]
            centroid_lag = self.centroid_lag[0]
            centroid_std = self.centroid_lag[1]
            label_suffix = " (GP samples)"
        else:
            peak_lag = self.peak_lag
            centroid_lag = self.centroid_lag
            peak_std = centroid_std = None
            label_suffix = ""

        ax.axvline(peak_lag, color='orange', linestyle='--',
                label=f"Peak lag = {peak_lag:.2f}")
        ax.axvline(centroid_lag, color='blue', linestyle=':',
                label=f"Centroid lag = {centroid_lag:.2f}")

        # Overlay lag distributions on same plot
        if show_mc:
            # Need to modify this to use the confidence intervals from earlier.
            # No else statement to ensure code execution for default show_mc even for no mc
            peak_data, centroid_data = None, None
            if self.monte_carlo:
                peak_data = self.peak_lags_mc
                centroid_data = self.centroid_lags_mc
                label_suffix = " (MC)"
                peak_std = np.std(peak_data)
                centroid_std = np.std(centroid_data)

            elif self.is_model1 and self.is_model2:
                peak_data = self.peak_lags
                centroid_data = self.centroid_lags

            if peak_data is not None:
                ax.hist(peak_data, bins=30, density=True, color='orange', alpha=0.3,
                        label=f"Peak lag dist{label_suffix}, σ={peak_std:.2f}", zorder=1)
            if centroid_data is not None:
                ax.hist(centroid_data, bins=30, density=True, color='blue', alpha=0.3,
                        label=f"Centroid lag dist{label_suffix}, σ={centroid_std:.2f}", zorder=1)

        ax.set_xlabel("Time Lag")
        ax.set_ylabel("Correlation Coefficient / Density")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()