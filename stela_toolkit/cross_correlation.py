import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from ._check_inputs import _CheckInputs


class CrossCorrelation:
    """
    Compute the time-domain cross-correlation function (CCF) between two light curves or GP models.

    This class supports three primary use cases:

    1. **Regularly sampled `LightCurve` objects**  
       Computes the CCF directly using Pearson correlation across lag values. Requires aligned time grids.

    2. **Irregularly sampled `LightCurve` objects**  
       Uses the interpolated cross-correlation method (ICCF; Gaskell & Peterson 1987), which linearly interpolates
       one light curve onto the other’s grid to estimate correlations across lags despite uneven sampling.

    3. **`GaussianProcess` models**  
       If both inputs are trained GP models with sampled realizations (via `.sample()`), STELA computes the CCF for each
       realization pair, then averages the resulting peak and centroid lags. The final outputs are returned as tuples:
       `(mean, standard deviation)` for peak lag, centroid lag, and maximum correlation (rmax).
       
       If samples have not yet been generated, 1000 realizations will be drawn automatically on a 1000-point time grid.

    Additionally, optional Monte Carlo resampling is available to assess lag uncertainties from observational errors.

    Parameters
    ----------
    lc_or_model1 : LightCurve or GaussianProcess
        First input light curve or trained GP model.
    
    lc_or_model2 : LightCurve or GaussianProcess
        Second input light curve or trained GP model.
    
    run_monte_carlo : bool, optional
        Whether to estimate lag uncertainties using Monte Carlo resampling.
    
    n_trials : int, optional
        Number of Monte Carlo trials.
    
    min_lag : float or "auto", optional
        Minimum lag to evaluate. If "auto", set to `-duration / 2`.
    
    max_lag : float or "auto", optional
        Maximum lag to evaluate. If "auto", set to `+duration / 2`.
    
    dt : float or "auto", optional
        Time step for lag evaluation. If "auto", set to 1/5 of the mean sampling interval.
    
    centroid_threshold : float, optional
        Threshold (as a fraction of peak correlation) for defining the centroid lag region.
    
    mode : {"regular", "interp"}, optional
        CCF computation mode. Use "regular" for direct shifting, or "interp" for ICCF-based interpolation.
    
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
                 run_monte_carlo=False,
                 n_trials=1000,
                 min_lag="auto",
                 max_lag="auto",
                 dt="auto",
                 centroid_threshold=0.8,
                 mode="regular",
                 rmax_threshold=0.0):

        req_reg_samp = True if mode=="regular" else False
        data1 = _CheckInputs._check_lightcurve_or_model(lc_or_model1, req_reg_samp=req_reg_samp)
        data2 = _CheckInputs._check_lightcurve_or_model(lc_or_model2, req_reg_samp=req_reg_samp)

        if data1['type'] == 'model':
            if not hasattr(lc_or_model1, 'samples'):
                raise ValueError("Model 1 must have generated samples via GP.sample().")
            self.times = lc_or_model1.pred_times.numpy()
            self.rates1 = lc_or_model1.samples
            self.is_model1 = True
        else:
            self.times, self.rates1, self.errors1 = data1['data']
            self.is_model1 = False

        if data2['type'] == 'model':
            if not hasattr(lc_or_model2, 'samples'):
                raise ValueError("Model 2 must have generated samples via GP.sample().")
            self.times = lc_or_model2.pred_times.numpy()
            self.rates2 = lc_or_model2.samples
            self.is_model2 = True
        else:
            self.times, self.rates2, self.errors2 = data2['data']
            self.is_model2 = False

        self.n_trials = n_trials
        self.centroid_threshold = centroid_threshold
        self.mode = mode
        self.rmax_threshold = rmax_threshold

        duration = self.times[-1] - self.times[0]
        self.min_lag = -duration / 2 if min_lag=="auto" else min_lag
        self.max_lag = duration / 2 if max_lag=="auto" else max_lag

        if mode == "regular":
            self.dt = np.diff(self.times)[0]
            self.lags = np.arange(self.min_lag, self.max_lag + self.dt, self.dt)

            if self.is_model1 and self.is_model2:
                if self.rates1.shape[0] != self.rates2.shape[0]:
                    raise ValueError("Model sample shapes do not match.")
                
                peak_lags, centroid_lags, rmaxs = [], [], []

                # Compute ccf and lags for each pair of realizations
                for i in range(self.rates1.shape[0]):
                    ccf = self.compute_ccf(self.rates1[i], self.rates2[i])
                    peak_lag, centroid_lag = self.find_peak_and_centroid(self.lags, ccf)
                    rmax = np.max(ccf)

                    peak_lags.append(peak_lag)
                    centroid_lags.append(centroid_lag)
                    rmaxs.append(rmax)

                self.peak_lag = (np.mean(peak_lags), np.std(peak_lags))
                self.centroid_lag = (np.mean(centroid_lags), np.std(centroid_lags))
                self.rmax = (np.mean(rmaxs), np.std(rmaxs))
                    
            else:
                self.ccf = self.compute_ccf(self.rates1, self.rates2)

        else:
            self.dt = np.mean(np.diff(self.times)) / 5 if dt=="auto" else dt
            self.lags = np.arange(self.min_lag, self.max_lag + self.dt, self.dt)
            self.ccf = self.compute_ccf_interp()

            self.rmax = np.max(self.ccf)
            self.peak_lag, self.centroid_lag = self.find_peak_and_centroid(self.lags, self.ccf)

        self.peak_lags_mc = None
        self.centroid_lags_mc = None
        self.peak_lag_ci = None
        self.centroid_lag_ci = None

        if run_monte_carlo:
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

    def compute_ccf_interp(self):
        """
        Compute the cross-correlation function using symmetric linear interpolation.

        Returns
        -------
        ccf : ndarray
            Interpolated cross-correlation values for each lag.
        """

        interp1 = interp1d(self.times, self.rates1, bounds_error=False, fill_value=0.0)
        interp2 = interp1d(self.times, self.rates2, bounds_error=False, fill_value=0.0)
        ccf = []

        for lag in self.lags:
            t_shift1 = self.times + lag
            t_shift2 = self.times - lag

            mask1 = (t_shift1 >= self.times[0]) & (t_shift1 <= self.times[-1])
            mask2 = (t_shift2 >= self.times[0]) & (t_shift2 <= self.times[-1])

            if np.sum(mask1) < 2 or np.sum(mask2) < 2:
                ccf.append(0.0)
                continue

            r1 = np.corrcoef(self.rates1[mask1], interp2(t_shift1[mask1]))[0, 1]
            r2 = np.corrcoef(self.rates2[mask2], interp1(t_shift2[mask2]))[0, 1]
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
        Run Monte Carlo simulations to estimate lag uncertainties.

        Perturbs input light curves based on their errors and computes peak and centroid
        lags for each realization.

        Returns
        -------
        peak_lags : ndarray
            Peak lag values from all trials.
        
        centroid_lags : ndarray
            Centroid lag values from all trials.
        """

        peak_lags = []
        centroid_lags = []

        for _ in range(self.n_trials):
            r1_pert = np.random.normal(self.rates1, self.errors1)
            r2_pert = np.random.normal(self.rates2, self.errors2)

            if self.mode == "interp":
                interp1 = interp1d(self.times, r1_pert, bounds_error=False, fill_value=0.0)
                interp2 = interp1d(self.times, r2_pert, bounds_error=False, fill_value=0.0)
                ccf = []

                for lag in self.lags:
                    t_shift1 = self.times + lag
                    t_shift2 = self.times - lag

                    mask1 = (t_shift1 >= self.times[0]) & (t_shift1 <= self.times[-1])
                    mask2 = (t_shift2 >= self.times[0]) & (t_shift2 <= self.times[-1])

                    if np.sum(mask1) < 2 or np.sum(mask2) < 2:
                        ccf.append(0.0)
                        continue

                    r1 = np.corrcoef(r1_pert[mask1], interp2(t_shift1[mask1]))[0, 1]
                    r2 = np.corrcoef(r2_pert[mask2], interp1(t_shift2[mask2]))[0, 1]
                    ccf_val = (r1 + r2) / 2
                    ccf.append(ccf_val)
                ccf = np.array(ccf)
            else:
                ccf = self.compute_ccf(r1_pert, r2_pert)

            if np.max(ccf) < self.rmax_threshold:
                continue

            peak, centroid = self.find_peak_and_centroid(self.lags, ccf)
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
        Plot the cross-correlation function and optional Monte Carlo lag distributions.

        Arguments
        ----------
        show_mc : bool
            Plots results of Monte Carlo peak and centroid lag distributions.
        """

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(self.lags, self.ccf, label="CCF", color='black')
        ax.axvline(self.peak_lag, color='red', linestyle='--',
                   label=f"Peak lag = {self.peak_lag:.2f}")
        ax.axvline(self.centroid_lag, color='blue', linestyle=':',
                   label=f"Centroid lag = {self.centroid_lag:.2f}")
        ax.set_xlabel("Lag (same unit as input)")
        ax.set_ylabel("Correlation coefficient")
        ax.grid(True)
        ax.legend()

        if show_mc and self.peak_lags_mc is not None:
            fig_mc, ax_mc = plt.subplots(1, 2, figsize=(10, 4))
            ax_mc[0].hist(self.peak_lags_mc, bins=30, color='red', alpha=0.7)
            ax_mc[0].set_title("Peak Lag Distribution (MC)")
            ax_mc[0].set_xlabel("Lag")
            ax_mc[0].set_ylabel("Count")
            ax_mc[0].grid(True)

            ax_mc[1].hist(self.centroid_lags_mc, bins=30, color='blue', alpha=0.7)
            ax_mc[1].set_title("Centroid Lag Distribution (MC)")
            ax_mc[1].set_xlabel("Lag")
            ax_mc[1].set_ylabel("Count")
            ax_mc[1].grid(True)

        plt.tight_layout()
        plt.show()