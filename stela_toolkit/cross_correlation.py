import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from ._check_inputs import _CheckInputs


class CrossCorrelation:
    """
    Compute the time-domain cross-correlation function (CCF) between two light curves or GP models.

    This class supports three primary use cases:

    1. **Regularly sampled LightCurve objects**: The CCF is computed via direct shifting
    using Pearson correlation coefficients across lag values.

    2. **Irregularly sampled LightCurve objects**: Uses the interpolated cross-correlation
    method (ICCF), introduced by Gaskell & Peterson (1987), which linearly interpolates
    one light curve onto the other's grid to allow for lag estimation despite gaps.

    3. **GaussianProcess models**: If both inputs are trained GP models, the CCF is computed
    across all sampled realizations and averaged. If no samples exist, 1000 will be generated
    automatically on a 1000-point grid. Lag uncertainties are derived from the spread in
    lag values across realizations.

    Monte Carlo resampling is also available for estimating confidence intervals using observational
    error bars.

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
    min_lag : float, optional
        Minimum lag (in time units) to evaluate.
    max_lag : float, optional
        Maximum lag to evaluate.
    centroid_threshold : float, optional
        Threshold (fraction of peak CCF) for defining centroid lag region.
    mode : {'regular', 'interp'}, optional
        CCF computation mode. Use 'regular' for direct shifting (requires aligned time grids),
        or 'interp' for ICCF-style interpolation.
    rmax_threshold : float, optional
        Monte Carlo trials with maximum correlation below this value are discarded.

    Attributes
    ----------
    lags : ndarray
        Array of lag values evaluated.
    ccf : ndarray
        Cross-correlation coefficients.
    peak_lag : float
        Lag corresponding to the peak correlation.
    centroid_lag : float
        Centroid lag from the high-correlation region.
    rmax : float
        Maximum correlation coefficient.
    peak_lags_mc : ndarray or None
        Peak lags from Monte Carlo trials.
    centroid_lags_mc : ndarray or None
        Centroid lags from Monte Carlo trials.
    peak_lag_ci : tuple or None
        Confidence interval (16th–84th percentile) on peak lag.
    centroid_lag_ci : tuple or None
        Confidence interval on centroid lag.
    """
    
    def __init__(self,
                 lc_or_model1,
                 lc_or_model2,
                 run_monte_carlo=False,
                 n_trials=1000,
                 min_lag=None,
                 max_lag=None,
                 centroid_threshold=0.8,
                 mode="regular",
                 rmax_threshold=0.0):

        data1 = _CheckInputs._check_lightcurve_or_model(lc_or_model1)
        data2 = _CheckInputs._check_lightcurve_or_model(lc_or_model2)

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

        t1, r1, e1 = _CheckInputs._check_input_data(lc_or_model1, req_reg_samp=True)
        t2, r2, e2 = _CheckInputs._check_input_data(lc_or_model2, req_reg_samp=True)

        if mode == "regular" and not np.array_equal(t1, t2):
            raise ValueError("Time grids of both light curves must match for regular mode.")

        self.times = t1
        self.dt = np.round(np.diff(self.times)[0], 10)
        self.rates1, self.rates2 = r1, r2
        self.errors1, self.errors2 = e1, e2
        self.n_trials = n_trials
        self.centroid_threshold = centroid_threshold
        self.mode = mode
        self.rmax_threshold = rmax_threshold

        duration = self.times[-1] - self.times[0]
        self.min_lag = min_lag if min_lag is not None else -duration / 2
        self.max_lag = max_lag if max_lag is not None else duration / 2
        self.lags = np.arange(self.min_lag, self.max_lag + self.dt, self.dt)

        if mode == "interp":
            self.ccf = self.compute_ccf_interp()
        else:
            self.lags, self.ccf = self.compute_ccf(self.rates1, self.rates2)

        if self.is_model1 and self.is_model2:
            if self.rates1.shape[0] != self.rates2.shape[0]:
                raise ValueError("Model sample shapes do not match.")
            self.ccf = np.mean([
                self.compute_ccf(self.rates1[i], self.rates2[i])[1]
                for i in range(self.rates1.shape[0])
            ], axis=0)
            self.lags = np.arange(self.min_lag, self.max_lag + self.dt, self.dt)

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

        min_shift = int(self.min_lag / self.dt)
        max_shift = int(self.max_lag / self.dt)
        lags = np.arange(min_shift, max_shift + 1) * self.dt
        ccf = []

        for shift in range(min_shift, max_shift + 1):
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

        return lags, np.array(ccf)

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
        Identify the peak and centroid lag of the cross-correlation function.

        Parameters
        ----------
        lags : ndarray
            Array of lag values.
        ccf : ndarray
            Cross-correlation function values.

        Returns
        -------
        peak_lag : float
            Lag at maximum correlation.
        centroid_lag : float
            Centroid lag within the high-correlation region.
        """
        
        peak_idx = np.nanargmax(ccf)
        peak_val = ccf[peak_idx]
        peak_lag = lags[peak_idx]

        mask = ccf >= (self.centroid_threshold * peak_val)
        if np.any(mask):
            centroid = np.sum(lags[mask] * ccf[mask]) / np.sum(ccf[mask])
        else:
            centroid = np.nan

        return peak_lag, centroid

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
                _, ccf = self.compute_ccf(r1_pert, r2_pert)

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

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for customizing the plot.
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