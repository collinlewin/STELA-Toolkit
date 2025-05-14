import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from ._check_inputs import _CheckInputs
from .data_loader import LightCurve

class RegularCCF:
    def __init__(self, lc1, lc2, run_monte_carlo=False, n_trials=1000,
                 min_lag=None, max_lag=None, centroid_threshold=0.8,
                 mode="regular", rmax_threshold=0.0):
        """
        Compute the cross-correlation function (CCF) between two light curves to estimate time lags.

        This class supports both simple regular shifting and linear interpolation (additional method to come!) to calculate 
        the correlation between two light curves over a range of time lags. It can also perform 
        Monte Carlo resampling to estimate uncertainties on the lag measurements.

        - The two input light curves must be regularly sampled and represented as LightCurve objects.
        - If mode='regular', the two light curves must share the exact same time grid.
        - All lags are computed using Pearson correlation coefficients (hence CCF).

        Parameters
        ----------
        lc1 : LightCurve
            The first input light curve.

        lc2 : LightCurve
            The second input light curve.

        run_monte_carlo : bool, optional (default=False)
            Whether to run a Monte Carlo simulation to estimate uncertainty on the peak and centroid lag.

        n_trials : int, optional (default=1000)
            Number of Monte Carlo realizations to run if Monte Carlo is enabled.

        min_lag : float, optional
            The minimum lag (in same units as the time axis) to consider in the CCF.
            Default is -duration/2 of the time series.

        max_lag : float, optional
            The maximum lag (in same units as the time axis) to consider in the CCF.
            Default is +duration/2 of the time series.

        centroid_threshold : float, optional (default=0.8)
            Fraction of the peak correlation to use when defining the centroid lag region.

        mode : str, optional (default='regular')
            Determines the method used for cross-correlation.
            - 'regular' : Use shifting of the time series (requires identical time grid).
            - 'interp'  : Use symmetric linear interpolation of one light curve onto the other.

        rmax_threshold : float, optional (default=0.0)
            Minimum allowed correlation coefficient in a Monte Carlo realization.
            Trials with peak correlation below this threshold will be discarded from the statistics.

        Attributes
        ----------
        lags : ndarray
            Array of lag values evaluated in the CCF.

        ccf : ndarray
            The cross-correlation values corresponding to each lag.

        peak_lag : float
            The lag value corresponding to the maximum correlation.

        centroid_lag : float
            The centroid lag computed using the region where the CCF exceeds `centroid_threshold` peak.

        rmax : float
            The maximum correlation value in the computed CCF.

        peak_lags_mc : ndarray or None
            Peak lag values from the Monte Carlo trials (if run).

        centroid_lags_mc : ndarray or None
            Centroid lag values from the Monte Carlo trials (if run).

        peak_lag_ci : tuple or None
            16th and 84th percentile confidence interval on the peak lag (if MC run).

        centroid_lag_ci : tuple or None
            16th and 84th percentile confidence interval on the centroid lag (if MC run).

        Methods
        -------
        plot(show_mc=True)
            Plot the CCF and optionally the Monte Carlo distributions.
        """
        if not isinstance(lc1, LightCurve) or not isinstance(lc2, LightCurve):
            raise TypeError("Both inputs must be LightCurve objects.")

        t1, r1, e1 = _CheckInputs._check_input_data(lc1, req_reg_samp=True)
        t2, r2, e2 = _CheckInputs._check_input_data(lc2, req_reg_samp=True)

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
        Compute confidence intervals for MC lag distributions.
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
        Plot the CCF and optional Monte Carlo lag distributions (if avalable).
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