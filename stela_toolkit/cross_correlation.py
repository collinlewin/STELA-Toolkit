import numpy as np
import matplotlib.pyplot as plt
from ._check_inputs import _CheckInputs
from .data_loader import LightCurve

class RegularCCF:
    def __init__(self, lc1, lc2, run_monte_carlo=False, n_trials=1000,
                 max_lag=None, centroid_threshold=0.8):
        """
        Cross-correlation function (CCF) for regularly sampled LightCurve objects, i.e., 
        after using an interpolation/regression class like GaussianProcess.

        Parameters
        ----------
        lc1, lc2 : LightCurve
            LightCurve objects. Must be uniformly sampled and on the same time grid.
        run_monte_carlo : bool
            Whether to run flux-randomized Monte Carlo to estimate lag uncertainties.
        n_trials : int
            Number of Monte Carlo iterations.
        max_lag : float, optional
            Maximum lag to consider. If None, defaults to half of total duration.
        centroid_threshold : float
            Threshold for centroid calculation, relative to peak CCF (default: 0.8).
        """
        if not isinstance(lc1, LightCurve) or not isinstance(lc2, LightCurve):
            raise TypeError("Both inputs must be LightCurve objects.")

        t1, r1, e1 = _CheckInputs._check_input_data(lc1, req_reg_samp=True)
        t2, r2, e2 = _CheckInputs._check_input_data(lc2, req_reg_samp=True)

        if not np.array_equal(t1, t2):
            raise ValueError("Time grids of both light curves must match.")

        self.times = t1
        self.dt = np.round(np.diff(self.times)[0], 10)
        self.rates1, self.rates2 = r1, r2
        self.errors1, self.errors2 = e1, e2
        self.n_trials = n_trials
        self.centroid_threshold = centroid_threshold

        self.max_lag = max_lag if max_lag is not None else (self.times[-1] - self.times[0]) / 2

        # Core CCF computation
        self.lags, self.ccf = self.compute_ccf()
        self.rmax = np.max(self.ccf)
        self.peak_lag, self.centroid_lag = self.find_peak_and_centroid(self.lags, self.ccf)

        # MC results with default confidence intervals--(16, 84): 1 sigma
        self.peak_lags_mc = None
        self.centroid_lags_mc = None
        self.peak_lag_ci = None
        self.centroid_lag_ci = None

        if run_monte_carlo:
            if np.all(self.errors1 == 0) or np.all(self.errors2 == 0):
                print("⚠️  Skipping Monte Carlo: zero errors in one or both light curves.")
            else:
                self.peak_lags_mc, self.centroid_lags_mc = self.run_monte_carlo()
                self.compute_confidence_intervals()  

    def compute_ccf(self):
        max_shift = int(self.max_lag / self.dt)
        lags = np.arange(-max_shift, max_shift + 1) * self.dt
        ccf = []

        for shift in range(-max_shift, max_shift + 1):
            if shift < 0:
                x = self.rates1[:shift]
                y = self.rates2[-shift:]
            elif shift > 0:
                x = self.rates1[shift:]
                y = self.rates2[:-shift]
            else:
                x = self.rates1
                y = self.rates2

            if len(x) < 2:
                ccf.append(0.0)
            else:
                r = np.corrcoef(x, y)[0, 1]
                ccf.append(r)

        return lags, np.array(ccf)

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

            lags, ccf = self.compute_ccf_from(r1_pert, r2_pert)
            peak, centroid = self.find_peak_and_centroid(lags, ccf)

            peak_lags.append(peak)
            centroid_lags.append(centroid)

        return np.array(peak_lags), np.array(centroid_lags)

    def compute_ccf_from(self, rates1, rates2):
        max_shift = int(self.max_lag / self.dt)
        lags = np.arange(-max_shift, max_shift + 1) * self.dt

        ccf = []
        for shift in range(-max_shift, max_shift + 1):
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

    def compute_confidence_intervals(self, lower_percentile=16, upper_percentile=84):
        """
        Compute confidence intervals for MC lag distributions.

        Parameters
        ----------
        lower_percentile : float
            Lower percentile for CI (default 16).
        upper_percentile : float
            Upper percentile for CI (default 84).
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