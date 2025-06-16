import numpy as np
import matplotlib.pyplot as plt
from ._check_inputs import _CheckInputs


class CrossCorrelation:
    """
    Compute the time-domain cross-correlation function (CCF) between two light curves or GP models.

    This class supports three primary use cases:

    1. **Regularly sampled `LightCurve` objects**  
       Computes the CCF directly using Pearson correlation across lag values. Requires aligned time grids.

    2. **`GaussianProcess` models**  
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
                 rmax_threshold=0.0):

        data1 = _CheckInputs._check_lightcurve_or_model(lc_or_model1, req_reg_samp=True)
        data2 = _CheckInputs._check_lightcurve_or_model(lc_or_model2, req_reg_samp=True)

        self.is_model1 = data1['type'] == 'model'
        self.is_model2 = data2['type'] == 'model'

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

        self.dt = np.diff(self.times)[0]
        self.lags = np.arange(self.min_lag, self.max_lag + self.dt, self.dt)

        if self.is_model1 and self.is_model2:
            if self.rates1.shape[0] != self.rates2.shape[0]:
                raise ValueError("Model sample shapes do not match.")
            
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

            self.ccf = np.mean(ccfs, axis=0)
            self.peak_lag = (np.mean(self.peak_lags), np.std(self.peak_lags))
            self.centroid_lag = (np.mean(self.centroid_lags), np.std(self.centroid_lags))
            self.rmax = (np.mean(self.rmaxs), np.std(self.rmaxs))
                
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
            peak_data = None
            centroid_data = None

            if self.peak_lags_mc is not None:
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