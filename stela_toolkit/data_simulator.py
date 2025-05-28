import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, unit_impulse
from scipy.stats import norm, lognorm
from .data_loader import LightCurve

class SimulateLightCurve:
    """
    Simulates light curves with a specified power spectral density (PSD) and optional lag injection
    via an impulse response function (IRF). The light curve is generated using the Timmer & Koenig
    (1995) method: amplitudes in Fourier space are set by the PSD, and phases are randomized
    uniformly. The signal is then inverse-Fourier transformed to obtain the time-domain light curve.

    The clean light curve is rescaled to have the desired mean and standard deviation. Poisson noise
    may be added to simulate counting statistics, including optional background noise.

    Supports both regularly and irregularly sampled time grids:
    - For regular grids: the light curve is oversampled (by default 10×) and then trimmed.
    - For irregular grids: the light curve is generated on a fine grid and sampled at the closest
      points (no interpolation).

    Optional lag injection is supported by convolving the light curve with an IRF.

    Parameters
    ----------
    time_grid : ndarray
        The array of time stamps for which to simulate the light curve. Can be regular or irregular,
        but must be sorted and contain at least two points.

    psd_type : str
        Type of power spectral density (PSD) to use. Options are:
        - 'powerlaw': a simple power law PSD.
        - 'broken_powerlaw': a PSD with two slopes joined at a break frequency.

    psd_params : dict
        Parameters for the PSD. Required keys depend on the PSD type:
        - For 'powerlaw': {'slope', 'plnorm'}
        - For 'broken_powerlaw': {'slope1', 'f_break', 'slope2', 'plnorm'}

    mean : float
        Desired mean count rate of the simulated light curve (after rescaling).

    std : float
        Desired standard deviation of the simulated light curve (after rescaling).

    add_noise : bool, optional
        If True, Poisson noise is added to the light curve (default: False).

    exposure_times : ndarray or None, optional
        Exposure durations to use for adding the Poisson noise. If None,
        the time spacing is used to approximate exposure times.

    bkg_rate : float, optional
        Background rate in counts per unit time to include in the noise simulation (default: 0.0).
        If set, Poisson noise from two background realizations is added and subtracted.

    oversample : int, optional
        For regular grids: how much to oversample before trimming to the target grid (default: 10).

    fine_factor : int, optional
        For irregular grids: how densely to simulate the light curve before selecting closest points
        (default: 100).

    inject_lag : bool, optional
        If True, applies lag injection by convolving with an IRF (default: False).

    response_type : str or None, optional
        Type of impulse response function to use. Options are:
        - 'delta': a single delay spike at a fixed lag
        - 'normal': Gaussian-shaped IRF
        - 'lognormal': log-normal IRF with skewed tail
        - 'manual': user-supplied kernel
        If None, no lag is injected (default: None).

    response_params : dict or None, optional
        Parameters for the selected response_type:
        - For 'delta': {'lag': float}
        - For 'normal': {'mean': float, 'sigma': float, 'duration': float (optional)}
        - For 'lognormal': {'median': float, 'sigma': float, 'duration': float (optional)}
        - For 'manual': {'response': array_like}
    """
    
    def __init__(self,
                 time_grid,
                 psd_type,
                 psd_params,
                 mean,
                 std,
                 add_noise=False,
                 exposure_times=None,
                 bkg_rate=0.0,
                 oversample=10,
                 fine_factor=100,
                 inject_lag=False,
                 response_type=None,
                 response_params=None):

        self.time_grid = np.asarray(time_grid)
        self.psd_type = psd_type
        self.psd_params = psd_params
        self.mean = mean
        self.std = std
        self.oversample = oversample
        self.fine_factor = fine_factor
        self.bkg_rate = bkg_rate
        self.inject_lag = inject_lag
        self.response_type = response_type
        self.response_params = response_params
        self.exposure_times = np.asarray(exposure_times) if exposure_times is not None else None

        result = self.generate(self.time_grid)
        if isinstance(result, tuple):
            rates, rates_lagged = result
        else:
            rates = result
            rates_lagged = None

        errors = np.zeros(len(rates))
        if add_noise:
            rates, errors = self.add_poisson_noise(rates, self.time_grid, bkg_rate=self.bkg_rate, exposure_times=self.exposure_times)
            if rates_lagged is not None:
                rates_lagged, _ = self.add_poisson_noise(rates_lagged, self.time_grid, bkg_rate=self.bkg_rate, exposure_times=self.exposure_times)

        self.rates = rates
        self.errors = errors
        self.simlc = LightCurve(times=self.time_grid, rates=rates, errors=errors)
        self.simlc_lagged = (
            LightCurve(times=self.time_grid, rates=rates_lagged, errors=errors)
            if rates_lagged is not None else None
        )

    def generate(self, time_grid):
        """
        Generate the clean (noise-free) light curve.

        Handles regular vs. irregular time grids and applies normalization.
        If lag injection is enabled, convolves with a response function.

        Parameters
        ----------
        time_grid : array-like
            Desired output time grid.

        Returns
        -------
        rates : ndarray
            Simulated light curve values.
        rates_lagged : ndarray or None
            Lagged version of the light curve if `inject_lag` is True.
        """

        time_grid = np.array(time_grid)
        n_target = len(time_grid)
        dt_array = np.diff(time_grid)
        is_regular = np.allclose(dt_array, dt_array[0], rtol=1e-5)

        if is_regular:
            n_sim = int(self.oversample * n_target)
            t_sim = np.linspace(time_grid[0], time_grid[-1], n_sim)
            lc_sim = self._simulate_on_grid(t_sim)

            start = (n_sim - n_target) // 2
            end = start + n_target
            lc = lc_sim[start:end]

            lc -= np.mean(lc)
            lc /= np.std(lc)
            lc = lc * self.std + self.mean

            if self.inject_lag:
                kernel = self._build_impulse_response(time_grid)
                convolved_full = fftconvolve(lc_sim, kernel, mode="full")
                convolved = convolved_full[start:end]

                convolved -= np.mean(convolved)
                convolved /= np.std(convolved)
                convolved = convolved * self.std + self.mean
                return lc, convolved
            else:
                return lc

        else:
            n_fine = int(self.fine_factor * len(time_grid))
            t_fine = np.linspace(time_grid.min(), time_grid.max(), n_fine)
            lc_fine = self._simulate_on_grid(t_fine)

            lc_fine -= np.mean(lc_fine)
            lc_fine /= np.std(lc_fine)
            lc_fine = lc_fine * self.std + self.mean

            if self.inject_lag:
                kernel = self._build_impulse_response(t_fine)
                lc_fine_lagged_full = fftconvolve(lc_fine, kernel, mode="full")
                lc_fine_lagged = lc_fine_lagged_full[:len(t_fine)]
            else:
                lc_fine_lagged = None

            indices = np.searchsorted(t_fine, time_grid, side="left")
            indices = np.clip(indices, 0, n_fine - 1)
            for i, ti in enumerate(time_grid):
                if indices[i] > 0 and abs(t_fine[indices[i] - 1] - ti) < abs(t_fine[indices[i]] - ti):
                    indices[i] -= 1

            lc = lc_fine[indices]
            if lc_fine_lagged is not None:
                lc_lagged = lc_fine_lagged[indices]
                return lc, lc_lagged
            else:
                return lc

    def add_poisson_noise(self, lc, time_grid, bkg_rate=0.0, exposure_times=None, min_error_floor=1e-10):
        """
        Add Poisson noise to a simulated light curve.

        Parameters
        ----------
        lc : ndarray
            Clean light curve values.
        time_grid : ndarray
            Time values.
        bkg_rate : float, optional
            Background count rate.
        exposure_times : ndarray or None, optional
            Exposure duration for each point. If None, use time spacing
            to approximate integration time per bin.
        min_error_floor : float, optional
            Minimum uncertainty to avoid zeros.

        Returns
        -------
        noisy_lc : ndarray
            Noisy light curve.
        noise_estimate : ndarray
            Estimated error bars.
        """

        lc = np.asarray(lc)
        time_grid = np.asarray(time_grid)

        if len(time_grid) < 2:
            raise ValueError("time_grid must have at least two points.")

        if exposure_times is not None:
            dt = np.asarray(exposure_times)
        else:
            dt_array = np.diff(time_grid)
            if np.allclose(dt_array, dt_array[0], rtol=1e-5):
                dt = dt_array[0]
            else:
                dt = np.zeros_like(time_grid)
                dt[1:-1] = (time_grid[2:] - time_grid[:-2]) / 2
                dt[0] = time_grid[1] - time_grid[0]
                dt[-1] = time_grid[-1] - time_grid[-2]

        counts = lc * dt
        counts = np.clip(counts, 0, None)

        noisy_counts = np.random.poisson(counts)

        if bkg_rate > 0:
            bkg_counts1 = np.random.poisson(bkg_rate * dt)
            bkg_counts2 = np.random.poisson(bkg_rate * dt)
            noisy_counts += bkg_counts1
            noisy_counts -= bkg_counts2

        noisy_lc = noisy_counts / dt

        with np.errstate(divide='ignore', invalid='ignore'):
            noise_estimate = np.sqrt(np.clip(noisy_counts, 0, None)) / dt
            noise_estimate = np.where(noise_estimate == 0, min_error_floor, noise_estimate)

        return noisy_lc, noise_estimate

    def add_regular_gaps(self, lc, time_grid, gap_period, gap_duration):
        """
        Simulate regular gaps in the light curve.

        Parameters
        ----------
        lc : ndarray
            Input light curve values.
        time_grid : ndarray
            Time values.
        gap_period : float
            Period between gaps.
        gap_duration : float
            Duration of each gap.

        Returns
        -------
        gapped_lc : ndarray
            Light curve with NaNs inserted for gaps.
        """
                
        lc = np.asarray(lc)
        time_grid = np.asarray(time_grid)
        gapped_lc = lc.copy()

        time_since_start = (time_grid - time_grid[0]) % gap_period
        in_gap = time_since_start < gap_duration
        gapped_lc[in_gap] = np.nan

        return gapped_lc

    def create_psd(self, freq):
        """
        Construct the PSD array based on the selected type and parameters.

        Parameters
        ----------
        freq : ndarray
            Frequency array.

        Returns
        -------
        psd : ndarray
            Power spectral density values.
        """

        freq = np.abs(freq)
        psd = np.zeros_like(freq)
        nonzero_mask = freq > 0  # avoid division by 0
        plnorm = self.psd_params.get("plnorm", 1.0)

        if self.psd_type == "powerlaw":
            slope = self.psd_params.get("slope")
            psd[nonzero_mask] = plnorm * (2 * np.pi * freq[nonzero_mask]) ** (-slope / 2)

        elif self.psd_type == "broken_powerlaw":
            slope1 = self.psd_params.get("slope1")
            f_break = self.psd_params.get("f_break")
            slope2 = self.psd_params.get("slope2")
            psd[nonzero_mask] = np.where(
                freq[nonzero_mask] <= f_break,
                plnorm * (2 * np.pi * freq[nonzero_mask]) ** (-slope1 / 2),
                plnorm * ((2 * np.pi * f_break) ** ((slope2 - slope1) / 2)) *
                (2 * np.pi * freq[nonzero_mask]) ** (-slope2 / 2)
            )
        else:
            raise ValueError(f"Unsupported PSD type: {self.psd_type}")

        # set freq=0 psd to 0 to avoid infinite psd
        # the value of this will be adjusted by the mean during rescaling
        psd[freq==0] = 0
        return psd
    
    def plot(self):
        """
        Plot the simulated light curve/s. Shows both the original and lagged data (if available).
        """
        plt.figure(figsize=(12, 5))
        plt.plot(self.simlc.times, self.simlc.rates, label='Simulated', lw=1.5)

        if self.simlc_lagged is not None:
            plt.plot(self.simlc_lagged.times, self.simlc_lagged.rates,
                     label='Lagged', lw=1.5, alpha=0.8)

        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def _simulate_on_grid(self, t_sim):
        """
        Generate a Fourier-based light curve realization on a time grid.
        """
        n_sim = len(t_sim)
        dt = t_sim[1] - t_sim[0]
        freqs = np.fft.fftfreq(n_sim, d=dt)

        psd = self.create_psd(freqs)
        phases = np.random.uniform(0, 2*np.pi, n_sim)
        amplitudes = np.sqrt(psd)

        ft = amplitudes * np.exp(1j * phases)

        # check for hermitian symmetry to ensure real-valued IFFT
        if n_sim % 2 == 0:
            ft[int(n_sim / 2) + 1:] = np.conj(ft[1:int(n_sim / 2)][::-1])
        else:
            ft[int(n_sim / 2) + 1:] = np.conj(ft[1:int(n_sim / 2) + 1][::-1])

        lc_sim = np.fft.ifft(ft).real
        return lc_sim
    
    def _build_impulse_response(self, times):
        """
        Generate an impulse response function for lag injection.
        Supported types: 'delta', 'normal', 'lognormal', 'manual'.
        """
        dt = times[1] - times[0]

        if self.response_type == "delta":
            lag = self.response_params["lag"]
            lag_n = int(round(lag / dt))
            response = unit_impulse(len(times), lag_n)
            return response

        elif self.response_type == "normal":
            mu = self.response_params["mean"]
            sigma = self.response_params["sigma"]
            duration = self.response_params.get("duration", 5 * sigma)
            t = np.arange(0, duration, dt)
            kernel = norm.pdf(t, loc=mu, scale=sigma)
            return kernel / np.sum(kernel)

        elif self.response_type == "lognormal":
            median = self.response_params["median"]
            sigma = self.response_params["sigma"]
            duration = self.response_params.get("duration", 5 * median)
            t = np.arange(dt, duration, dt)  # starts at dt to avoid log(0)

            # Convert median + sigma to shape and scale for lognorm
            # lognorm.pdf(x, s, loc=0, scale=median)
            s = sigma
            scale = median
            kernel = lognorm.pdf(t, s=s, scale=scale)
            return kernel / np.sum(kernel)

        elif self.response_type == "manual":
            return np.asarray(self.response_params["response"])

        else:
            raise ValueError(f"Unsupported response_type: {self.response_type}")
