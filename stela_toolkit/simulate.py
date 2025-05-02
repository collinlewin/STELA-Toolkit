import numpy as np

class Simulate:
    def __init__(self, time_grid, psd_type, psd_params, mean, std):
        """
        Initialize and immediately generate (and optionally noise) the light curve.

        Parameters
        ----------
        time_grid : ndarray
            The time grid to generate the light curve on.
        psd_type : str
            The type of PSD to use. Options:
            - 'powerlaw'
            - 'broken_powerlaw'
        psd_params : dict
            PSD parameters (depends on psd_type).
        mean : float
            Desired mean of the light curve.
        std : float
            Desired standard deviation of the light curve.

        Attributes
        ----------
        lc_clean : ndarray
            The simulated clean light curve.
        lc_noisy : ndarray or None
            The noisy light curve (if add_noise=True).
        noise_estimate : ndarray or None
            The estimated noise (if add_noise=True).
        """
        self.time_grid = np.asarray(time_grid)
        if len(self.time_grid) < 2:
            raise ValueError("time_grid must have at least two points.")
        if not np.all(np.diff(self.time_grid) > 0):
            raise ValueError("time_grid must be sorted in ascending order.")

        self.psd_type = psd_type
        self.psd_params = psd_params
        self.mean = mean
        self.std = std

        # Generate the clean light curve immediately
        self.lc = self.generate(self.time_grid)

    def generate(self, time_grid):
        """
        Generate a simulated light curve matching the provided time grid.
        """
        time_grid = np.asarray(time_grid)
        if len(time_grid) < 2:
            raise ValueError("time_grid must have at least two points.")

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
        else:
            n_fine = int(self.fine_factor * len(time_grid))
            t_fine = np.linspace(time_grid.min(), time_grid.max(), n_fine)
            lc_fine = self._simulate_on_grid(t_fine)

            indices = np.searchsorted(t_fine, time_grid, side="left")
            indices = np.clip(indices, 0, n_fine - 1)

            for i, ti in enumerate(time_grid):
                if indices[i] > 0 and abs(t_fine[indices[i] - 1] - ti) < abs(t_fine[indices[i]] - ti):
                    indices[i] -= 1

            lc = lc_fine[indices]

        lc -= np.mean(lc)
        lc /= np.std(lc)
        lc = lc * self.std + self.mean
        return lc

    def add_poisson_noise(self, lc, time_grid, bkg_rate=0.0, min_error_floor=1e-10):
        """
        Add Poisson noise to a simulated light curve.
        """
        lc = np.asarray(lc)
        time_grid = np.asarray(time_grid)
        if len(time_grid) < 2:
            raise ValueError("time_grid must have at least two points.")

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