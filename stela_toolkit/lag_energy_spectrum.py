import numpy as np
import matplotlib.pyplot as plt

from .frequency_binning import FrequencyBinning
from .lag_frequency_spectrum import LagFrequencySpectrum


class LagEnergySpectrum:
    """
    Compute the time lag as a function of energy between two sets of light curves or GP models.

    This class accepts lists of LightCurve objects or trained GaussianProcess models, one per energy bin.
    If the inputs are GP models, the most recently generated samples will be used automatically.
    If no samples are found, 1000 realizations will be generated on a 1000-point grid.

    Each light curve pair (one per energy bin) is used to compute a single lag by integrating the
    cross-spectrum over a specified frequency range. This yields one lag per energy bin, forming
    the lag-energy spectrum.

    A **positive lag** means that the time series in `lcs_or_models1` is **lagging behind**
    the common reference band `lcs_or_models2`.

    Coherence values are also computed for each energy bin to assess correlation strength,
    and noise bias correction can be applied to the coherence before estimating uncertainties.

    Parameters
    ----------
    lcs_or_models1 : list of LightCurve or GaussianProcess
        First set of inputs, one per energy bin.
    
    lcs_or_models2 : list of LightCurve or GaussianProcess
        Input light curve or trained GP model for shared reference band.
        List will typically contain a single object, except for the case of a broad reference band
        where each object is a model trained for the (reference band - band of interest).

    fmin : float
        Minimum frequency to include when integrating.
    
    fmax : float
        Maximum frequency to include when integrating.
    
    bin_edges : array-like
        Edges of the energy bins corresponding to the light curves.
    
    subtract_coh_bias : bool, optional
        Whether to subtract the coherence noise bias before estimating lag uncertainties.

    subtract_from_ref : bool, optional 
        Whether to subtract each band of interest from the common reference band.
        Use to remove shared variability when the reference band is a broad
        band that includes each of the bands of interest.
        
        - Use only for regular sampled input light curves and not a model. A model 
        should be trained on each instance 

    Attributes
    ----------
    energies : array-like
        Mean energy of each bin.
    
    energy_widths : array-like
        Half-width of each energy bin.
    
    lags : array-like
        Integrated time lag per energy bin.
    
    lag_errors : array-like
        Uncertainties (1 sigma) in each lag value.
    
    cohs : array-like
        Coherence values per energy bin.
    
    coh_errors : array-like
        Uncertainties (1 sigma) in the coherence values.
    """

    def __init__(self,
                 lcs_or_models1,
                 lcs_or_models2,
                 fmin,
                 fmax,
                 bin_edges=[],
                 subtract_coh_bias=True,
                 subtract_from_ref=False):
        
        if subtract_from_ref and type(lcs_or_models2[0]).__name__ != "LightCurve":
            raise AttributeError("Subtract_from_ref=True only works for regularly sampled data! " \
            "Separate GP models should be trained on the common reference band - each band of interest first.")
        
        self.data_models1 = lcs_or_models1
        self.data_models2 = lcs_or_models2

        self.energies = [np.mean([bin_edges[i], bin_edges[i+1]]) for i in range(len(bin_edges[:-1]))]
        self.energies = np.array(self.energies)
        self.energy_widths = np.diff(bin_edges) / 2

        self.fmin, self.fmax = fmin, fmax
        lag_spectrum = self.compute_lag_spectrum(subtract_coh_bias=subtract_coh_bias,
                                                 subtract_from_ref=subtract_from_ref
                                                )
        self.lags, self.lag_errors, self.cohs, self. coh_errors = lag_spectrum

    def compute_lag_spectrum(self, subtract_coh_bias, subtract_from_ref):
        """
        Compute the lag and coherence for each energy bin.

        Parameters
        ----------
        subtract_coh_bias : bool
            Whether to subtract Poisson noise bias from the coherence.

        subtract_from_ref : bool, optional 
            Whether to subtract each band of interest from the common reference band.
            Use to remove shared variability when the reference band is a broad
            band that includes each of the bands of interest.

        Returns
        -------
        lags : list
            List of integrated lags for each bin.
        
        lag_errors : list
            List of lag uncertainties.
        
        cohs : list
            List of mean coherence values.
        
        coh_errors : list
            List of coherence uncertainties.
        """

        lags, lag_errors, cohs, coh_errors = [], [], [], []
        for i in range(len(self.data_models1)):
            ref = self.data_models2[i] if len(self.data_models2) > 1 else self.data_models2

            if subtract_from_ref:
                ref -= self.data_models1[i]

            lfs = LagFrequencySpectrum(self.data_models1[i],
                                       ref,
                                       fmin=self.fmin,
                                       fmax=self.fmax,
                                       num_bins=1,
                                       subtract_coh_bias=subtract_coh_bias,
                                       )
            lags.append(lfs.lags)
            lag_errors.append(lfs.lag_errors)
            cohs.append(lfs.cohs)
            coh_errors.append(lfs.coh_errors)

        return lags, lag_errors, cohs, coh_errors

    def plot(self, energies=None, energy_widths=None, lags=None, lag_errors=None, cohs=None, coh_errors=None, **kwargs):
        """
        Plot the lag-energy spectrum and associated coherence values.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for customizing the plot (e.g., xlabel, xscale, yscale).
        """
        energies = self.energies if energies is None else energies
        energy_widths = self.energy_widths if energy_widths is None else energy_widths
        lags = self.lags if lags is None else lags
        lag_errors = self.lag_errors if lag_errors is None else lag_errors
        cohs = self.cohs if cohs is None else cohs
        coh_errors = self.coh_errors if coh_errors is None else coh_errors

        figsize = kwargs.get('figsize', (8, 6))
        xlabel = kwargs.get('xlabel', 'Energy')
        ylabel = kwargs.get('ylabel', 'Time Lag')
        xscale = kwargs.get('xscale', 'log')
        yscale = kwargs.get('yscale', 'linear')

        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=figsize, sharex=True)
        plt.subplots_adjust(hspace=0.05)

        # Lag-energy spectrum
        ax1.errorbar(energies, lags, xerr=energy_widths, yerr=lag_errors, fmt='o', color='black', ms=3, lw=1.5)
        ax1.set_xscale(xscale)
        ax1.set_yscale(yscale)
        ax1.set_ylabel(ylabel, fontsize=12)
        ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax1.tick_params(which='both', direction='in', length=6, width=1, top=True, right=True, labelsize=12)

        # Coherence spectrum
        if cohs is not None and coh_errors is not None:
            ax2.errorbar(energies, cohs, xerr=energy_widths, yerr=coh_errors, fmt='o', color='black', ms=3, lw=1.5)
            ax2.set_xscale(xscale)
            ax2.set_ylabel('Coherence', fontsize=12)
            ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            ax2.tick_params(which='both', direction='in', length=6, width=1, top=True, right=True, labelsize=12)

        fig.text(0.5, 0.04, xlabel, ha='center', va='center', fontsize=12)
        plt.tight_layout()
        plt.show()
        
    def count_frequencies_in_bins(self, fmin=None, fmax=None, num_bins=None, bin_type=None, bin_edges=[]):
        """
        Counts the number of frequencies in each frequency bin.
        Wrapper method to use FrequencyBinning.count_frequencies_in_bins with class attributes.
        """

        return FrequencyBinning.count_frequencies_in_bins(
            self, fmin=fmin, fmax=fmax, num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges
        )