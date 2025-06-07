import numpy as np
from ._check_inputs import _CheckInputs
from .frequency_binning import FrequencyBinning
from .plot import Plotter
from .data_loader import LightCurve


class PowerSpectrum:
    """
    Compute the power spectrum of a light curve using the FFT.

    This class accepts either a STELA LightCurve object or a trained GaussianProcess model.
    If a GaussianProcess is passed, the most recently generated samples are used. 
    If no samples exist, the toolkit will automatically generate 1000 posterior realizations
    on a 1000-point grid.

    For single light curves, the FFT is applied directly to the time series.
    For GP models, the power spectrum is computed for each sampled realization,
    and the mean and standard deviation across all samples are returned.

    Power spectra are computed in variance units by default (i.e., normalized to units
    of squared flux), allowing for direct interpretation in the context of variability
    amplitude and fractional RMS.

    Frequency binning is supported via linear, logarithmic, or user-defined bins.

    Parameters
    ----------
    lc_or_model : LightCurve or GaussianProcess
        Input light curve or trained GP model.
    
    fmin : float or 'auto', optional
        Minimum frequency to include. If 'auto', uses the lowest nonzero FFT frequency.
    
    fmax : float or 'auto', optional
        Maximum frequency to include. If 'auto', uses the Nyquist frequency.
    
    num_bins : int, optional
        Number of frequency bins.
    
    bin_type : str, optional
        Binning type: 'log' or 'linear'.
    
    bin_edges : array-like, optional
        Custom bin edges (overrides `num_bins` and `bin_type`).
    
    norm : bool, optional
        Whether to normalize the power spectrum to variance units (i.e., PSD units).

    Attributes
    ----------
    freqs : array-like
        Center frequencies of each bin.
    
    freq_widths : array-like
        Bin widths for each frequency bin.
    
    powers : array-like
        Power spectrum values (or mean if using GP samples).
    
    power_errors : array-like
        Uncertainties in the power spectrum (std across GP samples if applicable).
    """

    def __init__(self,
                 lc_or_model,
                 fmin='auto',
                 fmax='auto',
                 num_bins=None,
                 bin_type="log",
                 bin_edges=[],
                 norm=True):
        
        # To do: ValueError for norm=True acting on mean=0 (standardized data)
        input_data = _CheckInputs._check_lightcurve_or_model(lc_or_model)
        if input_data['type'] == 'model':
            self.times, self.rates = input_data['data']
        else:
            self.times, self.rates, _ = input_data['data']
        _CheckInputs._check_input_bins(num_bins, bin_type, bin_edges)

        # Use absolute min and max frequencies if set to 'auto'
        self.dt = np.diff(self.times)[0]
        self.fmin = np.fft.rfftfreq(len(self.rates), d=self.dt)[1] if fmin == 'auto' else fmin
        self.fmax = np.fft.rfftfreq(len(self.rates), d=self.dt)[-1] if fmax == 'auto' else fmax  # nyquist frequency

        self.num_bins = num_bins
        self.bin_type = bin_type
        self.bin_edges = bin_edges

        # if multiple light curve are provided, compute the stacked power spectrum
        if len(self.rates.shape) == 2:
            power_spectrum = self.compute_stacked_power_spectrum(norm=norm)
        else:
            power_spectrum = self.compute_power_spectrum(norm=norm)

        self.freqs, self.freq_widths, self.powers, self.power_errors = power_spectrum

    def compute_power_spectrum(self, times=None, rates=None, norm=True):
        """
        Compute the power spectrum for a single light curve.

        Applies the FFT to the light curve and optionally normalizes the result
        to variance (PSD) units. If binning is enabled, returns binned power.

        Parameters
        ----------
        times : array-like, optional
            Time array to use (defaults to internal value).
        
        rates : array-like, optional
            Rate array to use (defaults to internal value).
        
        norm : bool, optional
            Whether to normalize to variance units.

        Returns
        -------
        freqs : array-like
            Frequencies of the power spectrum.
        
        freq_widths : array-like or None
            Bin widths (if binned).
        
        powers : array-like
            Power spectrum values.
        
        power_errors : array-like or None
            Power spectrum uncertainties (if binned).
        """

        times = self.times if times is None else times
        rates = self.rates if rates is None else rates
        length = len(rates)

        freqs, fft = LightCurve(times=times, rates=rates).fft()
        powers = np.abs(fft) ** 2

        # Filter frequencies within [fmin, fmax]
        valid_mask = (freqs >= self.fmin) & (freqs <= self.fmax)
        freqs = freqs[valid_mask]
        powers = powers[valid_mask]

        if norm:
            powers /= length * np.mean(rates) ** 2 / (2 * self.dt)

        # Apply binning
        if self.num_bins or self.bin_edges:
            
            if self.bin_edges:
                bin_edges = FrequencyBinning.define_bins(self.fmin, self.fmax, num_bins=self.num_bins, 
                                                         bin_type=self.bin_type, bin_edges=self.bin_edges
                                                        )

            elif self.num_bins:
                bin_edges = FrequencyBinning.define_bins(self.fmin, self.fmax, num_bins=self.num_bins, bin_type=self.bin_type)

            else:
                raise ValueError("Either num_bins or bin_edges must be provided.\n"
                                 "In other words, you must specify the number of bins or the bin edges.")

            binned_power = FrequencyBinning.bin_data(freqs, powers, bin_edges)
            freqs, freq_widths, powers, power_errors = binned_power
        else:
            freq_widths, power_errors = None, None

        return freqs, freq_widths, powers, power_errors

    def compute_stacked_power_spectrum(self, norm=True):
        """
        Compute power spectrum for each GP sample and return the mean and std.
        This method is used automatically when a GP model with samples is passed.

        Parameters
        ----------
        norm : bool, optional
            Whether to normalize to variance units.

        Returns
        -------
        freqs : array-like
            Frequencies of the power spectrum.
        
        freq_widths : array-like
            Widths of frequency bins.
        
        power_mean : array-like
            Mean power spectrum values.
        
        power_std : array-like
            Standard deviation of power values across realizations.
        """

        powers = []
        for i in range(self.rates.shape[0]):
            power_spectrum = self.compute_power_spectrum(self.times, self.rates[i], norm=norm)
            freqs, freq_widths, power, _ = power_spectrum
            powers.append(power)

        # Stack the collected powers and errors
        powers = np.vstack(powers)
        power_mean = np.mean(powers, axis=0)
        power_std = np.std(powers, axis=0)

        return freqs, freq_widths, power_mean, power_std

    def plot(self, freqs=None, freq_widths=None, powers=None, power_errors=None, **kwargs):
        """
        Plot the power spectrum.

        Parameters
        ----------
        **kwargs : dict
            Custom plotting options (xlabel, yscale, etc.).
        """

        freqs = self.freqs if freqs is None else freqs
        freq_widths = self.freq_widths if freq_widths is None else freq_widths
        powers = self.powers if powers is None else powers
        power_errors = self.power_errors if power_errors is None else power_errors

        kwargs.setdefault('xlabel', 'Frequency')
        kwargs.setdefault('ylabel', 'Power')
        kwargs.setdefault('xscale', 'log')
        kwargs.setdefault('yscale', 'log')
        Plotter.plot(x=freqs, y=powers, xerr=freq_widths, yerr=power_errors, **kwargs)

    def count_frequencies_in_bins(self, fmin=None, fmax=None, num_bins=None, bin_type=None, bin_edges=[]):
        """
        Counts the number of frequencies in each frequency bin.
        Wrapper method to use FrequencyBinning.count_frequencies_in_bins with class attributes.
        """

        return FrequencyBinning.count_frequencies_in_bins(
            self, fmin=fmin, fmax=fmax, num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges
        )