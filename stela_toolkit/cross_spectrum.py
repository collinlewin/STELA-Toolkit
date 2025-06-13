import numpy as np
from ._check_inputs import _CheckInputs
from .frequency_binning import FrequencyBinning
from .data_loader import LightCurve


class CrossSpectrum:
    """
    Compute the cross-spectrum between two light curves or trained Gaussian Process models.

    This class accepts LightCurve objects or GaussianProcess models from this package.
    For GP models, if posterior samples have already been generated, those are used.
    If not, the class automatically generates 1000 samples across a 1000-point grid.

    The cross-spectrum is computed using the Fourier transform of one time series
    multiplied by the complex conjugate of the other, yielding frequency-dependent phase
    and amplitude information.

    If both inputs are GP models, the cross-spectrum is computed across all sample pairs,
    and the mean and standard deviation across realizations are returned.

    Frequency binning is available with options for logarithmic, linear, or custom spacing.

    Parameters
    ----------
    lc_or_model1 : LightCurve or GaussianProcess
        First input light curve or trained GP model.
    
    lc_or_model2 : LightCurve or GaussianProcess
        Second input light curve or trained GP model.
    
    fmin : float or 'auto', optional
        Minimum frequency to include. If 'auto', uses lowest nonzero FFT frequency.
    
    fmax : float or 'auto', optional
        Maximum frequency to include. If 'auto', uses the Nyquist frequency.
    
    num_bins : int, optional
        Number of frequency bins.
    
    bin_type : str, optional
        Binning type: 'log' or 'linear'.
    
    bin_edges : array-like, optional
        Custom frequency bin edges. Overrides `num_bins` and `bin_type` if provided.
    
    norm : bool, optional
        Whether to normalize the cross-spectrum to variance units (i.e., PSD units).

    Attributes
    ----------
    freqs : array-like
        Frequency bin centers.
    
    freq_widths : array-like
        Frequency bin widths.
    
    cs : array-like
        Complex cross-spectrum values.
    
    cs_errors : array-like
        Uncertainties in the binned cross-spectrum (if stacked).
    """

    def __init__(self,
                 lc_or_model1,
                 lc_or_model2,
                 fmin='auto',
                 fmax='auto',
                 num_bins=None,
                 bin_type="log",
                 bin_edges=[],
                 norm=True):
        
        # To do: update main docstring
        input_data = _CheckInputs._check_lightcurve_or_model(lc_or_model1)
        if input_data['type'] == 'model':
            self.times1, self.rates1 = input_data['data']
        else:
            self.times1, self.rates1, _ = input_data['data']

        input_data = _CheckInputs._check_lightcurve_or_model(lc_or_model2)
        if input_data['type'] == 'model':
            self.times2, self.rates2 = input_data['data']
        else:
            self.times2, self.rates2, _ = input_data['data']

        _CheckInputs._check_input_bins(num_bins, bin_type, bin_edges)

        if not np.allclose(self.times1, self.times2):
            raise ValueError("The time arrays of the two light curves must be identical.")

        # Use absolute min and max frequencies if set to 'auto'
        self.dt = np.diff(self.times1)[0]
        self.fmin = np.fft.rfftfreq(len(self.rates1), d=self.dt)[1] if fmin == 'auto' else fmin
        self.fmax = np.fft.rfftfreq(len(self.rates1), d=self.dt)[-1] if fmax == 'auto' else fmax  # nyquist frequency

        self.num_bins = num_bins
        self.bin_type = bin_type
        self.bin_edges = bin_edges

        # Check if the input rates are for multiple realizations
        # this needs to be corrected for handling different shapes and dim val1 != dim val2
        if len(self.rates1.shape) == 2 and len(self.rates2.shape) == 2:
            cross_spectrum = self.compute_stacked_cross_spectrum(norm=norm)
        else:
            cross_spectrum = self.compute_cross_spectrum(norm=norm)

        self.freqs, self.freq_widths, self.cs, self.cs_errors = cross_spectrum

    def compute_cross_spectrum(self, times1=None, rates1=None, times2=None, rates2=None, norm=True):
        """
        Compute the cross-spectrum for a single pair of light curves.

        Parameters
        ----------
        times1 : array-like, optional
            Time values for the first light curve.

        rates1 : array-like, optional
            Flux or count rate values for the first light curve.

        times2 : array-like, optional
            Time values for the second light curve.

        rates2 : array-like, optional
            Flux or count rate values for the second light curve.

        norm : bool, optional
            Whether to normalize the result to power spectral density units.

        Returns
        -------
        freqs : array-like
            Frequencies at which the cross-spectrum is evaluated.

        freq_widths : array-like
            Widths of frequency bins (for error bars or plotting).

        cross_spectrum : array-like
            Complex cross-spectrum values for each frequency bin.

        cross_spectrum_errors : array-like or None
            Uncertainties in the binned cross-spectrum values. None if unbinned.
        """

        times1 = self.times1 if times1 is None else times1
        rates1 = self.rates1 if rates1 is None else rates1
        times2 = self.times2 if times2 is None else times2
        rates2 = self.rates2 if rates2 is None else rates2

        freqs, fft1 = LightCurve(times=times1, rates=rates1).fft()
        _, fft2 = LightCurve(times=times2, rates=rates2).fft()

        cross_spectrum = np.conj(fft1) * fft2

        # Filter frequencies within [fmin, fmax]
        valid_mask = (freqs >= self.fmin) & (freqs <= self.fmax)
        freqs = freqs[valid_mask]
        cross_spectrum = cross_spectrum[valid_mask]

        # Normalize power spectrum to units of variance (PSD)
        if norm:
            length = len(rates1)
            norm_factor = length * np.mean(rates1) * np.mean(rates2) / (2 * self.dt)
            cross_spectrum /= np.abs(norm_factor)

        # Apply binning
        if self.num_bins or self.bin_edges:
            if self.bin_edges:
                # use custom bin edges
                bin_edges = FrequencyBinning.define_bins(
                    self.fmin, self.fmax, num_bins=self.num_bins,
                    bin_type=self.bin_type, bin_edges=self.bin_edges
                )
            elif self.num_bins:
                # use equal-width bins in log or linear space
                bin_edges = FrequencyBinning.define_bins(
                    self.fmin, self.fmax, num_bins=self.num_bins, bin_type=self.bin_type
                )
            else:
                raise ValueError("Either num_bins or bin_edges must be provided.\n"
                                 "In other words, you must specify the number of bins or the bin edges.")

            binned_cross_spectrum = FrequencyBinning.bin_data(freqs, cross_spectrum, bin_edges)
            freqs, freq_widths, cross_spectrum, cross_spectrum_errors = binned_cross_spectrum
        else:
            freq_widths, cross_spectrum_errors = None, None

        return freqs, freq_widths, cross_spectrum, cross_spectrum_errors

    def compute_stacked_cross_spectrum(self, norm=True):
        """
        Compute the cross-spectrum across stacked GP samples.

        Computes the cross-spectrum for each realization and returns the mean and
        standard deviation across samples.

        Parameters
        ----------
        norm : bool, optional
            Whether to normalize the result to power spectral density units.

        Returns
        -------
        freqs : array-like
            Frequencies of the cross-spectrum.
        
        freq_widths : array-like
            Widths of frequency bins.
        
        cross_spectra_mean : array-like
            Mean cross-spectrum across GP samples.
        
        cross_spectra_std : array-like
            Standard deviation of the cross-spectrum across samples.
        """

        cross_spectra = []
        for i in range(self.rates1.shape[0]):
            cross_spectrum = self.compute_cross_spectrum(
                times1=self.times1, rates1=self.rates1[i],
                times2=self.times2, rates2=self.rates2[i],
                norm=norm
            )
            cross_spectra.append(cross_spectrum[2])

        cross_spectra = np.vstack(cross_spectra)
        # Real and imaginary std devs
        cs_real_mean = np.mean(cross_spectra.real, axis=0)
        cs_imag_mean = np.mean(cross_spectra.imag, axis=0)
        cs_real_std = np.std(cross_spectra.real, axis=0)
        cs_imag_std = np.std(cross_spectra.imag, axis=0)

        cross_spectra_mean = cs_real_mean + 1j * cs_imag_mean
        cross_spectra_std = cs_real_std + 1j * cs_imag_std
        freqs, freq_widths = cross_spectrum[0], cross_spectrum[1]

        return freqs, freq_widths, cross_spectra_mean, cross_spectra_std

    def plot(self, freqs=None, freq_widths=None, cs=None, cs_errors=None, **kwargs):
        """
        Plot the real and imaginary parts of the cross-spectrum.

        Parameters
        ----------
        freqs : array-like, optional
            Frequencies at which the cross-spectrum is evaluated.
        
        freq_widths : array-like, optional
            Widths of the frequency bins.
        
        cs : array-like, optional
            Cross-spectrum values.
       
        cs_errors : array-like, optional
            Uncertainties in the cross-spectrum.
        
        **kwargs : dict
            Additional keyword arguments for plot customization.
        """
        import matplotlib.pyplot as plt

        freqs = self.freqs if freqs is None else freqs
        freq_widths = self.freq_widths if freq_widths is None else freq_widths
        cs = self.cs if cs is None else cs
        cs_errors = self.cs_errors if cs_errors is None else cs_errors

        figsize = kwargs.get('figsize', (8, 4.5))
        xlabel = kwargs.get('xlabel', 'Frequency')
        ylabel = kwargs.get('ylabel', 'Cross-Spectrum')
        xscale = kwargs.get('xscale', 'log')
        yscale = kwargs.get('yscale', 'log')

        plt.figure(figsize=figsize)

        # Real part
        if cs_errors is not None:
            plt.errorbar(freqs, cs.real, xerr=freq_widths, yerr=cs_errors.real,
                         fmt='o', color='black', ms=3, lw=1.5, label='Real')
        else:
            plt.plot(freqs, cs.real, 'o-', color='black', ms=3, lw=1.5, label='Real')

        # Imaginary part
        if cs_errors is not None:
            plt.errorbar(freqs, cs.imag, xerr=freq_widths, yerr=cs_errors.imag,
                         fmt='o', color='red', ms=3, lw=1.5, label='Imag')
        else:
            plt.plot(freqs, cs.imag, 'o-', color='red', ms=3, lw=1.5, label='Imag')

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tick_params(which='both', direction='in', length=6, width=1,
                        top=True, right=True, labelsize=12)
        plt.show()

    def count_frequencies_in_bins(self, fmin=None, fmax=None, num_bins=None, bin_type=None, bin_edges=[]):
        """
        Counts the number of frequencies in each frequency bin.
        Wrapper method to use FrequencyBinning.count_frequencies_in_bins with class attributes.
        """

        return FrequencyBinning.count_frequencies_in_bins(
            self, fmin=fmin, fmax=fmax, num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges
        )