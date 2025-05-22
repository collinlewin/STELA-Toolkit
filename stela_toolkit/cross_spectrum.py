import numpy as np
from ._check_inputs import _CheckInputs
from .plot import Plotter
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
        times1, rates1 : array-like, optional
            Time and rate arrays for the first light curve.
        times2, rates2 : array-like, optional
            Time and rate arrays for the second light curve.
        norm : bool, optional
            Whether to normalize the result to power spectral density units.

        Returns
        -------
        freqs : array-like
            Frequencies of the cross-spectrum.
        freq_widths : array-like
            Widths of frequency bins.
        cross_spectrum : array-like
            Complex cross-spectrum values.
        cross_spectrum_errors : array-like or None
            Uncertainties in the binned cross-spectrum (None if not binned).
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
            cross_spectrum /= norm_factor

            # negative norm factor shifts the phase by pi
            if norm_factor < 0:
                phase = np.angle(cross_spectrum)
                cross_spectrum = np.abs(cross_spectrum) * np.exp(1j * phase)

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
        cross_spectra_mean = np.mean(cross_spectra, axis=0)
        cross_spectra_std = np.std(cross_spectra, axis=0)

        freqs, freq_widths = cross_spectrum[0], cross_spectrum[1]

        return freqs, freq_widths, cross_spectra_mean, cross_spectra_std

    def plot(self, freqs=None, freq_widths=None, cs=None, cs_errors=None, **kwargs):
        """
        Plot the cross-spectrum.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for plot customization.
        """

        freqs = self.freqs if freqs is None else freqs
        freq_widths = self.freq_widths if freq_widths is None else freq_widths
        cs = self.cs if cs is None else cs
        cs_errors = self.cs_errors if cs_errors is None else cs_errors

        kwargs.setdefault('xlabel', 'Frequency')
        kwargs.setdefault('ylabel', 'Cross-Spectrum')
        kwargs.setdefault('xscale', 'log')
        kwargs.setdefault('yscale', 'log')
        Plotter.plot(
            x=freqs, y=cs, xerr=freq_widths, yerr=cs_errors, **kwargs
        )

    def count_frequencies_in_bins(self, fmin=None, fmax=None, num_bins=None, bin_type=None, bin_edges=[]):
        """
        Counts the number of frequencies in each frequency bin.
        Wrapper method to use FrequencyBinning.count_frequencies_in_bins with class attributes.
        """

        return FrequencyBinning.count_frequencies_in_bins(
            self, fmin=fmin, fmax=fmax, num_bins=num_bins, bin_type=bin_type, bin_edges=bin_edges
        )