import numpy as np
from astropy.io import fits
from .plot import Plotter
from ._check_inputs import _CheckInputs


class LightCurve:
    """
    Container for light curve data, including time, rate, and optional error arrays.

    This class is the standard format for handling time series in the STELA Toolkit.
    Light curves can be initialized directly from arrays or loaded from supported file formats
    (FITS, CSV, or plain text). Many analysis modules assume regular time sampling, which is 
    enforced or checked as needed.

    Supports basic arithmetic operations (addition, subtraction, division) with other LightCurve 
    objects and provides utilities for plotting and computing Fourier transforms.

    Parameters
    ----------
    times : array-like, optional
        Array of time values.
    
    rates : array-like, optional
        Array of measured rates (e.g., flux, count rate).
    
    errors : array-like, optional
        Array of uncertainties on the rates. Optional but recommended.
    
    file_path : str, optional
        Path to a file to load light curve data from. Supports FITS and text formats.
    
    file_columns : list of int or str, optional
        List specifying the columns to read as [time, rate, error]. Column names or indices allowed.

    Attributes
    ----------
    times : ndarray
        Array of time values.
    
    rates : ndarray
        Array of rate values.
    
    errors : ndarray
        Array of errors, if provided.
    """
    
    def __init__(self,
                 times=[],
                 rates=[],
                 errors=[],
                 file_path=None,
                 file_columns=[0, 1, 2]):
        
        if file_path:
            if not (2 <= len(file_columns) <= 3):
                raise ValueError(
                    "The 'file_columns' parameter must be a list with 2 or 3 items: "
                    "[time_column, rate_column, optional error_column]."
                )

            file_data = self.load_file(file_path, file_columns=file_columns)
            times, rates, errors = file_data

        elif len(times) > 0 and len(rates) > 0:
            pass

        else:
            raise ValueError(
                "Please provide time and rate arrays or a file path."
            )

        self.times, self.rates, self.errors = _CheckInputs._check_input_data(lightcurve=None,
                                                                             times=times,
                                                                             rates=rates,
                                                                             errors=errors,
                                                                             req_reg_samp=False
                                                                             )

    @property
    def mean(self):
        """
        Return the mean of the light curve rates.
        """
        return np.mean(self.rates)

    @property
    def std(self):
        """
        Return the standard deviation of the light curve rates.
        """
        return np.std(self.rates)

    def load_file(self, file_path, file_columns=[0, 1, 2]):
        """
        Load light curve data from a FITS or text-based file.

        Parameters
        ----------
        file_path : str
            Path to the input file.

        file_columns : list of int or str, optional
            Column indices or names to use for [time, rate, error].
            Defaults to [0, 1, 2].

        Returns
        -------
        times : ndarray
            Array of time values.

        rates : ndarray
            Array of flux or count rate values.

        errors : ndarray
            Array of measurement uncertainties.
        """

        try:
            times, rates, errors = self.load_fits(file_path, file_columns)

        except:
            try:
                times, rates, errors = self.load_text_file(file_path, file_columns)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read the file '{file_path}' with fits or text-based loader.\n"
                    "Verify the file path and file_columns, or file format unsupported.\n"
                    f"Error message: {e}"
                )

        return times, rates, errors

    def load_fits(self, file_path, file_columns=[0, 1, 2], hdu=1):
        """
        Load light curve data from a specified HDU of a FITS file.

        Parameters
        ----------
        file_path : str
            Path to the FITS file.

        file_columns : list of int or str
            Columns to extract, specified as [time, rate, error].

        hdu : int
            Index of the Header/Data Unit (HDU) to read from.

        Returns
        -------
        times : ndarray
            Array of time values.

        rates : ndarray
            Array of flux or count rate values.

        errors : ndarray
            Array of measurement uncertainties.
        """

        time_column, rate_column = file_columns[0], file_columns[1]
        error_column = file_columns[2] if len(file_columns) == 3 else None

        with fits.open(file_path) as hdul:
            try:
                data = hdul[hdu].data
            except IndexError:
                raise ValueError(f"HDU {hdu} does not exist in the FITS file.")

            try:
                times = np.array(
                    data.field(time_column) if isinstance(time_column, int)
                    else data[time_column]
                ).astype(float)

                rates = np.array(
                    data.field(rate_column) if isinstance(rate_column, int)
                    else data[rate_column]
                ).astype(float)

                if error_column:
                    errors = np.array(
                        data.field(error_column) if isinstance(error_column, int)
                        else data[error_column]
                    ).astype(float)
                else:
                    errors = []

            except KeyError:
                raise ValueError(
                    "Specified column/s not found in the FITS file."
                )

        return times, rates, errors

    def load_text_file(self, file_path, file_columns=[0, 1, 2], delimiter=None):
        """
        Load light curve data from a CSV or plain-text file.

        Parameters
        ----------
        file_path : str
            Path to the input text file.

        file_columns : list of int or str
            Columns to extract, specified as [time, rate, error].

        delimiter : str, optional
            Delimiter used in the file (e.g., ',' for CSV, '\\t' for tab-separated).
            If not provided, the delimiter is inferred from the file extension.

        Returns
        -------
        times : ndarray
            Array of time values.

        rates : ndarray
            Array of flux or count rate values.

        errors : ndarray
            Array of measurement uncertainties.
        """

        time_column, rate_column = file_columns[0], file_columns[1]
        error_column = file_columns[2] if len(file_columns) == 3 else None

        # Load data, assuming delimiter based on file extension if unspecified
        if delimiter is None:
            delimiter = ',' if file_path.endswith('.csv') else None

        try:
            data = np.genfromtxt(
                file_path,
                delimiter=delimiter,
            )

        except Exception as e:
            raise (f"Failed to read the file '{file_path}' with np.genfromtxt.")

        # Retrieve file_columns by name or index directly, simplifying access
        times = np.array(
            data[time_column] if isinstance(time_column, str)
            else data[:, time_column]
        ).astype(float)

        rates = np.array(
            data[rate_column] if isinstance(rate_column, str)
            else data[:, rate_column]
        ).astype(float)

        if error_column:
            errors = np.array(
                data[error_column]if isinstance(error_column, str)
                else data[:, error_column]
            ).astype(float)

        else:
            errors = []

        return times, rates, errors

    def plot(self, **kwargs):
        """
        Plot the light curve.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for customizing the plot.
        """

        kwargs.setdefault('xlabel', 'Time')
        kwargs.setdefault('ylabel', 'Rate')
        Plotter.plot(x=self.times, y=self.rates, yerr=self.errors, **kwargs)

    def fft(self):
        """
        Compute the Fast Fourier Transform (FFT) of the light curve.

        Returns
        -------
        freqs : ndarray
            Frequencies of the FFT.
        
        fft_values : ndarray
            Complex FFT values.

        Raises
        ------
        ValueError
            If the time sampling is not uniform. Interpolation is required before applying FFT.
        """

        time_diffs = np.round(np.diff(self.times), 10)
        if np.unique(time_diffs).size > 1:
            raise ValueError("Light curve must have a uniform sampling interval.\n"
                             "Interpolate the data to a uniform grid first."
                             )
        dt = np.diff(self.times)[0]
        length = len(self.rates)

        fft_values = np.fft.rfft(self.rates)
        freqs = np.fft.rfftfreq(length, d=dt)

        return freqs, fft_values

    def __add__(self, other_lightcurve):
        """
        Add two LightCurve objects element-wise.

        Returns
        -------
        LightCurve
            New LightCurve with summed rates and propagated uncertainties.
        """

        if not isinstance(other_lightcurve, LightCurve):
            raise TypeError(
                "Both light curve must be an instance of the LightCurve class."
            )

        if not np.array_equal(self.times, other_lightcurve.times):
            raise ValueError("Time arrays do not match.")

        new_rates = self.rates + other_lightcurve.rates
        if self.errors.size == 0 or other_lightcurve.errors.size == 0:
            new_errors = []

        else:
            new_errors = np.sqrt(self.errors**2 + other_lightcurve.errors**2)

        return LightCurve(times=self.times,
                          rates=new_rates,
                          errors=new_errors)

    def __sub__(self, other_lightcurve):
        """
        Subtract one LightCurve from another element-wise.

        Returns
        -------
        LightCurve
            New LightCurve with difference of rates and propagated uncertainties.
        """

        if not isinstance(other_lightcurve, LightCurve):
            raise TypeError(
                "Both light curve must be an instance of the LightCurve class."
            )

        if not np.array_equal(self.times, other_lightcurve.times):
            raise ValueError("Time arrays do not match.")

        new_rates = self.rates - other_lightcurve.rates
        if self.errors.size == 0 or other_lightcurve.errors.size == 0:
            new_errors = []

        else:
            new_errors = np.sqrt(self.errors**2 + other_lightcurve.errors**2)

        return LightCurve(times=self.times,
                          rates=new_rates,
                          errors=new_errors
                          )

    def __truediv__(self, other_lightcurve):
        """
        Divide one LightCurve by another element-wise.

        Returns
        -------
        LightCurve
            New LightCurve with element-wise division and propagated relative uncertainties.
        """

        if not isinstance(other_lightcurve, LightCurve):
            raise TypeError(
                "Both light curve must be an instance of the LightCurve class."
            )

        if not np.array_equal(self.times, other_lightcurve.times):
            raise ValueError("Time arrays do not match.")

        new_rates = self.rates / other_lightcurve.rates
        if self.errors.size == 0 or other_lightcurve.errors.size == 0:
            new_errors = []

        else:
            new_errors = np.sqrt(
                (self.errors / self.rates) ** 2
                + (other_lightcurve.errors / other_lightcurve.rates) ** 2
            )

        return LightCurve(times=self.times,
                          rates=new_rates,
                          errors=new_errors)
