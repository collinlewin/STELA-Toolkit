import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from .plot import Plotter
from ._check_inputs import _CheckInputs


class LightCurve:
    def __init__(self,
                 times=[],
                 rates=[],
                 errors=[],
                 file_path=None,
                 file_columns=[0, 1, 2],
                 ):
        # To do: Improve commenting and docstrings, use _check_inputs
        """
        Initializes the LightCurve object.

        The method initializes the time, rate, and error arrays,
        either from provided arrays or by reading from a specified file.

        Parameters:
        - times (array-like): Array of time values.
        - rates (array-like): Array of measured rates (e.g., flux, counts).
        - errors (array-like): Array of uncertainties for the measured rates (optional).
        - file_path (str): Path to a file containing the data (FITS, CSV, or text).
        - file_columns (list): List specifying the columns for time, rate, and error
        (e.g., [time_column, rate_column, optional_error_column]).
        """
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
                                                                             errors=errors
                                                                             )

    @property
    def mean(self):
        return np.mean(self.rates)

    @property
    def std(self):
        return np.std(self.rates)

    def load_file(self, file_path, file_columns=[0, 1, 2]):
        """
        Loads light curve data from a specified file. Supports FITS and text-based files.

        Parameters:
        - file_path (str): Path to the file to load.
        - file_columns (list): List specifying the columns for time, rate, and error.

        Returns:
        - tuple: Arrays of times, rates, and errors.
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
        Loads light curve data from a FITS file, from a specified HDU.

        Parameters:
        - file_path (str): Path to the FITS file to load.
        - file_columns (list): List specifying the columns for time, rate, and error.
        - hdu (int): The HDU index to read from (default is 1).

        Returns:
        - tuple: Arrays of times, rates, and errors.
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
        Loads light curve data from a text-based file. Assumes a delimiter based on
        file extension if none is provided.

        Parameters:
        - file_path (str): Path to the text file to load.
        - file_columns (list): List specifying the columns for time, rate, and error.
        - delimiter (str): Column delimiter (optional).

        Returns:
        - tuple: Arrays of times, rates, and errors.
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
        Plots the light curve data.

        Parameters:
        - **kwargs: Additional keyword arguments for plot customization (e.g., xlabel, ylabel, title).
        """
        kwargs.setdefault('xlabel', 'Time')
        kwargs.setdefault('ylabel', 'Rate')
        Plotter.plot(x=self.times, y=self.rates, yerr=self.errors, **kwargs)

    def fft(self):
        """
        Computes the Fast Fourier Transform (FFT) of the light curve data.

        Returns:
        - freqs (array-like): Frequencies of the FFT.
        - fft_values (array-like): FFT values.
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
        Adds two LightCurve objects.
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
        Subtracts two LightCurve objects.
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
        Divides two LightCurve objects.
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
