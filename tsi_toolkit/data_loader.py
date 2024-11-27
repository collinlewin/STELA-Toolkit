import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from .plot import Plotter


class TimeSeries:
    def __init__(self,
                 times=[],
                 values=[],
                 sigmas=[],
                 file_path=None,
                 file_columns=[0,1,2],
                 ):
        # To do: Improve commenting and docstrings
        """
        Initializes the TimeSeries object.

        The method initializes the time, value, and sigma arrays,
        either from provided arrays or by reading from a specified file.

        Parameters:
        - times (array-like): Array of time values.
        - values (array-like): Array of measured values (e.g., flux, counts).
        - sigmas (array-like): Array of uncertainties for the measured values (optional).
        - file_path (str): Path to a file containing the data (FITS, CSV, or text).
        - file_columns (list): List specifying the columns for time, value, and sigma
        (e.g., [time_column, value_column, optional_sigma_column]).
        """
        if file_path:
            if not (2 <= len(file_columns) <= 3):
                raise ValueError(
                    "The 'file_columns' parameter must be a list with 2 or 3 items: "
                    "[time_column, value_column, optional sigma_column]."
                )

            file_data = self.load_file(file_path, file_columns=file_columns)
            self.times = file_data[0]
            self.values = file_data[1]
            self.sigmas = file_data[2]

        elif times.size > 0 and values.size > 0:
            self.times = np.array(times)
            self.values = np.array(values)
            self.sigmas = np.array(sigmas)
            
        else:
            raise ValueError(
                "Please provide time and value arrays or a file path."
            )
        
        if len(self.times) != len(self.values):
            raise ValueError("Times and values arrays must have the same length.")

    @property
    def mean(self):
        return np.mean(self.values)
    
    @property
    def std(self):
        return np.std(self.values)
    
    def load_file(self, file_path, file_columns=[0, 1, 2]):
        """
        Loads time series data from a specified file. Supports FITS and text-based files.

        Parameters:
        - file_path (str): Path to the file to load.
        - file_columns (list): List specifying the columns for time, value, and sigma.

        Returns:
        - tuple: Arrays of times, values, and sigmas.
        """
        try:
            times, values, sigmas = self.load_fits(file_path, file_columns)

        except:
            try:
                times, values, sigmas = self.load_text_file(file_path, file_columns)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read the file '{file_path}' with fits or text-based loader.\n"
                    "Verify the file path and file_columns, or file format unsupported.\n"
                    f"Error message: {e}"
                )

        return times, values, sigmas

    def load_fits(self, file_path, file_columns=[0, 1, 2], hdu=1):
        """
        Loads time series data from a FITS file, from a specified HDU.

        Parameters:
        - file_path (str): Path to the FITS file to load.
        - file_columns (list): List specifying the columns for time, value, and sigma.
        - hdu (int): The HDU index to read from (default is 1).

        Returns:
        - tuple: Arrays of times, values, and sigmas.
        """
        time_column, value_column = file_columns[0], file_columns[1]
        sigma_column = file_columns[2] if len(file_columns) == 3 else None

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

                values = np.array(
                    data.field(value_column) if isinstance(value_column, int)
                    else data[value_column]
                ).astype(float)

                if sigma_column:
                    sigmas = np.array(
                        data.field(sigma_column) if isinstance(sigma_column, int)
                        else data[sigma_column]
                    ).astype(float)
                else:
                    sigmas = []

            except KeyError:
                raise ValueError(
                    "Specified column/s not found in the FITS file."
                )

        return times, values, sigmas

    def load_text_file(self, file_path, file_columns=[0, 1, 2], delimiter=None):
        """
        Loads time series data from a text-based file. Assumes a delimiter based on
        file extension if none is provided.

        Parameters:
        - file_path (str): Path to the text file to load.
        - file_columns (list): List specifying the columns for time, value, and sigma.
        - delimiter (str): Column delimiter (optional).

        Returns:
        - tuple: Arrays of times, values, and sigmas.
        """
        time_column, value_column = file_columns[0], file_columns[1]
        sigma_column = file_columns[2] if len(file_columns) == 3 else None

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

        values = np.array(
            data[value_column] if isinstance(value_column, str)
            else data[:, value_column]
        ).astype(float)

        if sigma_column:
            sigmas = np.array(
                data[sigma_column]if isinstance(sigma_column, str)
                else data[:, sigma_column]
            ).astype(float)

        else:
            sigmas = []

        return times, values, sigmas

    def plot(self, **kwargs):
        """
        Plots the time series data.

        Parameters:
        - **kwargs: Additional keyword arguments for plot customization (e.g., xlabel, ylabel, title).
        """
        kwargs.setdefault('xlabel', 'Time')
        kwargs.setdefault('ylabel', 'Values')
        Plotter.plot(x=self.times, y=self.values, yerr=self.sigmas, **kwargs)

    def fft(self):
        """
        Computes the Fast Fourier Transform (FFT) of the time series data.

        Returns:
        - freqs (array-like): Frequencies of the FFT.
        - fft_values (array-like): FFT values.
        """
        time_diffs = np.round(np.diff(self.times), 10)
        if np.unique(time_diffs).size > 1:
            raise ValueError("Time series must have a uniform sampling interval.\n"
                            "Interpolate the data to a uniform grid first."
                        )
        dt = np.diff(self.times)[0]
        length = len(self.values)

        fft_values = np.fft.rfft(self.values)
        freqs = np.fft.rfftfreq(length, d=dt)

        return freqs, fft_values
    
    def __add__(self, other_timeseries):
        """
        Adds two TimeSeries objects.
        """
        if not isinstance(other_timeseries, TimeSeries):
            raise TypeError(
                "Both time series must be an instance of the TimeSeries class."
            )

        if not np.array_equal(self.times, other_timeseries.times):
            raise ValueError("Time arrays do not match.")

        new_values = self.values + other_timeseries.values
        if self.sigmas.size == 0 or other_timeseries.sigmas.size == 0:
            new_sigmas = []

        else:
            new_sigmas = np.sqrt(self.sigmas**2 + other_timeseries.sigmas**2)

        return TimeSeries(times=self.times,
                          values=new_values,
                          sigmas=new_sigmas)

    def __sub__(self, other_timeseries):
        """
        Subtracts two TimeSeries objects.
        """
        if not isinstance(other_timeseries, TimeSeries):
            raise TypeError(
                "Both time series must be an instance of the TimeSeries class."
            )

        if not np.array_equal(self.times, other_timeseries.times):
            raise ValueError("Time arrays do not match.")

        new_values = self.values - other_timeseries.values
        if self.sigmas.size == 0 or other_timeseries.sigmas.size == 0:
            new_sigmas = []

        else:
            new_sigmas = np.sqrt(self.sigmas**2 + other_timeseries.sigmas**2)

        return TimeSeries(times=self.times,
                          values=new_values,
                          sigmas=new_sigmas
                          )

    def __truediv__(self, other_timeseries):
        """
        Divides two TimeSeries objects.
        """
        if not isinstance(other_timeseries, TimeSeries):
            raise TypeError(
                "Both time series must be an instance of the TimeSeries class."
            )

        if not np.array_equal(self.times, other_timeseries.times):
            raise ValueError("Time arrays do not match.")

        new_values = self.values / other_timeseries.values
        if self.sigmas.size == 0 or other_timeseries.sigmas.size == 0:
            new_sigmas = []

        else:
            new_sigmas = np.sqrt(
                (self.sigmas / self.values) ** 2
                + (other_timeseries.sigmas / other_timeseries.values) ** 2
            )

        return TimeSeries(times=self.times,
                          values=new_values,
                          sigmas=new_sigmas)
