import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


class TimeSeries:
    def __init__(self,
                 times=[],
                 values=[],
                 errors=[],
                 file_path=None,
                 file_columns=[0,1,2],
                 plot_data=False
                 ):
        # to do: implement snipping, use of the lower functions from arguments
        # need data cleaning somewhere: standardize, nan, trimming, etc.
        """
        Initializes the TimeSeries class.

        Parameters:
        - times (array-like): Array of time values.
        - values (array-like): Array of values or flux values.
        - errors (array-like): Array of errors associated with each value.
        - file_path (str): Path to file containing data (.fits, .csv,
          text files).
        - file_columns (list): List of file_columns for time, value,
          and optionally error. The list should contain
          2 or 3 integers (column index) or strings (column_name):
          [time_column, value_column, error_column (optional)]
        - **kwargs: Additional keyword arguments.
            - hdu (int): HDU index for reading FITS files.
            - delimiter (str): Column delimiter for text files.
        """
        if file_path:
            if not (2 <= len(file_columns) <= 3):
                raise ValueError(
                    "The 'file_columns' parameter must be a list with 2 or 3 items: "
                    "[time_column, value_column, optional error_column]."
                )

            file_data = self.load_file(file_path, file_columns=file_columns)
            self.times = file_data[0]
            self.values = file_data[1]
            self.errors = file_data[2]

        elif times.size > 0 and values.size > 0:
            self.times = np.array(times)
            self.values = np.array(values)
            self.errors = np.array(errors)

        else:
            raise ValueError(
                "Please provide time and value arrays or a file path."
            )
        
        self.mean = np.mean(self.values)
        self.std = np.std(self.values)
        
        if plot_data:
            self.plot()

    def load_file(self, file_path, file_columns=[0,1,2]):
        """
        Loads data from a file and sets the times, values, and errors.
        Supports .fits and text-based files.
        """
        file_extension = file_path.split('.')[-1].lower()

        if file_extension == 'fits':
            times, values, errors = self.load_fits(file_path, file_columns)

        else:
            try:
                times, values, errors = self.load_text_file(file_path, file_columns)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read the file '{file_path}' with load_text_file. Error message: {e}"
                    )

        return times, values, errors
    
    def load_fits(self, file_path, file_columns=[0,1,2], hdu=1):
        """Loads data from a FITS file."""
        time_column, value_column = file_columns[0], file_columns[1]
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
                )

                values = np.array(
                    data.field(value_column) if isinstance(value_column, int)
                    else data[value_column]
                )

                if error_column:
                    errors = np.array(
                        data.field(error_column) if isinstance(error_column, int)
                        else data[error_column]
                    )
                else:
                    errors = []

            except KeyError:
                raise ValueError(
                    "Specified column/s not found in the FITS file."
                )
            
        return times, values, errors

    def load_text_file(self, file_path, file_columns=[0,1,2], delimiter=None):
        time_column, value_column = file_columns[0], file_columns[1]
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
            raise(f"Failed to read the file '{file_path}' with np.genfromtxt.")

        # Retrieve file_columns by name or index directly, simplifying access
        times = np.array(
            data[time_column] if isinstance(time_column, str)
            else data[:, time_column]
        ).astype(float)

        values = np.array(
            data[value_column] if isinstance(value_column, str)
            else data[:, value_column]
        ).astype(float)

        if error_column:
            errors = np.array(
                data[error_column] if isinstance(error_column, str)
                else data[:, error_column]
            ).astype(float)

        else:
            errors = []

        return times, values, errors

    def standardize(self):
        """Standardizes the time series data."""
        self.values = (self.values - self.mean) / self.std

        if self.errors.size > 0:
            self.errors = self.errors / self.std

    def unstandardize(self):
        """Unstandardizes the time series data."""
        self.values = (self.values * self.std) + self.mean

        if self.errors.size > 0:
            self.errors = self.errors * self.std

    def trim(self, start_time, end_time):
        """Trims the time series data to a specific time range."""
        mask = (self.times >= start_time) & (self.times <= end_time)
        self.times = self.times[mask]
        self.values = self.values[mask]

        if self.errors.size > 0:
            self.errors = self.errors[mask]

    def plot(self):
        """Plots the time series with error bars if errors are provided."""
        plt.figure(figsize=(8, 4))

        if self.errors.size > 0:
            plt.errorbar(self.times, self.values, yerr=self.errors, fmt='o', color='black', ms=2, lw=1)

        else:
            plt.scatter(self.times, self.values, color='black', s=2)

        plt.xlabel("Time")
        plt.ylabel("Values")
        plt.show()

    def __add__(self, other_timeseries):
        """Adds two TimeSeries objects with matching times."""
        if not isinstance(other_timeseries, TimeSeries):
            raise TypeError(
                "Both time series must be an instance of the TimeSeries class."
            )

        if not np.array_equal(self.times, other_timeseries.times):
            raise ValueError("Time arrays do not match.")

        new_values = self.values + other_timeseries.values
        if self.errors.size == 0 or other_timeseries.errors.size == 0:
            new_errors = []

        else:
            new_errors = np.sqrt(self.errors**2 + other_timeseries.errors**2)

        return TimeSeries(times=self.times,
                          values=new_values,
                          errors=new_errors)

    def __sub__(self, other_timeseries):
        """Subtracts two TimeSeries objects with matching times."""
        if not isinstance(other_timeseries, TimeSeries):
            raise TypeError(
                "Both time series must be an instance of the TimeSeries class."
            )

        if not np.array_equal(self.times, other_timeseries.times):
            raise ValueError("Time arrays do not match.")

        new_values = self.values - other_timeseries.values
        if self.errors.size == 0 or other_timeseries.errors.size == 0:
            new_errors = []

        else:
            new_errors = np.sqrt(self.errors**2 + other_timeseries.errors**2)

        return TimeSeries(times=self.times,
                          values=new_values,
                          errors=new_errors)

    def __truediv__(self, other_timeseries):
        """Divides two TimeSeries objects with matching times."""
        if not isinstance(other_timeseries, TimeSeries):
            raise TypeError(
                "Both time series must be an instance of the TimeSeries class."
            )

        if not np.array_equal(self.times, other_timeseries.times):
            raise ValueError("Time arrays do not match.")

        new_values = self.values / other_timeseries.values
        if self.errors.size == 0 or other_timeseries.errors.size == 0:
            new_errors = []

        else:
            new_errors = np.sqrt(
                (self.errors / self.values) ** 2 +
                (other_timeseries.errors / other_timeseries.values) ** 2
            )

        return TimeSeries(times=self.times,
                          values=new_values,
                          errors=new_errors)
    