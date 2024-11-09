import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


class TimeSeries:
    def __init__(self,
                 times=[],
                 rates=[],
                 errors=[],
                 file_path=None,
                 file_columns=[0,1,2],
                 plot=True
                 ):
        # to do: implement snipping, use of the lower functions from arguments
        # need data cleaning somewhere: standardize, nan, trimming, etc.
        """
        Initializes the TimeSeries class.

        Parameters:
        - times (array-like): Array of time rates.
        - rates (array-like): Array of rates or flux rates.
        - errors (array-like): Array of errors associated with each rate.
        - file_path (str): Path to file containing data (.fits, .csv,
          text files).
        - file_columns (list): List of file_columns for time, rate,
          and optionally error. The list should contain
          2 or 3 integers (column index) or strings (column_name):
          [time_column, rate_column, error_column (optional)]
        - **kwargs: Additional keyword arguments.
            - hdu (int): HDU index for reading FITS files.
            - delimiter (str): Column delimiter for text files.
        """
        if file_path:
            if not (2 <= len(file_columns) <= 3):
                raise ValueError(
                    "The 'file_columns' parameter must be a list with 2 or 3 items: "
                    "[time_column, rate_column, optional error_column]."
                )

            file_data = self.load_file(file_path, file_columns)
            self.times = file_data[0]
            self.rates = file_data[1]
            self.errors = file_data[2]

        elif times and rates:
            self.times = np.array(times)
            self.rates = np.array(rates)
            self.errors = np.array(errors) if errors else None

        else:
            raise ValueError(
                "Please provide time and rate arrays or a file path."
            )
        
        self.mean = np.mean(self.rates)
        self.std = np.std(self.rates)
        
        if plot:
            self.plot()

    def load_file(self, file_path, file_columns):
        """
        Loads data from a file and sets the times, rates, and errors.
        Supports .fits and text-based files.
        """
        file_extension = file_path.split('.')[-1].lower()

        if file_extension == 'fits':
            self._load_fits(file_path, file_columns)

        else:
            try:
                self._load_text_file(file_path, file_columns)
            except Exception as e:
                print(f"Failed to read the file '{file_path}' with _load_text_file.")
                print(f"Error message: {e}")
                
            
    def load_fits(self, file_path, file_columns, hdu=1):
        """Loads data from a FITS file."""
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
                )

                rates = np.array(
                    data.field(rate_column) if isinstance(rate_column, int)
                    else data[rate_column]
                )

                if error_column:
                    errors = np.array(
                        data.field(error_column) if isinstance(error_column, int)
                        else data[error_column]
                    )
                else:
                    errors = None

            except KeyError:
                raise ValueError(
                    "Specified column/s not found in the FITS file."
                )
            
        return times, rates, errors

    def load_text_file(self, file_path, file_columns, delimiter=None):
        """Loads data from a CSV or DAT file."""
        time_column, rate_column = file_columns[0], file_columns[1]
        error_column = file_columns[2] if len(file_columns) == 3 else None

        # Check if the file has a header by inspecting the first line
        try:
            with open(file_path, 'r') as file:
                first_line = file.readline()
                has_header = not first_line.strip().replace('.', '').isdigit()

        except FileNotFoundError:
            raise ValueError(f"File {file_path} not found.")

        # Load data, assuming delimiter based on file extension if unspecified
        if delimiter is None:
            delimiter = ',' if file_path.endswith('.csv') else None

        data = np.genfromtxt(
            file_path,
            delimiter=delimiter,
            names=has_header
        )

        # Retrieve file_columns by name or index directly, simplifying access
        self.times = np.array(
            data[time_column] if isinstance(time_column, str)
            else data[:, time_column]
        ).astype(float)

        self.rates = np.array(
            data[rate_column] if isinstance(rate_column, str)
            else data[:, rate_column]
        ).astype(float)

        if error_column:
            self.errors = np.array(
                data[error_column] if isinstance(error_column, str)
                else data[:, error_column]
            ).astype(float)

        else:
            self.errors = None

    def standardize(self):
        """Standardizes the time series data."""
        self.rates = (self.rates - self.mean) / self.std

        if self.errors:
            self.errors = self.errors / self.std

    def unstandardize(self):
        """Unstandardizes the time series data."""
        self.rates = (self.rates * self.std) + self.mean

        if self.errors:
            self.errors = self.errors * self.std

    def trim(self, start_time, end_time):
        """Trims the time series data to a specific time range."""
        mask = (self.times >= start_time) & (self.times <= end_time)
        self.times = self.times[mask]
        self.rates = self.rates[mask]

        if self.errors:
            self.errors = self.errors[mask]

    def plot(self):
        """Plots the time series with error bars if errors are provided."""
        plt.figure(figsize=(10, 6))

        if self.errors:
            plt.errorbar(self.times, self.rates, yerr=self.errors)

        else:
            plt.scatter(self.times, self.rates)

        plt.xlabel("Time")
        plt.ylabel("rates")
        plt.title("Time Series")
        plt.legend()
        plt.show()

    def __add__(self, other_timeseries):
        """Adds two TimeSeries objects with matching times."""
        if not isinstance(other_timeseries, TimeSeries):
            raise TypeError(
                "Both time series must be an instance of the TimeSeries class."
            )

        if not np.array_equal(self.times, other_timeseries.times):
            raise ValueError("Time arrays do not match.")

        new_rates = self.rates + other_timeseries.rates
        if self.errors and other_timeseries.errors:
            new_errors = np.sqrt(self.errors**2 + other_timeseries.errors**2)

        else:
            new_errors = None

        return TimeSeries(times=self.times,
                          rates=new_rates,
                          errors=new_errors)

    def __sub__(self, other_timeseries):
        """Subtracts two TimeSeries objects with matching times."""
        if not isinstance(other_timeseries, TimeSeries):
            raise TypeError(
                "Both time series must be an instance of the TimeSeries class."
            )

        if not np.array_equal(self.times, other_timeseries.times):
            raise ValueError("Time arrays do not match.")

        new_rates = self.rates - other_timeseries.rates
        if self.errors and other_timeseries.errors:
            new_errors = np.sqrt(self.errors**2 + other_timeseries.errors**2)

        else:
            new_errors = None

        return TimeSeries(times=self.times,
                          rates=new_rates,
                          errors=new_errors)

    def __truediv__(self, other_timeseries):
        """Divides two TimeSeries objects with matching times."""
        if not isinstance(other_timeseries, TimeSeries):
            raise TypeError(
                "Both time series must be an instance of the TimeSeries class."
            )

        if not np.array_equal(self.times, other_timeseries.times):
            raise ValueError("Time arrays do not match.")

        new_rates = self.rates / other_timeseries.rates
        if self.errors and other_timeseries.errors:
            new_errors = np.sqrt(
                (self.errors / self.rates) ** 2 +
                (other_timeseries.errors / other_timeseries.rates) ** 2
            )

        else:
            new_errors = None

        return TimeSeries(times=self.times,
                          rates=new_rates,
                          errors=new_errors)
    