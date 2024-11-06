import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


class TimeSeries:
    def __init__(self,
                 times=None,
                 values=None,
                 errors=None,
                 file_path=None,
                 columns=None
                 ):
        
        """
        Initializes the TimeSeries class.

        Parameters:
        - times (array-like): Array of time values.
        - values (array-like): Array of values or flux values.
        - errors (array-like): Array of errors associated with each value.
        - file_path (str): Path to file containing data (.fits, .csv,
          text files).
        - columns (list): List of columns for time, value,
          and optionally error. The list should contain
          2 or 3 integers (column index) or strings (column_name):
          [time_column, value_column, error_column (optional)]
        """
        if file_path:
            if not isinstance(columns, list) or not (2 <= len(columns) <= 3):
                raise ValueError(
                    "The 'columns' parameter must be a list with 2 or 3 items: "
                    "[time_column, value_column, optional error_column]."
                )
            
            self._load_file(file_path, columns)

        elif times is not None and values is not None:
            # Use provided arrays for initialization
            self.times = np.array(times)
            self.values = np.array(values)
            self.errors = np.array(errors) if errors is not None else None

        else:
            raise ValueError(
                "Please provide time and value arrays or a file path."
                )

    def _load_file(self, file_path, columns):
        """
        Loads data from a file and sets the times, values, and errors.
        Supports .fits, .csv, .dat, .txt files.
        """
        file_extension = file_path.split('.')[-1].lower()

        if file_extension == 'fits':
            self._load_fits(file_path, columns)

        elif file_extension in ['csv', 'dat']:
            self._load_csv_dat_txt(file_path, columns)

        else:
            raise ValueError(
                "Unsupported file format: .fits, .csv, text-format files "
                "supported."
            )

    def _load_fits(self, file_path, columns, hdu=1):
        """Loads data from a FITS file."""
        time_column, value_column = columns[0], columns[1]
        error_column = columns[2] if len(columns) == 3 else None

        with fits.open(file_path) as hdul:
            try:
                data = hdul[hdu].data
            except IndexError:
                raise ValueError(f"HDU {hdu} does not exist in the FITS file.")
            
            try:
                self.times = np.array(
                    data.field(time_column) if isinstance(time_column, int)
                    else data[time_column]
                    )
                
                self.values = np.array(
                    data.field(value_column) if isinstance(value_column, int)
                    else data[value_column]
                    )
                
                if error_column is not None:
                    self.errors = np.array(
                        data.field(error_column) if isinstance(error_column, int)
                        else data[error_column]
                        )
                else:
                    self.errors = None

            except KeyError:
                raise ValueError(
                    "Specified column/s not found in the FITS file."
                    )

    def _load_csv_or_dat(self, file_path, columns, delimiter=None):
        """Loads data from a CSV or DAT file."""
        time_column, value_column = columns[0], columns[1]
        error_column = columns[2] if len(columns) == 3 else None

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

        # Retrieve columns by name or index directly, simplifying access
        self.times = np.array(
            data[time_column] if isinstance(time_column, str)
            else data[:, time_column]
            ).astype(float)
        
        self.values = np.array(
            data[value_column] if isinstance(value_column, str)
            else data[:, value_column]
            ).astype(float)

        if error_column is not None:
            self.errors = np.array(
                data[error_column] if isinstance(error_column, str)
                else data[:, error_column]
                ).astype(float)  
                     
        else:
            self.errors = None

    def trim(self, start_time, end_time):
        """Trims the time series data to a specific time range."""
        mask = (self.times >= start_time) & (self.times <= end_time)
        self.times = self.times[mask]
        self.values = self.values[mask]

        if self.errors is not None:
            self.errors = self.errors[mask]

        print(f"Data trimmed to range: {start_time} to {end_time}")

    def plot(self):
        """Plots the time series with error bars if errors are provided."""
        plt.figure(figsize=(10, 6))

        if self.errors is not None:
            plt.errorbar(self.times, self.values, yerr=self.errors)

        else:
            plt.scatter(self.times, self.values)

        plt.xlabel("Time")
        plt.ylabel("Values")
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
        
        new_values = self.values + other_timeseries.values
        if self.errors is not None and other_timeseries.errors is not None:
            new_errors = np.sqrt(self.errors**2 + other_timeseries.errors**2)

        else:
            new_errors = None

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
        if self.errors is not None and other_timeseries.errors is not None:
            new_errors = np.sqrt(self.errors**2 + other_timeseries.errors**2)

        else:
            new_errors = None

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
        if self.errors is not None and other_timeseries.errors is not None:
            new_errors = np.sqrt(
                (self.errors / self.values) ** 2 +
                (other_timeseries.errors / other_timeseries.values) ** 2
            )

        else:
            new_errors = None

        return TimeSeries(times=self.times,
                          values=new_values,
                          errors=new_errors)