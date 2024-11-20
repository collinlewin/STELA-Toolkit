import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from .data_loader import TimeSeries


class TimeSeries:
    def __init__(self,
                 times=[],
                 values=[],
                 errors=[],
                 file_path=None,
                 file_columns=[0, 1, 2],
                 remove_nans=True,
                 detect_outliers=False,
                 outlier_threshold=1.5,
                 outlier_rolling_window=10,
                 plot_data=False,
                 verbose=True
                 ):
        # To do: Improve commenting and docstrings, check lengths of input arrays
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
        
        if len(self.times) != len(self.values):
            raise ValueError("times and values arrays must have the same length.")
        
        if remove_nans:
            self.remove_nans(verbose=verbose)

        if detect_outliers:
            self.remove_outliers(threshold=outlier_threshold,
                                 rolling_window=outlier_rolling_window,
                                 verbose=verbose)

        if plot_data:
            self.plot_timeseries()

    @property
    def mean(self):
        return np.mean(self.values)
    
    @property
    def std(self):
        return np.std(self.values)
    
    def load_file(self, file_path, file_columns=[0, 1, 2]):
        """
        Loads data from a file and sets the times, values, and errors.
        Supports .fits and text-based files.
        """
        try:
            times, values, errors = self.load_fits(file_path, file_columns)

        except:
            try:
                times, values, errors = self.load_text_file(file_path, file_columns)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read the file '{file_path}' with fits or text-based loader."
                    "Verify the file path and file_columns, or file format unsupported."
                    f"Error message: {e}"
                )

        return times, values, errors

    def load_fits(self, file_path, file_columns=[0, 1, 2], hdu=1):
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
                ).astype(float)

                values = np.array(
                    data.field(value_column) if isinstance(value_column, int)
                    else data[value_column]
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

        return times, values, errors

    def load_text_file(self, file_path, file_columns=[0, 1, 2], delimiter=None):
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

        if error_column:
            errors = np.array(
                data[error_column]if isinstance(error_column, str)
                else data[:, error_column]
            ).astype(float)

        else:
            errors = []

        return times, values, errors

    def plot(self, **kwargs):
        """
        Plots the time series data.

        Keyword arguments:
        - figsize (tuple): Figure size (width, height).
        - title (str): Title of the plot.
        - xlabel (str): Label for the x-axis.
        - ylabel (str): Label for the y-axis.
        - xlim (tuple): Limits for the x-axis.
        - ylim (tuple): Limits for the y-axis.
        - fig_kwargs (dict): Additional keyword arguments for the figure function.
        - plot_kwargs (dict): Additional keyword arguments for the plot function.
        - major_tick_kwargs (dict): Additional keyword arguments for the tick_params function.
        - minor_tick_kwargs (dict): Additional keyword arguments for the tick_params function.
        """
        title = kwargs.get('title', None)

        # Default plotting settings
        if self.errors.size > 0:
            default_plot_kwargs = {'color': 'black', 'fmt': 'o', 'ms': 2, 'lw': 1, 'label': None}
        else:
            default_plot_kwargs = {'color': 'black', 's': 2, 'label': None}

        figsize = kwargs.get('figsize', (8, 4))
        fig_kwargs = {'figsize': figsize, **kwargs.pop('fig_kwargs', {})}
        plot_kwargs = {**default_plot_kwargs, **kwargs.pop('plot_kwargs', {})}
        major_tick_kwargs = {'which': 'major', **kwargs.pop('major_tick_kwargs', {})}
        minor_tick_kwargs = {'which': 'minor', **kwargs.pop('minor_tick_kwargs', {})}

        plt.figure(**fig_kwargs)
        if self.errors.size > 0:
            plt.errorbar(self.times, self.values, yerr=self.errors, **plot_kwargs)
        else:
            plt.scatter(self.times, self.values, **plot_kwargs)

        # Set labels and title
        plt.xlabel(kwargs.get('xlabel', 'Time'))
        plt.ylabel(kwargs.get('ylabel', 'Values'))
        plt.xlim(kwargs.get('xlim', None))
        plt.ylim(kwargs.get('ylim', None))

        # Show legend if label is provided
        if plot_kwargs['label'] is not None:
            plt.legend()

        if title is not None:
            plt.title(title)

        plt.tick_params(**major_tick_kwargs)
        if len(minor_tick_kwargs) > 1:
            plt.minorticks_on()
            plt.tick_params(**minor_tick_kwargs)

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
                          errors=new_errors
                          )

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
                (self.errors / self.values) ** 2
                + (other_timeseries.errors / other_timeseries.values) ** 2
            )

        return TimeSeries(times=self.times,
                          values=new_values,
                          errors=new_errors)
