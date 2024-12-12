import numpy as np


class _CheckInputs:
    """
    A utility class for checking and validating input data and binning.
    """
    @staticmethod
    def _check_input_data(lightcurve, times=[], rates=[], errors=[], req_reg_samp=True):
        """
        Validates and extracts time and rate arrays from input or LightCurve objects.
        """
        if lightcurve:
            if type(lightcurve).__name__ != "LightCurve":
                raise TypeError(
                    "lightcurve must be an instance of the LightCurve class.")

            times = lightcurve.times
            rates = lightcurve.rates
            errors = lightcurve.errors

        # check input arrays if not lightcurve object
        elif len(times) > 0 and len(rates) > 0:
            times = np.array(times)
            rates = np.array(rates)
            errors = np.array(errors)

            if len(rates.shape) == 1 and len(times) != len(rates):
                raise ValueError("Times and rates must have the same length.")

            elif len(rates.shape) == 2:
                if rates.shape[1] != len(times):
                    raise ValueError(
                        "Times and rates must have the same length for each light curve.\n"
                        "Check the shape of the rates array: expecting (n_series, n_times)."
                    )

            if len(errors) > 0:
                if np.min(errors) <= 0:
                    raise ValueError(
                        "Uncertainties of the input data must be positive.")
        else:
            raise ValueError(
                "Either provide a LightCurve object or times and rates arrays.")

        # check for regular sampling
        if req_reg_samp:
            time_sampling = np.round(np.diff(times), 10)
            if np.unique(time_sampling).size > 1:
                raise ValueError("Time series must have a uniform sampling interval.\n"
                                 "Interpolate the data to a uniform grid first."
                                 )

        return times, rates, errors

    @staticmethod
    def _check_input_model(model):
        if type(model).__name__ == "GaussianProcess":
            if hasattr(model, "samples"):
                num_samp = model.samples.shape[0]
                kernel_form = model.kernel_form
                print(
                    f"Detected {num_samp} samples generated using a {kernel_form} kernel.")

                pred_times = model.pred_times
                samples = model.samples
            else:
                print("No samples detected. Generating 1000 samples to use...")
                step = (model.train_times.max()-model.train_times.min())/1000
                pred_times = np.arange(
                    model.train_times.min(), model.train_times.max()+step, step)
                samples = model.sample(pred_times, 1000)
        else:
            raise TypeError(
                "Model must be an instance of the Gaussian Process class.")

        return pred_times, samples

    @staticmethod
    def _check_input_bins(num_bins, bin_type, bin_edges):
        """
        Validates and returns bin edges for frequency binning.
        """
        if len(bin_edges) > 0:
            # Use custom bins
            if np.diff(bin_edges) <= 0:
                raise ValueError(
                    "Custom bin edges must be monotonically increasing.")
            if num_bins is not None:
                print(
                    "Custom bin_edges detected: num_bins is ignored when custom bins are provided.")

        elif num_bins is not None:
            if not isinstance(num_bins, int) or num_bins < 1:
                raise ValueError(
                    "Number of bins (num_bins) must be a positive integer.")
            if bin_type is None:
                raise ValueError(
                    "bin_type must be provided if num_bins is used.")

        if bin_type not in ["log", "linear"]:
            raise ValueError(
                f"Unsupported bin_type '{bin_type}'. Choose 'log', 'linear', or provide custom bins.")
