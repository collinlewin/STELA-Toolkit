import numpy as np


class _CheckInputs:
    """
    Internal utilities for validating and formatting inputs across STELA classes.
    Includes checks for regular sampling, valid model types, and binning logic.
    """

    @staticmethod
    def _check_input_data(lightcurve, times=[], rates=[], errors=[], req_reg_samp=True):
        """
        Validate and extract times, rates, and errors from a LightCurve or from arrays.

        Checks input dimensions and ensures that error values are nonnegative. If
        `req_reg_samp` is True, the method also verifies that the time grid is evenly spaced.

        Parameters
        ----------
        lightcurve : LightCurve or None
            LightCurve object to extract data from. If None, uses `times`, `rates`, and `errors`.

        times : array-like, optional
            Time values (used if no LightCurve is provided).

        rates : array-like, optional
            Rate values corresponding to the time grid.

        errors : array-like, optional
            Measurement uncertainties associated with the rate values.

        req_reg_samp : bool
            Whether to enforce regular (evenly spaced) time sampling.

        Returns
        -------
        times : ndarray
            Validated time values.

        rates : ndarray
            Validated rate values.

        errors : ndarray
            Validated error values (if provided).
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
                if np.min(errors) < 0:
                    raise ValueError(
                        "Uncertainties of the input data must be nonnegative.")
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
        """
        Validate a GaussianProcess model and extract its samples.

        If no samples exist yet, generates 1000 samples over 1000 evenly spaced points
        using the model's training time range.

        Parameters
        ----------
        model : GaussianProcess
            Trained GP model.

        Returns
        -------
        pred_times : ndarray
            Time grid for the samples.
        
        samples : ndarray
            Realizations from the GP posterior.
        """

        # update the list here when adding new models!
        if type(model).__name__ in ["GaussianProcess"]:
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
    def _check_lightcurve_or_model(lightcurve_or_model, req_reg_samp=False):
        """
        Identify whether the input is a LightCurve or GP model and validate accordingly.

        Dispatches to `_check_input_data` or `_check_input_model` and labels the result
        with its type for downstream use.

        Parameters
        ----------
        lightcurve_or_model : LightCurve or GaussianProcess
            Input to classify and validate.

        Returns
        -------
        dict
            Dictionary with keys 'type' (either 'lightcurve' or 'model') and 'data'.
        """

        # update the list here when adding new models!
        if type(lightcurve_or_model).__name__ in ["GaussianProcess"]:
            input_type = 'model'
            return {'type':input_type, 'data':_CheckInputs._check_input_model(lightcurve_or_model)}
        
        elif type(lightcurve_or_model).__name__ == "LightCurve":
            input_type = 'lightcurve'
            return {'type':input_type, 'data':_CheckInputs._check_input_data(lightcurve_or_model, req_reg_samp=req_reg_samp)}
        
    @staticmethod
    def _check_input_bins(num_bins, bin_type, bin_edges):
        """
        Validate user input for frequency binning.

        Ensures consistency between `num_bins`, `bin_type`, and `bin_edges`. Raises
        errors for missing or invalid combinations.

        Parameters
        ----------
        num_bins : int or None
            Number of bins.
        
        bin_type : str or None
            'log' or 'linear'.
        
        bin_edges : array-like
            Custom bin edges.
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
