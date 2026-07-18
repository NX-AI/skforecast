################################################################################
#                         Foundation Model Adapters                            #
#                                                                              #
# This work by skforecast team is licensed under the BSD 3-Clause License.     #
################################################################################
# coding=utf-8
# Each adapter imports its own backend library lazily (i.e. inside the method
# that first needs it) rather than at module level. This means that only the
# library required by the adapter you actually use needs to be installed, other
# foundation-model backends remain optional.

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import warnings


def _resolve_torch_device(device: str) -> str:
    """
    Resolve a device string to a concrete PyTorch device name.

    If `device` is `"auto"`, the best available accelerator is selected
    in priority order: CUDA > MPS (Apple Silicon) > CPU.

    Parameters
    ----------
    device : str
        Device string. Use `"auto"` for automatic selection, or an
        explicit name such as `"cuda"`, `"mps"`, or `"cpu"`.

    Returns
    -------
    device : str
        Resolved device name.

    """

    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ChronosAdapter:
    """
    Adapter for Amazon Chronos foundation models.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. "autogluon/chronos-2-small".
    pipeline : BaseChronosPipeline, default None
        Pre-loaded pipeline instance. If `None`, the pipeline is loaded
        lazily on the first call to `predict`.
    context_length : int, default 8192
        Maximum number of historical observations to use as context. At fit
        time only the last `context_length` observations are stored. At
        predict time, if `context` is longer than `context_length` it is
        trimmed to this length; if it is shorter, all available observations
        are used as-is. Defaults to 8192, which matches the maximum context
        window of Chronos. Must be a positive integer.
    predict_kwargs : dict, default None
        Additional keyword arguments forwarded to the pipeline's
        `predict_quantiles` method.
    device_map : str, default 'auto'
        Device placement for the model. `"auto"` selects the best
        available accelerator (CUDA > MPS > CPU). Also accepts explicit
        values such as `"cuda"`, `"mps"`, or `"cpu"`, forwarded to
        `BaseChronosPipeline.from_pretrained`.
    torch_dtype : object, default None
        Torch dtype forwarded to `BaseChronosPipeline.from_pretrained`.
    cross_learning : bool, default False
        If `True`, Chronos shares information across all series in
        the batch when predicting in multi-series mode. Forwarded
        directly to `predict_quantiles`. Ignored in single-series mode.

    Attributes
    ----------
    model_id : str
        HuggingFace model ID.
    context_ : dict
        Stored training series after fitting.
    context_exog_ : dict
        Stored historical exogenous variables after fitting.
    context_length : int
        Maximum number of historical observations used as context.
    predict_kwargs : dict
        Additional keyword arguments forwarded to `predict_quantiles`.
    device_map : str
        Device map string for model loading.
    torch_dtype : object
        Torch dtype for model loading.
    cross_learning : bool
        Whether cross-series learning is enabled.
    is_fitted : bool
        Whether the adapter has been fitted.

    References
    ----------
    .. [1] https://github.com/amazon-science/chronos-forecasting
    
    .. [2] https://huggingface.co/amazon/chronos-2

    """

    allow_exog: bool = True

    def __init__(
        self,
        model_id: str,
        *,
        pipeline: Any | None = None,
        context_length: int = 8192,
        predict_kwargs: dict[str, Any] | None = None,
        device_map: str = "auto",
        torch_dtype: Any | None = None,
        cross_learning: bool = False,
    ) -> None:
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID, e.g. "autogluon/chronos-2-small".
        pipeline : BaseChronosPipeline, default None
            Pre-loaded pipeline instance. If `None`, the pipeline is
            loaded lazily on the first call to `predict`.
        context_length : int, default 8192
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations of
            `series` (and `exog`) are stored. At `predict` time, if
            `context` is longer than `context_length` it is trimmed to
            this length before inference; if it is shorter, all available
            observations are passed as-is and the model handles reduced
            context gracefully. Defaults to 8192, which matches the
            maximum context window of Chronos. Must be a positive
            integer.
        predict_kwargs : dict, default None
            Additional keyword arguments forwarded verbatim to the
            pipeline's `predict_quantiles` method.
        device_map : str, default 'auto'
            Device placement for the model. `"auto"` selects the best
            available accelerator (CUDA > MPS > CPU). Also accepts
            explicit values such as `"cuda"`, `"mps"`, or `"cpu"`,
            forwarded to `BaseChronosPipeline.from_pretrained`.
        torch_dtype : object, default None
            Torch dtype forwarded to `BaseChronosPipeline.from_pretrained`
            (e.g. `torch.bfloat16`).
        cross_learning : bool, default False
            If `True`, Chronos shares information across all series in
            the batch when predicting in multi-series mode. Forwarded
            directly to `predict_quantiles`. Ignored in single-series mode.
        
        """

        if not isinstance(context_length, int) or context_length < 1:
            raise ValueError(
                f"`context_length` must be a positive integer. Got {context_length!r}."
            )

        self.model_id       = model_id
        self._pipeline      = pipeline
        self.context_       = None
        self.context_exog_  = None
        self.context_length = context_length
        self.predict_kwargs = predict_kwargs or {}
        self.device_map     = device_map
        self.torch_dtype    = torch_dtype
        self.cross_learning = cross_learning
        self.is_fitted      = False

    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters.

        Returns
        -------
        params : dict
            Keys: `model_id`, `cross_learning`, `context_length`,
            `device_map`, `torch_dtype`, `predict_kwargs`.
        
        """
        return {
            'model_id':       self.model_id,
            'cross_learning': self.cross_learning,
            'context_length': self.context_length,
            'device_map':     self.device_map,
            'torch_dtype':    self.torch_dtype,
            'predict_kwargs': self.predict_kwargs or None,
        }

    def set_params(self, **params) -> ChronosAdapter:
        """
        Set adapter parameters. Resets the pipeline when a device or dtype
        param changes, since those are baked into the loaded pipeline.

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `cross_learning`, `context_length`,
            `device_map`, `torch_dtype`, `predict_kwargs`.

        Returns
        -------
        self : ChronosAdapter

        """

        valid = {
            'model_id', 'cross_learning', 'context_length',
            'device_map', 'torch_dtype', 'predict_kwargs',
        }
        invalid = set(params) - valid
        if invalid:
            raise ValueError(
                f"Invalid parameter(s) for ChronosAdapter: {sorted(invalid)}. "
                f"Valid parameters are: {sorted(valid)}."
            )
        
        pipeline_reset_keys = {'model_id', 'device_map', 'torch_dtype'}
        if params.keys() & pipeline_reset_keys:
            self._pipeline = None
        
        for key, value in params.items():
            if key == 'predict_kwargs':
                self.predict_kwargs = value or {}
            elif key == 'context_length':
                if not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"`context_length` must be a positive integer. Got {value!r}."
                    )
                self.context_length = value
            else:
                setattr(self, key, value)
        
        return self

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None],
    ) -> ChronosAdapter:
        """
        Store the training series and optional historical exogenous variables.
        No model training occurs since Chronos is a zero-shot inference model.

        All input normalization and validation is performed upstream by
        `FoundationModel`; this method receives canonical dicts only.

        Parameters
        ----------
        context : dict pandas Series
            Normalized training series, one entry per series.
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series historical exogenous variables (past covariates).

        Returns
        -------
        self : ChronosAdapter

        """

        self.context_ = context
        self.context_exog_ = context_exog
        self.is_fitted = True

        return self

    def predict(
        self,
        steps: int,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None],
        exog: dict[str, pd.DataFrame | pd.Series | None],
        quantiles: list[float] | tuple[float] | None
    ) -> dict[str, np.ndarray]:
        """
        Generate predictions using the Chronos pipeline.

        All input normalization, validation, and context trimming is
        performed upstream by `FoundationModel`; this method receives
        pre-processed dicts only.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        context : dict
            Per-series context windows (already trimmed to
            `context_length`).
        context_exog : dict
            Per-series past covariates (already trimmed).
        exog : dict
            Per-series future covariates for the forecast horizon.
        quantiles : list of float or None
            Quantile levels to return. If `None`, a point forecast
            (median, quantile 0.5) is produced.

        Returns
        -------
        predictions : dict
            Keys are series names. Each value is a 2-D array of shape
            `(steps, n_quantiles)`.
        
        """

        # NOTE: the pipeline is loaded lazily here so that the adapter can be
        # instantiated and fitted without requiring Chronos to be installed.
        self._load_pipeline()

        series_names_in = list(context.keys())
        quantile_levels = list(quantiles) if quantiles is not None else [0.5]

        inputs_list = [
            self._build_chronos_input(
                context      = context[name].to_numpy(),
                context_exog = context_exog[name] if context_exog is not None else None,
                exog         = exog[name] if exog is not None else None,
            )
            for name in series_names_in
        ]

        quantile_preds, _ = self._pipeline.predict_quantiles(
            inputs            = inputs_list,
            prediction_length = steps,
            quantile_levels   = quantile_levels,
            cross_learning    = self.cross_learning if len(series_names_in) > 1 else False,
            **self.predict_kwargs,
        )

        predictions: dict[str, np.ndarray] = {}
        for i, name in enumerate(series_names_in):
            q_arr = quantile_preds[i].squeeze(0)
            if hasattr(q_arr, "detach"):
                q_arr = q_arr.detach().cpu().numpy()
            else:
                q_arr = np.asarray(q_arr)
            predictions[name] = q_arr

        return predictions

    def _load_pipeline(self) -> None:
        """
        Load the Chronos pipeline into `self._pipeline` if not already set.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If `chronos-forecasting` >=2.0 is not installed.

        Notes
        -----
        The pipeline is imported lazily from `chronos` and instantiated via
        `BaseChronosPipeline.from_pretrained`, which auto-dispatches to the
        correct pipeline class based on the model config. Optional
        `device_map` and `torch_dtype` stored at initialisation are
        forwarded to the constructor. This method is a no-op when
        `self._pipeline` is already populated.

        """

        if self._pipeline is not None:
            return
        try:
            from chronos import BaseChronosPipeline
        except ImportError as exc:
            raise ImportError(
                "chronos-forecasting >=2.0 is required. "
                "Install it with `pip install chronos-forecasting`."
            ) from exc

        kwargs: dict[str, Any] = {}
        kwargs["device_map"] = self.device_map
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        
        self._pipeline = BaseChronosPipeline.from_pretrained(self.model_id, **kwargs)

    @staticmethod
    def _to_covariate_array(col_data: Any) -> np.ndarray:
        """
        Convert a covariate column to a numpy array.

        Numeric columns (int, float) and boolean columns are cast to
        `float32`. All other dtypes (object, string, Categorical) are left
        as-is so that Chronos can handle them as categorical covariates
        natively.

        Parameters
        ----------
        col_data : array-like
            A single covariate column (e.g. a pandas Series or 1-D array).

        Returns
        -------
        col_array : numpy ndarray
            A 1-D numpy array. Numeric/bool are cast to `float32`. Others
            keep their original dtype (typically `object` for string and
            categorical data).
        
        """

        # Handle pandas Series first to correctly process nullable extension
        # dtypes (pd.Int64Dtype, pd.Float64Dtype, pd.BooleanDtype): np.asarray()
        # on those produces dtype=object with pd.NA sentinels instead of float32.
        if isinstance(col_data, pd.Series):
            if pd.api.types.is_numeric_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
                return col_data.astype(np.float32).to_numpy()
            return col_data.to_numpy()

        # Fallback for numpy arrays, lists, etc.
        arr = np.asarray(col_data)
        if arr.dtype.kind in ("i", "u", "f", "b"):  # integer, unsigned int, float, bool
            return arr.astype(np.float32)
        
        return arr

    def _build_chronos_input(
        self,
        context: np.ndarray,
        context_exog: pd.DataFrame | pd.Series | None = None,
        exog: pd.DataFrame | pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Build the input dict consumed by the pipeline's `predict_quantiles` method.

        Parameters
        ----------
        context : numpy ndarray
            1-D array of observed time series values used as context. Must be
            castable to `float32`.
        context_exog : pandas DataFrame, pandas Series, default None
            Historical exogenous variables whose index is aligned to
            `context`. Each column (or the single Series, referenced by
            its name) becomes an entry in the returned
            "past_covariates" dict. Numeric and boolean columns are
            cast to `float32`; string and categorical columns are passed
            as-is and handled natively by Chronos.
        exog : pandas DataFrame, pandas Series, default None
            Future-known exogenous variables covering the forecast horizon.
            Must have exactly `prediction_length` rows. Each column
            becomes an entry in the returned "future_covariates" dict.
            Numeric and boolean columns are cast to `float32`; string and
            categorical columns are passed as-is.

        Returns
        -------
        input_dict : dict
            Dictionary with mandatory key "target" (1-D `float32`
            `numpy ndarray`) and optional keys "past_covariates" and
            "future_covariates", each mapping column names to 1-D
            arrays (`float32` for numeric/bool columns, `object` dtype
            for string/categorical columns).
        
        """

        input_dict = {"target": np.asarray(context, dtype=np.float32)}
        if context_exog is not None:
            df = (
                context_exog
                if isinstance(context_exog, pd.DataFrame)
                else context_exog.to_frame()
            )
            input_dict["past_covariates"] = {
                col: ChronosAdapter._to_covariate_array(df[col]) for col in df.columns
            }
        if exog is not None:
            df = (
                exog
                if isinstance(exog, pd.DataFrame)
                else exog.to_frame()
            )
            input_dict["future_covariates"] = {
                col: ChronosAdapter._to_covariate_array(df[col]) for col in df.columns
            }
        
        return input_dict


class TimesFMAdapter:
    """
    Adapter for Google TimesFM foundation models.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. "google/timesfm-2.5-200m-pytorch".
    model : object, default None
        Pre-loaded and compiled TimesFM model instance. If `None`, the
        model is loaded and compiled lazily on the first `predict` call.
    context_length : int, default 512
        Maximum number of historical observations to use as context. At fit
        time only the last `context_length` observations are stored. At
        predict time, if `context` is longer than `context_length` it
        is trimmed to this length; if it is shorter, all available
        observations are used as-is. Must be a positive integer. Defaults to
        512. TimesFM supports up to 16_384.
    max_horizon : int, default 512
        Maximum forecast horizon. If `predict` is called with
        `steps > max_horizon`, a `ValueError` is raised. The model is
        compiled lazily for the exact requested `steps` (up to this
        ceiling) to avoid unnecessary decode iterations. Must be a
        positive integer.
    forecast_config_kwargs : dict, default None
        Additional keyword arguments forwarded verbatim to
        `timesfm.ForecastConfig` at compile time. Supported keys:
        `normalize_inputs`, `use_continuous_quantile_head`,
        `force_flip_invariance`, `infer_is_positive`,
        `fix_quantile_crossing`. Do **not** include `max_context` or
        `max_horizon` here — those are controlled by the corresponding
        adapter parameters.

    Attributes
    ----------
    model_id : str
        HuggingFace model ID.
    context_ : dict
        Stored training series after fitting.
    context_exog_ : dict
        Not used, present here for API consistency by convention.
    context_length : int
        Maximum number of historical observations used as context.
    max_horizon : int
        Maximum forecast horizon.
    forecast_config_kwargs : dict
        Additional keyword arguments forwarded to `ForecastConfig`.
    is_fitted : bool
        Whether the adapter has been fitted.

    Notes
    -----
    TimesFM supports only the fixed quantile levels
    `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`. Requesting any
    other level raises a `ValueError`.

    Covariate support (via TimesFM's `forecast_with_covariates`) is not
    yet implemented. Passing `exog` or `context_exog` issues an
    `IgnoredArgumentWarning` and the values are discarded.

    References
    ----------
    .. [1] https://github.com/google-research/timesfm

    .. [2] https://huggingface.co/google/timesfm-2.5-200m-pytorch

    """

    SUPPORTED_QUANTILES: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    allow_exog: bool = False

    def __init__(
        self,
        model_id: str,
        *,
        model: Any | None = None,
        context_length: int = 512,
        max_horizon: int = 512,
        forecast_config_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID, e.g. "google/timesfm-2.5-200m-pytorch".
        model : object, default None
            Pre-loaded and compiled TimesFM model instance. If `None`, the
            model is loaded and compiled lazily on the first `predict` call.
        context_length : int, default 512
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations of
            `series` are stored. At `predict` time, if `context` is
            longer than `context_length` it is trimmed to this length;
            if it is shorter, all available observations are passed as-is.
            Must be a positive integer.
        max_horizon : int, default 512
            Maximum forecast horizon. If `predict` is called with
            `steps > max_horizon`, a `ValueError` is raised. The model
            is compiled lazily for the exact requested `steps` (up to
            this ceiling) to avoid unnecessary decode iterations. Must
            be a positive integer.
        forecast_config_kwargs : dict, default None
            Additional keyword arguments forwarded verbatim to
            `timesfm.ForecastConfig` at compile time.
        
        """

        if not isinstance(context_length, int) or context_length < 1:
            raise ValueError(
                f"`context_length` must be a positive integer. Got {context_length!r}."
            )
        if not isinstance(max_horizon, int) or max_horizon < 1:
            raise ValueError(
                f"`max_horizon` must be a positive integer. Got {max_horizon!r}."
            )

        self.model_id               = model_id
        self._model                 = model
        self.context_               = None
        self.context_exog_          = None
        self.context_length         = context_length
        self.max_horizon            = max_horizon
        self.forecast_config_kwargs = dict(forecast_config_kwargs) if forecast_config_kwargs else {}
        self.is_fitted              = False

    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters.

        Returns
        -------
        params : dict
            Keys: `model_id`, `context_length`, `max_horizon`,
            `forecast_config_kwargs`.
        
        """
        return {
            'model_id':               self.model_id,
            'context_length':         self.context_length,
            'max_horizon':            self.max_horizon,
            'forecast_config_kwargs': self.forecast_config_kwargs or None,
        }

    def set_params(self, **params) -> TimesFMAdapter:
        """
        Set adapter parameters. Resets the model when parameters that affect
        compilation change (`model_id`, `context_length`, `max_horizon`,
        `forecast_config_kwargs`).

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `context_length`, `max_horizon`,
            `forecast_config_kwargs`.

        Returns
        -------
        self : TimesFMAdapter

        """

        valid = {'model_id', 'context_length', 'max_horizon', 'forecast_config_kwargs'}
        invalid = set(params) - valid
        if invalid:
            raise ValueError(
                f"Invalid parameter(s) for TimesFMAdapter: {sorted(invalid)}. "
                f"Valid parameters are: {sorted(valid)}."
            )
        model_reset_keys = {'model_id', 'context_length', 'max_horizon', 'forecast_config_kwargs'}
        if params.keys() & model_reset_keys:
            self._model = None
        for key, value in params.items():
            if key == 'context_length':
                if not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"`context_length` must be a positive integer. Got {value!r}."
                    )
                self.context_length = value
            elif key == 'max_horizon':
                if not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"`max_horizon` must be a positive integer. Got {value!r}."
                    )
                self.max_horizon = value
            elif key == 'forecast_config_kwargs':
                self.forecast_config_kwargs = dict(value) if value else {}
            else:
                setattr(self, key, value)
        
        return self

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: Any,
    ) -> TimesFMAdapter:
        """
        Store the training series.
        No model training occurs since TimesFM is a zero-shot inference model.

        All input normalization and validation is performed upstream by
        `FoundationModel`; this method receives canonical dicts only.

        Parameters
        ----------
        context : dict pandas Series
            Normalized training series, one entry per series.
        context_exog : Any
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : TimesFMAdapter

        """

        self.context_ = context
        self.is_fitted = True
        
        return self

    def predict(
        self,
        steps: int,
        context: dict[str, pd.Series],
        context_exog: Any,
        exog: Any,
        quantiles: list[float] | tuple[float] | None,
    ) -> dict[str, np.ndarray]:
        """
        Generate predictions using the TimesFM model.

        All input normalization, validation, and context trimming is
        performed upstream by `FoundationModel`; this method receives
        pre-processed dicts only.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        context : dict
            Per-series context windows (already trimmed to
            `context_length`).
        context_exog : Any
            Not used, present here for API consistency by convention.
        exog : Any
            Not used, present here for API consistency by convention.
        quantiles : list of float or None
            Quantile levels. Must be a subset of `SUPPORTED_QUANTILES`.

        Returns
        -------
        predictions : dict
            Keys are series names. Each value is a 2-D array of shape
            `(steps, n_quantiles)`.

        Raises
        ------
        ValueError
            If a requested quantile level is not in `SUPPORTED_QUANTILES`
            or `steps` exceeds `max_horizon`.
        
        """

        if quantiles is not None:
            quantile_list = list(quantiles)
            for q in quantile_list:
                if not any(abs(q - sq) < 1e-9 for sq in self.SUPPORTED_QUANTILES):
                    raise ValueError(
                        f"TimesFM only supports quantile levels "
                        f"{self.SUPPORTED_QUANTILES}. Got {q!r}. "
                        f"Quantile interpolation is not supported."
                    )
        else:
            quantile_list = None

        if steps > self.max_horizon:
            raise ValueError(
                f"`steps` ({steps}) exceeds `max_horizon` ({self.max_horizon})."
            )

        self._load_model()
        self._ensure_compiled(steps)

        series_names_in = list(context.keys())
        inputs_list = [
            context[name].to_numpy() for name in series_names_in
        ]

        point_forecast, quantile_forecast = self._model.forecast(
            horizon=steps,
            inputs=inputs_list,
        )
        # point_forecast  : (n_series, steps)
        # quantile_forecast: (n_series, steps, 10)  — idx 0 = mean, 1-9 = q0.1-q0.9

        predictions: dict[str, np.ndarray] = {}
        for i, name in enumerate(series_names_in):
            if quantile_list is None:
                # Point forecast: shape (steps, 1)
                predictions[name] = np.asarray(point_forecast[i]).reshape(-1, 1)
            else:
                q_indices = [round(q * 10) for q in quantile_list]
                qf = np.asarray(quantile_forecast[i])
                predictions[name] = qf[:, q_indices]  # (steps, n_quantiles)

        return predictions

    def _load_model(self) -> None:
        """
        Load (but do not compile) the TimesFM model into `self._model`
        if not already set.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If `timesfm[torch]` is not installed.

        Notes
        -----
        The model is imported lazily from `timesfm` and loaded via
        `TimesFM_2p5_200M_torch.from_pretrained`. Compilation is deferred to
        `_ensure_compiled`, which is called from `predict` with the actual
        forecast horizon so that the compiled decode graph is sized exactly
        for the requested number of steps rather than the (much larger)
        `max_horizon` ceiling. This method is a no-op when `self._model` is
        already populated.
        """

        if self._model is not None:
            return
        try:
            import timesfm
        except ImportError as exc:
            raise ImportError(
                "timesfm is required for TimesFMAdapter. "
                "Install it with `pip install git+https://github.com/google-research/timesfm.git`."
            ) from exc

        # Workaround for a compatibility issue between huggingface_hub and
        # timesfm: huggingface_hub's `from_pretrained` passes `proxies` and
        # `resume_download` to `_from_pretrained`, but timesfm's
        # `_from_pretrained` does not declare them as explicit parameters, so
        # they fall into **model_kwargs and are forwarded to __init__, raising
        # a TypeError. A local subclass overrides `_from_pretrained` to absorb
        # those kwargs without modifying any global state.
        class _TimesFMCompat(timesfm.TimesFM_2p5_200M_torch):
            @classmethod
            def _from_pretrained(cls, *, proxies=None, resume_download=None, **kwargs):  # type: ignore[override]
                return super()._from_pretrained(**kwargs)

        self._model = _TimesFMCompat.from_pretrained(self.model_id)

    def _ensure_compiled(self, steps: int) -> None:
        """
        Compile the model for the given forecast horizon if not already
        compiled for at least `steps` steps.

        Parameters
        ----------
        steps : int
            The forecast horizon that the model must support.

        Returns
        -------
        None

        Notes
        -----
        This is separated from `_load_model` so that compilation uses the
        *actual* number of requested forecast steps rather than `max_horizon`.
        TimesFM's compiled decode always runs `forecast_config.max_horizon`
        autoregressive decode iterations regardless of the requested horizon;
        the true horizon is only used to *slice* the output afterwards. When
        the compiled `max_horizon` is large (e.g. the default 512) but
        `steps` is small (e.g. 12), the model performs up to
        `(max_horizon - 1) // output_patch_len` unnecessary extra transformer
        forward passes per inference call. Compiling here with
        `max_horizon = steps` reduces those wasted passes to zero for the
        typical backtesting case where `steps` is constant across folds.

        If the model was already compiled for a horizon `>= steps` (e.g. a
        pre-compiled model passed via the `model` constructor argument), this
        method is a no-op.
        """

        fc = getattr(self._model, 'forecast_config', None)
        if fc is not None and steps <= fc.max_horizon:
            return

        import timesfm
        self._model.compile(
            timesfm.ForecastConfig(
                max_context = self.context_length,
                max_horizon = steps,
                **self.forecast_config_kwargs,
            )
        )


class MoiraiAdapter:
    """
    Adapter for Salesforce Moirai foundation models.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. `"Salesforce/moirai-2.0-R-small"`.
        Must be a `Salesforce/moirai-2.0-R-{small,base,large}` variant.
    module : object, default None
        Pre-loaded `Moirai2Module` instance. If `None`, the module is
        loaded lazily on the first call to `predict`.
    context_length : int, default 2048
        Maximum number of historical observations to use as context. At fit
        time only the last `context_length` observations are stored. At
        predict time, if `context` is longer than `context_length`
        it is trimmed to this length; if it is shorter, all available
        observations are used as-is. Must be a positive integer.
    device : str, default 'auto'
        Device placement for the model. `"auto"` selects the best
        available accelerator (CUDA > MPS > CPU). Also accepts explicit
        values such as `"cuda"`, `"mps"`, or `"cpu"`.

    Attributes
    ----------
    model_id : str
        HuggingFace model ID.
    context_ : dict
        Stored training series after fitting.
    context_exog_ : dict
        Not used, present here for API consistency by convention.
    context_length : int
        Maximum number of historical observations used as context.
    device : str
        Device placement for the model.
    _forecast_obj : object
        Internal Moirai forecast object, populated at the first call to
        `predict`.
    is_fitted : bool
        Whether the adapter has been fitted.

    Notes
    -----
    Moirai supports only the fixed quantile levels
    `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`. Requesting any
    other level raises a `ValueError`.

    Covariate support via the high-level `Moirai2Forecast.predict()` API
    is not functional: the padding/truncation loop inside `predict()`
    clips every list-valued field — including `feat_dynamic_real` — to
    `context_length`, discarding the future portion that future
    covariates require. Passing `exog` or `context_exog` issues an
    `IgnoredArgumentWarning` and the values are discarded.

    References
    ----------
    .. [1] https://github.com/SalesforceAIResearch/uni2ts

    .. [2] https://huggingface.co/Salesforce/moirai-2.0-R-small

    """

    SUPPORTED_QUANTILES: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    allow_exog: bool = False

    def __init__(
        self,
        model_id: str,
        *,
        module: Any | None = None,
        context_length: int = 2048,
        device: str = "auto",
    ) -> None:
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID, e.g. `"Salesforce/moirai-2.0-R-small"`.
        module : object, default None
            Pre-loaded `Moirai2Module` instance. If `None`, the module
            is loaded lazily on the first call to `predict`.
        context_length : int, default 2048
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations of
            `series` are stored. At `predict` time, if `context`
            is longer than `context_length` it is trimmed to this length;
            if it is shorter, all available observations are passed as-is.
            Must be a positive integer.
        device : str, default 'auto'
            Device placement for the model. `"auto"` selects the best
            available accelerator (CUDA > MPS > CPU). Also accepts
            explicit values such as `"cuda"`, `"mps"`, or `"cpu"`.
        
        """

        if not isinstance(context_length, int) or context_length < 1:
            raise ValueError(
                f"`context_length` must be a positive integer. "
                f"Got {context_length!r}."
            )

        self.model_id       = model_id
        self._module        = module
        self.context_       = None
        self.context_exog_  = None
        self.context_length = context_length
        self.device         = device
        self._forecast_obj  = None
        self.is_fitted      = False

    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters.

        Returns
        -------
        params : dict
            Keys: `model_id`, `context_length`, `device`.
        """
        return {
            'model_id':       self.model_id,
            'context_length': self.context_length,
            'device':         self.device,
        }

    def set_params(self, **params) -> MoiraiAdapter:
        """
        Set adapter parameters. Resets the module and forecast object when
        `model_id` or `context_length` changes.

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `context_length`, `device`.

        Returns
        -------
        self : MoiraiAdapter

        """

        valid = {'model_id', 'context_length', 'device'}
        invalid = set(params) - valid
        if invalid:
            raise ValueError(
                f"Invalid parameter(s) for MoiraiAdapter: {sorted(invalid)}. "
                f"Valid parameters are: {sorted(valid)}."
            )
        if params.keys() & {'model_id', 'context_length', 'device'}:
            self._module = None
            self._forecast_obj = None
        for key, value in params.items():
            if key == 'context_length':
                if not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"`context_length` must be a positive integer. "
                        f"Got {value!r}."
                    )
                self.context_length = value
            else:
                setattr(self, key, value)
        
        return self

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: Any,
    ) -> MoiraiAdapter:
        """
        Store the training series.
        No model training occurs since Moirai is a zero-shot inference model.

        All input normalization and validation is performed upstream by
        `FoundationModel`; this method receives canonical dicts only.

        Parameters
        ----------
        context : dict pandas Series
            Normalized training series, one entry per series.
        context_exog : Any
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : MoiraiAdapter

        """

        self.context_ = context
        self.is_fitted = True

        return self

    def predict(
        self,
        steps: int,
        context: dict[str, pd.Series],
        context_exog: Any,
        exog: Any,
        quantiles: list[float] | tuple[float] | None,
    ) -> dict[str, np.ndarray]:
        """
        Generate predictions using Moirai.

        All input normalization, validation, and context trimming is
        performed upstream by `FoundationModel`; this method receives
        pre-processed dicts only.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        context : dict pandas Series
            Per-series context windows (already trimmed to
            `context_length`).
        context_exog : Any
            Not used, present here for API consistency by convention.
        exog : Any
            Not used, present here for API consistency by convention.
        quantiles : list of float or None
            Quantile levels. Must be a subset of `SUPPORTED_QUANTILES`.

        Returns
        -------
        predictions : dict
            Keys are series names. Each value is a 2-D array of shape
            `(steps, n_quantiles)`.

        Raises
        ------
        ValueError
            If a requested quantile level is not in `SUPPORTED_QUANTILES`.
        
        """

        if quantiles is not None:
            quantile_list = list(quantiles)
            for q in quantile_list:
                if not any(abs(q - sq) < 1e-9 for sq in self.SUPPORTED_QUANTILES):
                    raise ValueError(
                        f"Moirai only supports quantile levels "
                        f"{self.SUPPORTED_QUANTILES}. Got {q!r}. "
                        f"Quantile interpolation is not supported."
                    )
        else:
            quantile_list = None

        quantile_levels = quantile_list if quantile_list is not None else [0.5]
        q_indices = [
            next(
                i for i, sq in enumerate(self.SUPPORTED_QUANTILES)
                if abs(q - sq) < 1e-9
            )
            for q in quantile_levels
        ]

        series_names_in = list(context.keys())
        inputs_list = [
            context[name].to_numpy(dtype=np.float32).reshape(-1, 1)
            for name in series_names_in
        ]

        raw = self._run_inference(inputs_list, steps)

        predictions: dict[str, np.ndarray] = {}
        for i, name in enumerate(series_names_in):
            predictions[name] = raw[i][q_indices, :].T  # (steps, n_quantiles)

        return predictions

    def _load_module(self) -> None:
        """
        Load the `Moirai2Module` into `self._module` if not already set.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If `uni2ts` is not installed.

        Notes
        -----
        The module is imported lazily from `uni2ts` and instantiated via
        `Moirai2Module.from_pretrained`, then set to evaluation mode.
        This method is a no-op when `self._module` is already populated.
        """

        if self._module is not None:
            return
        try:
            from uni2ts.model.moirai2 import Moirai2Module
        except ImportError as exc:
            raise ImportError(
                "uni2ts is required for MoiraiAdapter. "
                "Install it with `pip install uni2ts`."
            ) from exc
        self._module = Moirai2Module.from_pretrained(self.model_id)
        self._module.eval()

    def _ensure_forecast_obj(self) -> None:
        """
        Build the `Moirai2Forecast` inference wrapper if not already set.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If `uni2ts` is not installed.

        Notes
        -----
        Calls `_load_module` then wraps `self._module` in a
        `Moirai2Forecast` with `prediction_length=1` (overridden
        per-call via `hparams_context`), sets it to evaluation mode,
        and moves it to the device specified by `self.device`.
        This method is a no-op when `self._forecast_obj` is already
        populated.
        """

        if self._forecast_obj is not None:
            return
        
        self._load_module()
        from uni2ts.model.moirai2 import Moirai2Forecast

        self._forecast_obj = Moirai2Forecast(
            module                     = self._module,
            prediction_length          = 1,
            context_length             = self.context_length,
            target_dim                 = 1,
            feat_dynamic_real_dim      = 0,
            past_feat_dynamic_real_dim = 0,
        ).eval()

        resolved_device = _resolve_torch_device(self.device)
        if resolved_device == "mps":
            warnings.warn(
                "MPS device is not supported by Moirai because the uni2ts "
                "library uses float64 operations internally. Falling back "
                "to CPU.",
                stacklevel=6,
            )
            resolved_device = "cpu"
        self._forecast_obj.to(resolved_device)

    def _run_inference(
        self,
        inputs_list: list[np.ndarray],
        steps: int,
    ) -> np.ndarray:
        """
        Run batched inference with `Moirai2Forecast`.

        Parameters
        ----------
        inputs_list : list of numpy ndarray
            List of 2-D arrays with shape `(T, 1)`, one per series.
            Each array holds `float32` values.
        steps : int
            Forecast horizon.

        Returns
        -------
        raw : numpy ndarray
            Array of shape `(n_series, 9, steps)` containing quantile
            forecasts for the 9 fixed levels in `SUPPORTED_QUANTILES`
            order.
        
        """

        self._ensure_forecast_obj()
        with self._forecast_obj.hparams_context(prediction_length=steps):
            raw = self._forecast_obj.predict(inputs_list)
        
        return raw


class TabICLAdapter:
    """
    Adapter for TabICL zero-shot time-series foundation models.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. `"soda-inria/tabicl"`.
    model : object, default None
        Pre-instantiated `TabICLForecaster` instance. If `None`, a new
        instance is created lazily on the first call to `predict`. Intended
        for testing only.
    context_length : int, default 4096
        Maximum number of historical observations to use as context. At fit
        time only the last `context_length` observations are stored. At
        predict time, if `context` is longer than `context_length` it is
        trimmed to this length; if it is shorter, all available observations
        are used as-is. Must be a positive integer.
    point_estimate : str, default 'mean'
        Method used to derive the point forecast from the TabICL output.
        Accepted values: `'mean'`, `'median'`.
    tabicl_config : dict, default None
        Additional keyword arguments forwarded verbatim to
        `TabICLRegressor` at inference time. If `None`, defaults to empty
        dict (TabICL's own defaults).
    temporal_features : list, default None
        List of `TimeTransform` instances applied to the time series before
        inference. If `None`, TabICL uses its default transforms:
        `[IndexEncoder(), DatetimeEncoder(), AutoPeriodicEncoder()]`. Pass
        an empty list to disable all temporal feature engineering.

    Attributes
    ----------
    model_id : str
        HuggingFace model ID.
    context_ : dict
        Stored training series after fitting.
    context_exog_ : dict
        Stored historical exogenous variables after fitting.
    context_length : int
        Maximum number of historical observations used as context.
    point_estimate : str
        Point forecast method.
    tabicl_config : dict
        Additional configuration forwarded to `TabICLRegressor`.
    temporal_features : list
        Temporal feature transforms applied to the series.
    is_fitted : bool
        Whether the adapter has been fitted.
    _model : object
        Internal `TabICLForecaster` instance. `None` until the first call
        to `predict`, after which it is cached for reuse.

    Notes
    -----
    TabICL supports arbitrary quantile levels (any float in `[0, 1]`),
    unlike models with fixed quantile sets such as TimesFM or Moirai.

    Covariate support is available: extra columns in `context` and `exog`
    are forwarded as covariates. TabICL uses only the intersection of columns
    present in both context and future data (missing values are filled with
    `NaN`).

    Series with a `RangeIndex` are accepted. Internally, TabICL requires
    datetime timestamps, so a synthetic daily `DatetimeIndex` (starting
    2000-01-01) is used. Calendar-based transforms
    (`DatetimeEncoder`, `AutoPeriodicEncoder`) will not be meaningful for
    such series; consider passing `temporal_features=[]` or
    `temporal_features=[IndexEncoder()]` in that case.

    References
    ----------
    .. [1] https://github.com/soda-inria/tabicl

    .. [2] https://tabicl.readthedocs.io/en/latest/

    """

    allow_exog: bool = True

    def __init__(
        self,
        model_id: str,
        *,
        model: Any | None = None,
        context_length: int = 4096,
        point_estimate: str = "mean",
        tabicl_config: dict[str, Any] | None = None,
        temporal_features: list[Any] | None = None,
    ) -> None:
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID, e.g. `"soda-inria/tabicl"`.
        model : object, default None
            Pre-instantiated `TabICLForecaster` instance. If `None`, a new
            instance is created lazily on the first call to `predict`.
            Intended for testing only.
        context_length : int, default 4096
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations of
            `series` (and `exog`) are stored. At `predict` time, if
            `context` is longer than `context_length` it is trimmed to
            this length before inference; if it is shorter, all available
            observations are passed as-is. Must be a positive integer.
        point_estimate : str, default 'mean'
            Method used to derive the point forecast. Accepted values:
            `'mean'`, `'median'`.
        tabicl_config : dict, default None
            Additional keyword arguments forwarded verbatim to
            `TabICLRegressor` at inference time.
        temporal_features : list, default None
            List of `TimeTransform` instances applied before inference. If
            `None`, TabICL uses its defaults. Pass `[]` to disable all
            temporal feature engineering.

        """

        if not isinstance(context_length, int) or context_length < 1:
            raise ValueError(
                f"`context_length` must be a positive integer. Got {context_length!r}."
            )
        if point_estimate not in ("mean", "median"):
            raise ValueError(
                f"`point_estimate` must be 'mean' or 'median'. Got {point_estimate!r}."
            )

        self.model_id          = model_id
        self._model            = model
        self.context_          = None
        self.context_exog_     = None
        self.context_length    = context_length
        self.point_estimate    = point_estimate
        self.tabicl_config     = dict(tabicl_config) if tabicl_config else {}
        self.temporal_features = temporal_features
        self.is_fitted         = False

    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters.

        Returns
        -------
        params : dict
            Keys: `model_id`, `context_length`, `point_estimate`,
            `tabicl_config`, `temporal_features`. `tabicl_config` is
            returned as `None` when no additional config was set (i.e.
            when the internal dict is empty).

        """
        return {
            "model_id":          self.model_id,
            "context_length":    self.context_length,
            "point_estimate":    self.point_estimate,
            "tabicl_config":     self.tabicl_config or None,
            "temporal_features": self.temporal_features,
        }

    def set_params(self, **params) -> TabICLAdapter:
        """
        Set adapter parameters. Resets the model when any parameter changes,
        since the `TabICLForecaster` is instantiated lazily on the first
        `predict` call using the current adapter state.

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `context_length`, `point_estimate`,
            `tabicl_config`, `temporal_features`.

        Returns
        -------
        self : TabICLAdapter

        """

        valid = {
            "model_id", "context_length", "point_estimate",
            "tabicl_config", "temporal_features",
        }
        invalid = set(params) - valid
        if invalid:
            raise ValueError(
                f"Invalid parameter(s) for TabICLAdapter: {sorted(invalid)}. "
                f"Valid parameters are: {sorted(valid)}."
            )

        validated = {}
        for key, value in params.items():
            if key == "context_length":
                if not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"`context_length` must be a positive integer. Got {value!r}."
                    )
                validated[key] = value
            elif key == "point_estimate":
                if value not in ("mean", "median"):
                    raise ValueError(
                        f"`point_estimate` must be 'mean' or 'median'. Got {value!r}."
                    )
                validated[key] = value
            elif key == "tabicl_config":
                validated[key] = dict(value) if value else {}
            else:
                validated[key] = value

        actually_changed = {
            k: v for k, v in validated.items()
            if getattr(self, k) != v
        }
        if actually_changed:
            self._model = None
            for key, value in actually_changed.items():
                setattr(self, key, value)

        return self

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
    ) -> TabICLAdapter:
        """
        Store the training series and optional historical exogenous variables.
        No model training occurs since TabICL is a zero-shot inference model.

        All input normalization and validation is performed upstream by
        `FoundationModel`; this method receives canonical dicts only.

        Parameters
        ----------
        context : dict pandas Series
            Normalized training series, one entry per series.
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series historical exogenous variables (past covariates).

        Returns
        -------
        self : TabICLAdapter

        """

        self.context_      = context
        self.context_exog_ = context_exog
        self.is_fitted     = True

        return self

    def predict(
        self,
        steps: int,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        quantiles: list[float] | tuple[float] | None,
    ) -> dict[str, np.ndarray]:
        """
        Generate predictions using TabICL.

        All input normalization, validation, and context trimming is
        performed upstream by `FoundationModel`; this method receives
        pre-processed dicts only.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        context : dict pandas Series
            Per-series context windows (already trimmed to
            `context_length`).
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series past covariates (already trimmed).
        exog : dict pandas DataFrame, pandas Series, or None
            Per-series future covariates for the forecast horizon.
        quantiles : list of float or None
            Quantile levels to return. If `None`, a point forecast is
            produced (shape `(steps, 1)`). Accepts any float in `[0, 1]`.

        Returns
        -------
        predictions : dict
            Keys are series names. Each value is a 2-D numpy ndarray of
            shape `(steps, n_quantiles)`.

        """

        self._load_model()

        quantile_list = list(quantiles) if quantiles is not None else None
        tabicl_quantiles = (
            quantile_list
            if quantile_list is not None
            else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )

        series_names_in = list(context.keys())

        first_series = next(iter(context.values()))
        is_datetime = isinstance(first_series.index, pd.DatetimeIndex)

        if not is_datetime:
            warnings.warn(
                "TabICLAdapter received series with a non-DatetimeIndex. "
                "TabICL requires datetime timestamps internally; a synthetic "
                "daily DatetimeIndex (starting 2000-01-01) will be used. "
                "Calendar-based temporal features (DatetimeEncoder, "
                "AutoPeriodicEncoder) will not be meaningful for "
                "integer-indexed data. Consider passing "
                "`temporal_features=[]` to disable calendar feature "
                "transforms.",
                # stacklevel=3: TabICLAdapter.predict → FoundationModel.predict → user
                stacklevel=3,
            )

        context_df = self._build_context_df(
                         series_names = series_names_in, 
                         context      = context, 
                         context_exog = context_exog, 
                         is_datetime  = is_datetime
                     )
        
        future_df = self._build_future_df(
                        series_names = series_names_in, 
                        context      = context, 
                        exog         = exog, 
                        steps        = steps, 
                        is_datetime  = is_datetime
                    )

        result_df = self._model.predict_df(
                        context_df = context_df,
                        future_df  = future_df,
                        quantiles  = tabicl_quantiles,
                    )

        # result_df is a plain DataFrame with MultiIndex (item_id, timestamp).
        # columns: "target" (str) and quantile levels as float column names.
        predictions: dict[str, np.ndarray] = {}
        for name in series_names_in:
            group = result_df.loc[name]  # DataFrame indexed by timestamp
            if quantile_list is None:
                predictions[name] = group["target"].to_numpy().reshape(-1, 1)
            else:
                predictions[name] = group[quantile_list].to_numpy()

        return predictions

    def _load_model(self) -> None:
        """
        Load the `TabICLForecaster` into `self._model` if not already set.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If `tabicl[forecast]` is not installed.

        Notes
        -----
        The model is imported lazily from `tabicl` and instantiated with
        the current adapter parameters. This method is a no-op when
        `self._model` is already populated (either by a prior call or by
        the `model` test-injection parameter).
        """

        if self._model is not None:
            return
        try:
            from tabicl.forecast import TabICLForecaster
        except ImportError as exc:
            raise ImportError(
                "tabicl[forecast] is required for TabICLAdapter. "
                "Install it with `pip install tabicl[forecast]`."
            ) from exc
        
        self._model = TabICLForecaster(
                          max_context_length = self.context_length,
                          temporal_features  = self.temporal_features,
                          point_estimate     = self.point_estimate,
                          tabicl_config      = self.tabicl_config or {},
                      )

    def _get_timestamps(
        self, series: pd.Series, is_datetime: bool
    ) -> pd.DatetimeIndex:
        """
        Return datetime timestamps for a context series.

        For `DatetimeIndex` series the original index is returned. For
        `RangeIndex` series a synthetic daily `DatetimeIndex` starting at
        2000-01-01 is created so that TabICL's requirement for datetime
        timestamps is satisfied.

        Parameters
        ----------
        series : pandas Series
            The context series.
        is_datetime : bool
            Whether the series has a `DatetimeIndex`.

        Returns
        -------
        timestamps : pandas DatetimeIndex
            Datetime timestamps aligned with the series values.

        """

        if is_datetime:
            return series.index
        
        return pd.date_range("2000-01-01", periods=len(series), freq="D")

    def _get_future_timestamps(
        self, series: pd.Series, steps: int, is_datetime: bool
    ) -> pd.DatetimeIndex:
        """
        Return datetime timestamps for the forecast horizon.

        For `DatetimeIndex` series the horizon is appended at the inferred
        frequency. For `RangeIndex` series the synthetic daily timeline
        (2000-01-01 + len(context) days) is extended by `steps` days.

        Parameters
        ----------
        series : pandas Series
            The context series (used to determine the end timestamp and
            frequency).
        steps : int
            Number of steps ahead.
        is_datetime : bool
            Whether the series has a `DatetimeIndex`.

        Returns
        -------
        timestamps : pandas DatetimeIndex
            Datetime timestamps for the `steps` forecast steps.

        """

        if is_datetime:
            freq = series.index.freq
            if freq is None:
                freq = pd.tseries.frequencies.to_offset(
                    pd.infer_freq(series.index)
                )
            timestamps = pd.date_range(
                             start   = series.index[-1] + freq,
                             periods = steps,
                             freq    = freq,
                         )
        else:
            n = len(series)
            timestamps = pd.date_range(
                             start   = pd.Timestamp("2000-01-01") + pd.Timedelta(days=n),
                             periods = steps,
                             freq    = "D",
                         )
        
        return timestamps

    def _build_context_df(
        self,
        series_names: list,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | None] | None,
        is_datetime: bool,
    ) -> pd.DataFrame:
        """
        Build a long-format context DataFrame expected by TabICL.

        Each series' observations become rows with `item_id`, `timestamp`,
        `target`, and optional exogenous covariate columns.

        Parameters
        ----------
        series_names : list
            Ordered list of series names.
        context : dict pandas Series
            Per-series context windows.
        context_exog : dict or None
            Per-series historical exogenous variables.
        is_datetime : bool
            Whether the series have a `DatetimeIndex`.

        Returns
        -------
        context_df : pandas DataFrame
            Long-format DataFrame with columns `item_id`, `timestamp`,
            `target`, and any exogenous columns.

        """

        context_df = []
        for name in series_names:
            series = context[name]
            n = len(series)
            part = pd.DataFrame({
                "item_id":   np.full(n, name),
                "timestamp": np.asarray(self._get_timestamps(series, is_datetime)),
                "target":    series.to_numpy(dtype=float),
            })
            exog_entry = (
                context_exog.get(name) if context_exog is not None else None
            )
            if exog_entry is not None:
                part = pd.concat(
                    [part, exog_entry.reset_index(drop=True)], axis=1
                )
            context_df.append(part)

        context_df = pd.concat(context_df, ignore_index=True)

        return context_df

    def _build_future_df(
        self,
        series_names: list,
        context: dict[str, pd.Series],
        exog: dict[str, pd.DataFrame | None] | None,
        steps: int,
        is_datetime: bool,
    ) -> pd.DataFrame:
        """
        Build a long-format future DataFrame expected by TabICL.

        Each series' forecast horizon becomes rows with `item_id`,
        `timestamp`, and optional future exogenous covariate columns.

        Parameters
        ----------
        series_names : list
            Ordered list of series names.
        context : dict pandas Series
            Per-series context windows (used to derive future timestamps).
        exog : dict or None
            Per-series future exogenous variables covering the forecast
            horizon.
        steps : int
            Number of steps ahead.
        is_datetime : bool
            Whether the series have a `DatetimeIndex`.

        Returns
        -------
        future_df : pandas DataFrame
            Long-format DataFrame with columns `item_id`, `timestamp`, and
            any future exogenous columns.

        """

        future_df = []
        for name in series_names:
            series = context[name]
            part = pd.DataFrame({
                "item_id":   np.full(steps, name),
                "timestamp": np.asarray(
                    self._get_future_timestamps(series, steps, is_datetime)
                ),
            })
            future_exog = exog.get(name) if exog is not None else None
            if future_exog is not None:
                part = pd.concat(
                    [part, future_exog.reset_index(drop=True)], axis=1
                )
            future_df.append(part)

        future_df = pd.concat(future_df, ignore_index=True)

        return future_df


class TabPFNAdapter:
    """
    Adapter for Prior Labs TabPFN-TS zero-shot time-series foundation models.

    TabPFN-TS frames forecasting as tabular regression: the series is
    featurized (running index, calendar features, automatically detected
    seasonal features) and a TabPFN regressor predicts the forecast horizon
    zero-shot.

    Parameters
    ----------
    model_id : str
        Model ID, e.g. `"priorlabs/tabpfn-ts"`. Used only to resolve this
        adapter; the underlying checkpoint is controlled by
        `tabpfn_model_config` (key `model_path`).
    model : object, default None
        Pre-instantiated `TabPFNTSPipeline` instance. If `None`, a new
        instance is created lazily on the first call to `predict`. Intended
        for testing only.
    context_length : int, default 32768
        Maximum number of historical observations to use as context. At fit
        time only the last `context_length` observations are stored. At
        predict time, if `context` is longer than `context_length` it is
        trimmed to this length; if it is shorter, all available observations
        are used as-is. Defaults to 32768, which matches the TabPFN-TS ship
        configuration; lower values (e.g. 4096) speed up inference at a small
        accuracy cost. Must be a positive integer.
    mode : str, default 'local'
        Inference mode. `'local'` runs the TabPFN model locally (CUDA > MPS >
        CPU selected automatically by the library; the checkpoint is
        downloaded on first use). `'client'` sends the featurized data to the
        Prior Labs cloud API via `tabpfn-client` (no GPU needed, requires an
        account/API key).
    point_estimate : str, default 'median'
        Method used to aggregate the TabPFN ensemble output into the point
        forecast. Accepted values: `'mean'`, `'median'`, `'mode'`.
    tabpfn_model_config : dict, default None
        Additional configuration forwarded verbatim to the underlying TabPFN
        regressor (e.g. `model_path`, `device`). If `None`, the library
        defaults are used.
    temporal_features : list, default None
        List of `FeatureGenerator` instances applied to the time series
        before inference. If `None`, TabPFN-TS uses its default transforms:
        `[RunningIndexFeature(), CalendarFeature(), AutoSeasonalFeature()]`.
        Pass an empty list to disable all temporal feature engineering.

    Attributes
    ----------
    model_id : str
        Model ID.
    context_ : dict
        Stored training series after fitting.
    context_exog_ : dict
        Stored historical exogenous variables after fitting.
    context_length : int
        Maximum number of historical observations used as context.
    mode : str
        Inference mode, `'local'` or `'client'`.
    point_estimate : str
        Point forecast aggregation method.
    tabpfn_model_config : dict
        Additional configuration forwarded to the TabPFN regressor.
    temporal_features : list
        Temporal feature transforms applied to the series.
    is_fitted : bool
        Whether the adapter has been fitted.
    _model : object
        Internal `TabPFNTSPipeline` instance. `None` until the first call
        to `predict`, after which it is cached for reuse.

    Notes
    -----
    TabPFN-TS supports arbitrary quantile levels (any float in `(0, 1)`),
    unlike models with fixed quantile sets such as TimesFM or Moirai.

    Covariate support is available for *known-future* covariates: extra
    columns present in both the historical context and the forecast horizon
    are used by the model. Covariates without future values are discarded by
    the library.

    Series with a `RangeIndex` are accepted. Internally, TabPFN-TS requires
    datetime timestamps, so a synthetic daily `DatetimeIndex` (starting
    2000-01-01) is used. Calendar-based transforms (`CalendarFeature`) will
    not be meaningful for such series; consider passing
    `temporal_features=[]` or `[RunningIndexFeature()]` in that case.

    References
    ----------
    .. [1] https://github.com/PriorLabs/tabpfn-time-series

    .. [2] https://priorlabs.ai/

    """

    allow_exog: bool = True

    def __init__(
        self,
        model_id: str,
        *,
        model: Any | None = None,
        context_length: int = 32768,
        mode: str = "local",
        point_estimate: str = "median",
        tabpfn_model_config: dict[str, Any] | None = None,
        temporal_features: list[Any] | None = None,
    ) -> None:
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            Model ID, e.g. `"priorlabs/tabpfn-ts"`.
        model : object, default None
            Pre-instantiated `TabPFNTSPipeline` instance. If `None`, a new
            instance is created lazily on the first call to `predict`.
            Intended for testing only.
        context_length : int, default 32768
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations of
            `series` (and `exog`) are stored. At `predict` time, if
            `context` is longer than `context_length` it is trimmed to
            this length before inference; if it is shorter, all available
            observations are passed as-is. Must be a positive integer.
        mode : str, default 'local'
            Inference mode. Accepted values: `'local'`, `'client'`.
        point_estimate : str, default 'median'
            Method used to aggregate the TabPFN ensemble output into the
            point forecast. Accepted values: `'mean'`, `'median'`, `'mode'`.
        tabpfn_model_config : dict, default None
            Additional configuration forwarded verbatim to the underlying
            TabPFN regressor.
        temporal_features : list, default None
            List of `FeatureGenerator` instances applied before inference.
            If `None`, TabPFN-TS uses its defaults. Pass `[]` to disable all
            temporal feature engineering.

        """

        if not isinstance(context_length, int) or context_length < 1:
            raise ValueError(
                f"`context_length` must be a positive integer. Got {context_length!r}."
            )
        if mode not in ("local", "client"):
            raise ValueError(
                f"`mode` must be 'local' or 'client'. Got {mode!r}."
            )
        if point_estimate not in ("mean", "median", "mode"):
            raise ValueError(
                f"`point_estimate` must be 'mean', 'median' or 'mode'. "
                f"Got {point_estimate!r}."
            )

        self.model_id            = model_id
        self._model              = model
        self.context_            = None
        self.context_exog_       = None
        self.context_length      = context_length
        self.mode                = mode
        self.point_estimate      = point_estimate
        self.tabpfn_model_config = dict(tabpfn_model_config) if tabpfn_model_config else {}
        self.temporal_features   = temporal_features
        self.is_fitted           = False

    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters.

        Returns
        -------
        params : dict
            Keys: `model_id`, `context_length`, `mode`, `point_estimate`,
            `tabpfn_model_config`, `temporal_features`. `tabpfn_model_config`
            is returned as `None` when no additional config was set (i.e.
            when the internal dict is empty).

        """
        return {
            "model_id":            self.model_id,
            "context_length":      self.context_length,
            "mode":                self.mode,
            "point_estimate":      self.point_estimate,
            "tabpfn_model_config": self.tabpfn_model_config or None,
            "temporal_features":   self.temporal_features,
        }

    def set_params(self, **params) -> TabPFNAdapter:
        """
        Set adapter parameters. Resets the model when any parameter changes,
        since the `TabPFNTSPipeline` is instantiated lazily on the first
        `predict` call using the current adapter state.

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `context_length`, `mode`,
            `point_estimate`, `tabpfn_model_config`, `temporal_features`.

        Returns
        -------
        self : TabPFNAdapter

        """

        valid = {
            "model_id", "context_length", "mode", "point_estimate",
            "tabpfn_model_config", "temporal_features",
        }
        invalid = set(params) - valid
        if invalid:
            raise ValueError(
                f"Invalid parameter(s) for TabPFNAdapter: {sorted(invalid)}. "
                f"Valid parameters are: {sorted(valid)}."
            )

        validated = {}
        for key, value in params.items():
            if key == "context_length":
                if not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"`context_length` must be a positive integer. Got {value!r}."
                    )
                validated[key] = value
            elif key == "mode":
                if value not in ("local", "client"):
                    raise ValueError(
                        f"`mode` must be 'local' or 'client'. Got {value!r}."
                    )
                validated[key] = value
            elif key == "point_estimate":
                if value not in ("mean", "median", "mode"):
                    raise ValueError(
                        f"`point_estimate` must be 'mean', 'median' or 'mode'. "
                        f"Got {value!r}."
                    )
                validated[key] = value
            elif key == "tabpfn_model_config":
                validated[key] = dict(value) if value else {}
            else:
                validated[key] = value

        actually_changed = {
            k: v for k, v in validated.items()
            if getattr(self, k) != v
        }
        if actually_changed:
            self._model = None
            for key, value in actually_changed.items():
                setattr(self, key, value)

        return self

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
    ) -> TabPFNAdapter:
        """
        Store the training series and optional historical exogenous variables.
        No model training occurs since TabPFN-TS is a zero-shot inference
        model.

        All input normalization and validation is performed upstream by
        `FoundationModel`; this method receives canonical dicts only.

        Parameters
        ----------
        context : dict pandas Series
            Normalized training series, one entry per series.
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series historical exogenous variables (past covariates).

        Returns
        -------
        self : TabPFNAdapter

        """

        self.context_      = context
        self.context_exog_ = context_exog
        self.is_fitted     = True

        return self

    def predict(
        self,
        steps: int,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        quantiles: list[float] | tuple[float] | None,
    ) -> dict[str, np.ndarray]:
        """
        Generate predictions using TabPFN-TS.

        All input normalization, validation, and context trimming is
        performed upstream by `FoundationModel`; this method receives
        pre-processed dicts only.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        context : dict pandas Series
            Per-series context windows (already trimmed to
            `context_length`).
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series past covariates (already trimmed).
        exog : dict pandas DataFrame, pandas Series, or None
            Per-series future covariates for the forecast horizon.
        quantiles : list of float or None
            Quantile levels to return. If `None`, a point forecast is
            produced (shape `(steps, 1)`). Accepts any float in `[0, 1]`.

        Returns
        -------
        predictions : dict
            Keys are series names. Each value is a 2-D numpy ndarray of
            shape `(steps, n_quantiles)`.

        """

        self._load_model()

        quantile_list = list(quantiles) if quantiles is not None else None
        tabpfn_quantiles = (
            quantile_list
            if quantile_list is not None
            else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )

        series_names_in = list(context.keys())

        first_series = next(iter(context.values()))
        is_datetime = isinstance(first_series.index, pd.DatetimeIndex)

        if not is_datetime:
            warnings.warn(
                "TabPFNAdapter received series with a non-DatetimeIndex. "
                "TabPFN-TS requires datetime timestamps internally; a "
                "synthetic daily DatetimeIndex (starting 2000-01-01) will be "
                "used. Calendar-based temporal features (CalendarFeature) "
                "will not be meaningful for integer-indexed data. Consider "
                "passing `temporal_features=[]` to disable calendar feature "
                "transforms.",
                # stacklevel=3: TabPFNAdapter.predict → FoundationModel.predict → user
                stacklevel=3,
            )

        context_df = self._build_context_df(
                         series_names = series_names_in,
                         context      = context,
                         context_exog = context_exog,
                         is_datetime  = is_datetime
                     )

        future_df = self._build_future_df(
                        series_names = series_names_in,
                        context      = context,
                        exog         = exog,
                        steps        = steps,
                        is_datetime  = is_datetime
                    )

        result_df = self._model.predict_df(
                        context_df = context_df,
                        future_df  = future_df,
                        quantiles  = tabpfn_quantiles,
                    )

        # result_df is a DataFrame with MultiIndex (item_id, timestamp).
        # columns: "target" (str) and quantile levels as float column names.
        predictions: dict[str, np.ndarray] = {}
        for name in series_names_in:
            group = result_df.loc[name]  # DataFrame indexed by timestamp
            if quantile_list is None:
                predictions[name] = group["target"].to_numpy().reshape(-1, 1)
            else:
                predictions[name] = group[quantile_list].to_numpy()

        return predictions

    def _load_model(self) -> None:
        """
        Load the `TabPFNTSPipeline` into `self._model` if not already set.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If `tabpfn-time-series` is not installed.

        Notes
        -----
        The pipeline is imported lazily from `tabpfn_time_series` and
        instantiated with the current adapter parameters. This method is a
        no-op when `self._model` is already populated (either by a prior
        call or by the `model` test-injection parameter).
        """

        if self._model is not None:
            return
        try:
            from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline
        except ImportError as exc:
            raise ImportError(
                "tabpfn-time-series is required for TabPFNAdapter. "
                "Install it with `pip install tabpfn-time-series`."
            ) from exc

        kwargs: dict[str, Any] = {
            "max_context_length": self.context_length,
            "tabpfn_mode": (
                TabPFNMode.LOCAL if self.mode == "local" else TabPFNMode.CLIENT
            ),
            "tabpfn_output_selection": self.point_estimate,
        }
        if self.tabpfn_model_config:
            kwargs["tabpfn_model_config"] = self.tabpfn_model_config
        if self.temporal_features is not None:
            kwargs["temporal_features"] = self.temporal_features

        self._model = TabPFNTSPipeline(**kwargs)

    def _get_timestamps(
        self, series: pd.Series, is_datetime: bool
    ) -> pd.DatetimeIndex:
        """
        Return datetime timestamps for a context series.

        For `DatetimeIndex` series the original index is returned. For
        `RangeIndex` series a synthetic daily `DatetimeIndex` starting at
        2000-01-01 is created so that TabPFN-TS's requirement for datetime
        timestamps is satisfied.

        Parameters
        ----------
        series : pandas Series
            The context series.
        is_datetime : bool
            Whether the series has a `DatetimeIndex`.

        Returns
        -------
        timestamps : pandas DatetimeIndex
            Datetime timestamps aligned with the series values.

        """

        if is_datetime:
            return series.index

        return pd.date_range("2000-01-01", periods=len(series), freq="D")

    def _get_future_timestamps(
        self, series: pd.Series, steps: int, is_datetime: bool
    ) -> pd.DatetimeIndex:
        """
        Return datetime timestamps for the forecast horizon.

        For `DatetimeIndex` series the horizon is appended at the inferred
        frequency. For `RangeIndex` series the synthetic daily timeline
        (2000-01-01 + len(context) days) is extended by `steps` days.

        Parameters
        ----------
        series : pandas Series
            The context series (used to determine the end timestamp and
            frequency).
        steps : int
            Number of steps ahead.
        is_datetime : bool
            Whether the series has a `DatetimeIndex`.

        Returns
        -------
        timestamps : pandas DatetimeIndex
            Datetime timestamps for the `steps` forecast steps.

        """

        if is_datetime:
            freq = series.index.freq
            if freq is None:
                freq = pd.tseries.frequencies.to_offset(
                    pd.infer_freq(series.index)
                )
            timestamps = pd.date_range(
                             start   = series.index[-1] + freq,
                             periods = steps,
                             freq    = freq,
                         )
        else:
            n = len(series)
            timestamps = pd.date_range(
                             start   = pd.Timestamp("2000-01-01") + pd.Timedelta(days=n),
                             periods = steps,
                             freq    = "D",
                         )

        return timestamps

    def _build_context_df(
        self,
        series_names: list,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | None] | None,
        is_datetime: bool,
    ) -> pd.DataFrame:
        """
        Build a long-format context DataFrame expected by TabPFN-TS.

        Each series' observations become rows with `item_id`, `timestamp`,
        `target`, and optional exogenous covariate columns.

        Parameters
        ----------
        series_names : list
            Ordered list of series names.
        context : dict pandas Series
            Per-series context windows.
        context_exog : dict or None
            Per-series historical exogenous variables.
        is_datetime : bool
            Whether the series have a `DatetimeIndex`.

        Returns
        -------
        context_df : pandas DataFrame
            Long-format DataFrame with columns `item_id`, `timestamp`,
            `target`, and any exogenous columns.

        """

        context_df = []
        for name in series_names:
            series = context[name]
            n = len(series)
            part = pd.DataFrame({
                "item_id":   np.full(n, name),
                "timestamp": np.asarray(self._get_timestamps(series, is_datetime)),
                "target":    series.to_numpy(dtype=float),
            })
            exog_entry = (
                context_exog.get(name) if context_exog is not None else None
            )
            if exog_entry is not None:
                part = pd.concat(
                    [part, exog_entry.reset_index(drop=True)], axis=1
                )
            context_df.append(part)

        context_df = pd.concat(context_df, ignore_index=True)

        return context_df

    def _build_future_df(
        self,
        series_names: list,
        context: dict[str, pd.Series],
        exog: dict[str, pd.DataFrame | None] | None,
        steps: int,
        is_datetime: bool,
    ) -> pd.DataFrame:
        """
        Build a long-format future DataFrame expected by TabPFN-TS.

        Each series' forecast horizon becomes rows with `item_id`,
        `timestamp`, and optional future exogenous covariate columns.

        Parameters
        ----------
        series_names : list
            Ordered list of series names.
        context : dict pandas Series
            Per-series context windows (used to derive future timestamps).
        exog : dict or None
            Per-series future exogenous variables covering the forecast
            horizon.
        steps : int
            Number of steps ahead.
        is_datetime : bool
            Whether the series have a `DatetimeIndex`.

        Returns
        -------
        future_df : pandas DataFrame
            Long-format DataFrame with columns `item_id`, `timestamp`, and
            any future exogenous columns.

        """

        future_df = []
        for name in series_names:
            series = context[name]
            part = pd.DataFrame({
                "item_id":   np.full(steps, name),
                "timestamp": np.asarray(
                    self._get_future_timestamps(series, steps, is_datetime)
                ),
            })
            future_exog = exog.get(name) if exog is not None else None
            if future_exog is not None:
                part = pd.concat(
                    [part, future_exog.reset_index(drop=True)], axis=1
                )
            future_df.append(part)

        future_df = pd.concat(future_df, ignore_index=True)

        return future_df


class T0Adapter:
    """
    Adapter for The Forecasting Company T0 foundation models.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. "theforecastingcompany/t0-alpha".
    model : T0Forecaster, default None
        Pre-loaded model instance. If `None`, the model is loaded lazily
        on the first call to `predict`.
    context_length : int, default 8192
        Maximum number of historical observations to use as context. At fit
        time only the last `context_length` observations are stored. At
        predict time, if `context` is longer than `context_length` it is
        trimmed to this length; if it is shorter, all available observations
        are used as-is. Must be a positive integer.
    device_map : str, default 'auto'
        Device placement for the model. `"auto"` selects the best
        available accelerator (CUDA > MPS > CPU). Also accepts explicit
        values such as `"cuda"`, `"mps"`, or `"cpu"`.
    torch_dtype : object, default None
        Torch dtype the loaded model is cast to (e.g. `torch.bfloat16`).
        When `None` the model keeps its default `float32` weights.

    Attributes
    ----------
    model_id : str
        HuggingFace model ID.
    context_ : dict
        Stored training series after fitting.
    context_exog_ : dict
        Stored historical exogenous variables after fitting.
    context_length : int
        Maximum number of historical observations used as context.
    device_map : str
        Device map string for model loading.
    torch_dtype : object
        Torch dtype for model loading.
    is_fitted : bool
        Whether the adapter has been fitted.

    Notes
    -----
    T0 conditions on covariates that are known over both the context and the
    forecast horizon (future-known covariates). skforecast exogenous variables
    map exactly onto this channel: their historical values (`context_exog`,
    aligned to the context) are concatenated with their future values (`exog`,
    aligned to the horizon) to form the `[context + horizon]` covariate stream
    that T0 expects. Covariates must be numeric; encode categoricals as numbers
    before passing them. A series with no future exog is forecast without
    covariates.

    References
    ----------
    .. [1] https://github.com/theforecastingcompany/tfc-t0
    
    .. [2] https://huggingface.co/theforecastingcompany/t0-alpha

    """

    allow_exog: bool = True

    def __init__(
        self,
        model_id: str,
        *,
        model: Any | None = None,
        context_length: int = 8192,
        device_map: str = "auto",
        torch_dtype: Any | None = None,
    ) -> None:
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID, e.g. "theforecastingcompany/t0-alpha".
        model : T0Forecaster, default None
            Pre-loaded model instance. If `None`, the model is loaded
            lazily on the first call to `predict`.
        context_length : int, default 8192
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations of
            `series` (and `exog`) are stored. At `predict` time, if `context`
            is longer than `context_length` it is trimmed to this length
            before inference; if it is shorter, all available observations
            are passed as-is. Must be a positive integer.
        device_map : str, default 'auto'
            Device placement for the model. `"auto"` selects the best
            available accelerator (CUDA > MPS > CPU). Also accepts explicit
            values such as `"cuda"`, `"mps"`, or `"cpu"`.
        torch_dtype : object, default None
            Torch dtype the loaded model is cast to (e.g. `torch.bfloat16`).
            When `None` the model keeps its default `float32` weights.

        """

        if not isinstance(context_length, int) or context_length < 1:
            raise ValueError(
                f"`context_length` must be a positive integer. Got {context_length!r}."
            )

        self.model_id       = model_id
        self._model         = model
        self.context_       = None
        self.context_exog_  = None
        self.context_length = context_length
        self.device_map     = device_map
        self.torch_dtype    = torch_dtype
        self.is_fitted      = False

    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters.

        Returns
        -------
        params : dict
            Keys: `model_id`, `context_length`, `device_map`, `torch_dtype`.

        """
        return {
            'model_id':       self.model_id,
            'context_length': self.context_length,
            'device_map':     self.device_map,
            'torch_dtype':    self.torch_dtype,
        }

    def set_params(self, **params) -> T0Adapter:
        """
        Set adapter parameters. Resets the model when a device, dtype, or
        model_id param changes, since those are baked into the loaded model.

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `context_length`, `device_map`,
            `torch_dtype`.

        Returns
        -------
        self : T0Adapter

        """

        valid = {'model_id', 'context_length', 'device_map', 'torch_dtype'}
        invalid = set(params) - valid
        if invalid:
            raise ValueError(
                f"Invalid parameter(s) for T0Adapter: {sorted(invalid)}. "
                f"Valid parameters are: {sorted(valid)}."
            )

        model_reset_keys = {'model_id', 'device_map', 'torch_dtype'}
        if params.keys() & model_reset_keys:
            self._model = None

        for key, value in params.items():
            if key == 'context_length':
                if not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"`context_length` must be a positive integer. Got {value!r}."
                    )
                self.context_length = value
            else:
                setattr(self, key, value)

        return self

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None],
    ) -> T0Adapter:
        """
        Store the training series and optional historical exogenous variables.
        No model training occurs since T0 is a zero-shot inference model.

        All input normalization and validation is performed upstream by
        `FoundationModel`; this method receives canonical dicts only.

        Parameters
        ----------
        context : dict pandas Series
            Normalized training series, one entry per series.
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series historical exogenous variables (past covariates).

        Returns
        -------
        self : T0Adapter

        """

        self.context_ = context
        self.context_exog_ = context_exog
        self.is_fitted = True

        return self

    def predict(
        self,
        steps: int,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None],
        exog: dict[str, pd.DataFrame | pd.Series | None],
        quantiles: list[float] | tuple[float] | None
    ) -> dict[str, np.ndarray]:
        """
        Generate predictions using the T0 model.

        All input normalization, validation, and context trimming is
        performed upstream by `FoundationModel`; this method receives
        pre-processed dicts only.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        context : dict
            Per-series context windows (already trimmed to `context_length`).
        context_exog : dict
            Per-series past covariates (already trimmed).
        exog : dict
            Per-series future covariates for the forecast horizon.
        quantiles : list of float or None
            Quantile levels to return, in the requested order. If `None`, a
            point forecast (median, quantile 0.5) is produced.

        Returns
        -------
        predictions : dict
            Keys are series names. Each value is a 2-D array of shape
            `(steps, n_quantiles)` with columns ordered to match `quantiles`.

        """

        # NOTE: the model is loaded lazily here so that the adapter can be
        # instantiated and fitted without requiring tfc-t0 to be installed.
        self._load_model()

        requested = list(quantiles) if quantiles is not None else [0.5]
        # T0 requires sorted, unique levels in (0, 1); query those, then
        # reindex the columns back to the order the caller asked for.
        query_levels = sorted(set(requested))

        series_names = list(context.keys())
        arrays = [np.asarray(context[name].to_numpy(), dtype=np.float32) for name in series_names]
        lengths = [a.shape[0] for a in arrays]
        context_length = max(lengths)

        # All series are forecast in a single batched call. Series shorter than
        # the longest are left-padded with NaN, which T0 treats as MISSING; the
        # forecast origin therefore aligns at the end of the window for every
        # series.
        context_batch = np.full((len(series_names), context_length), np.nan, dtype=np.float32)
        for row, array in zip(context_batch, arrays):
            row[context_length - array.shape[0]:] = array

        future_covariates = self._build_future_covariates(
            series_names   = series_names,
            context_exog   = context_exog,
            exog           = exog,
            context_length = context_length,
            steps          = steps,
        )

        forecast = self._model.predict(
            context           = context_batch,
            horizon           = steps,
            quantiles         = query_levels,
            future_covariates = future_covariates,
        )

        q_arr = forecast.quantiles
        if hasattr(q_arr, "detach"):
            q_arr = q_arr.detach().cpu().numpy()
        else:
            q_arr = np.asarray(q_arr)

        column_for = [query_levels.index(q) for q in requested]
        return {name: q_arr[i][:, column_for] for i, name in enumerate(series_names)}

    def _load_model(self) -> None:
        """
        Load the T0 model into `self._model` if not already set.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If `tfc-t0` is not installed.

        Notes
        -----
        The model is imported lazily from `t0` and loaded via
        `T0Forecaster.from_pretrained`, then moved to the resolved device and
        switched to eval mode. This method is a no-op when `self._model` is
        already populated.

        """

        if self._model is not None:
            return
        try:
            from t0 import T0Forecaster
        except ImportError as exc:
            raise ImportError(
                "tfc-t0 is required for T0Adapter. "
                "Install it with `pip install tfc-t0`."
            ) from exc

        device = _resolve_torch_device(self.device_map)
        model = T0Forecaster.from_pretrained(self.model_id).to(device)
        if self.torch_dtype is not None:
            model = model.to(self.torch_dtype)
        self._model = model.eval()

    def _build_future_covariates(
        self,
        series_names: list[str],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        context_length: int,
        steps: int,
    ) -> np.ndarray | None:
        """
        Assemble T0's batched `[n_series, n_covariates, context_length + steps]`
        covariate array from per-series past and future exogenous values.

        Covariate columns are pooled across all series (first-seen order). For
        each series and column the historical values (from `context_exog`) are
        placed flush against the forecast origin and the future values (from
        `exog`) cover the horizon. Every unfilled cell — a padded timestep, or a
        column/series that lacks that covariate — stays NaN, which T0 treats as
        missing.

        Parameters
        ----------
        series_names : list of str
            Series order defining the batch rows.
        context_exog : dict or None
            Per-series historical exogenous values aligned to each context.
        exog : dict or None
            Per-series future-known exogenous values covering the horizon.
        context_length : int
            Width of the (left-padded) context window.
        steps : int
            Number of forecast steps.

        Returns
        -------
        future_covariates : numpy ndarray or None
            Array of shape `(n_series, n_covariates, context_length + steps)`,
            or `None` when no series has future exog.

        """

        if exog is None:
            return None

        future_frames = {
            name: (e if isinstance(e, pd.DataFrame) else e.to_frame())
            for name, e in exog.items()
            if e is not None
        }
        if not future_frames:
            return None

        columns: list[str] = []
        for frame in future_frames.values():
            for col in frame.columns:
                if col not in columns:
                    columns.append(col)

        total_length = context_length + steps
        covariates = np.full(
            (len(series_names), len(columns), total_length), np.nan, dtype=np.float32
        )
        column_index = {col: j for j, col in enumerate(columns)}
        for row, name in enumerate(series_names):
            future_df = future_frames.get(name)
            if future_df is None:
                continue
            past_df = None
            if context_exog is not None and context_exog.get(name) is not None:
                ctx = context_exog[name]
                past_df = ctx if isinstance(ctx, pd.DataFrame) else ctx.to_frame()
            for col in future_df.columns:
                j = column_index[col]
                future_values = self._to_float_array(future_df[col])
                covariates[row, j, context_length:context_length + future_values.shape[0]] = future_values
                if past_df is not None and col in past_df.columns:
                    past_values = self._to_float_array(past_df[col])
                    covariates[row, j, context_length - past_values.shape[0]:context_length] = past_values

        return covariates

    @staticmethod
    def _to_float_array(col_data: pd.Series) -> np.ndarray:
        """
        Convert a numeric or boolean covariate column to a `float32` array.

        Parameters
        ----------
        col_data : pandas Series
            A single covariate column.

        Returns
        -------
        col_array : numpy ndarray
            1-D `float32` array.

        Raises
        ------
        ValueError
            If the column is neither numeric nor boolean. T0 only conditions
            on numeric covariates; categoricals must be encoded as numbers.

        """

        if pd.api.types.is_numeric_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
            return col_data.astype(np.float32).to_numpy()

        raise ValueError(
            f"T0Adapter supports only numeric covariates. Column "
            f"{col_data.name!r} has dtype {col_data.dtype}. Encode categorical "
            f"covariates as numeric values before passing them."
        )


class TiRexAdapter:
    """
    Adapter for NXAI TiRex-2 foundation models.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. `"NX-AI/TiRex-2"`.
    model : object, default None
        Pre-loaded `ForecastModel` instance (as returned by `tirex2.load_model`).
        If `None`, the model is loaded lazily on the first call to `predict`.
    context_length : int, default 2048
        Maximum number of historical observations to use as context. At fit
        time only the last `context_length` observations are stored. At
        predict time, if `context` is longer than `context_length` it is
        trimmed to this length; if it is shorter, all available observations
        are used as-is. Must be a positive integer. The true maximum context
        supported by a given checkpoint may differ; this default is a
        conservative value and is not enforced against the loaded checkpoint.
    device : str, default 'auto'
        Device placement for the model. `"auto"` selects the best available
        accelerator (CUDA > MPS > CPU), except that MPS is not supported by
        TiRex-2's recurrent kernels and falls back to CPU with a warning.
        Also accepts explicit values such as `"cuda"` or `"cpu"`, forwarded
        to `tirex2.load_model`.
    hf_kwargs : dict, default None
        Additional keyword arguments forwarded to `tirex2.load_model`'s
        `hf_kwargs`, which in turn forwards them to
        `huggingface_hub.snapshot_download`. Use this to pass an access
        token for the gated `NX-AI/TiRex-2` HuggingFace repo, e.g.
        `{"token": "<HF_TOKEN>"}`, if not already set via `HF_TOKEN` or
        `huggingface-cli login`.
    multivariate : bool, default False
        If `True` and multiple series are being predicted, all series are
        stacked into a single joint multivariate forecast (TiRex-2 attends
        across variates jointly), analogous to Chronos's `cross_learning`.
        If `False` (default), each series is forecast independently via its
        own single-variate `TimeseriesType`. Joint mode only accepts `exog`
        that is identical across all series (a single shared covariate
        block), since TiRex-2 attaches one covariate tensor to the whole
        multivariate group; a `ValueError` is raised if per-series exog
        differs. Ignored in single-series mode.
    batch_size : int, default 512
        Maximum number of `TimeseriesType` entries forwarded to
        `ForecastModel.forecast` per call. Forwarded directly as `batch_size`.
    forecast_kwargs : dict, default None
        Additional keyword arguments forwarded verbatim to
        `ForecastModel.forecast` (e.g. `tta_sign_flip`, `tta_diff`).

    Attributes
    ----------
    model_id : str
        HuggingFace model ID.
    context_ : dict
        Stored training series after fitting.
    context_exog_ : dict
        Stored historical exogenous variables after fitting.
    context_length : int
        Maximum number of historical observations used as context.
    device : str
        Device placement for the model.
    hf_kwargs : dict
        Additional keyword arguments forwarded to `tirex2.load_model`.
    multivariate : bool
        Whether joint multivariate forecasting is enabled.
    batch_size : int
        Maximum batch size forwarded to `ForecastModel.forecast`.
    forecast_kwargs : dict
        Additional keyword arguments forwarded to `ForecastModel.forecast`.
    is_fitted : bool
        Whether the adapter has been fitted.

    Notes
    -----
    TiRex-2 does not accept a `quantile_levels` argument: it always forecasts
    at the checkpoint's own fixed quantile grid (9 levels for `NX-AI/TiRex-2`,
    read from the loaded model at predict time). Requested `quantiles` are
    obtained by linear interpolation over this native grid (values outside
    the native range are clamped to the nearest native quantile), following
    the same interpolation approach TiRex-2 itself uses internally for its
    FEV integration.

    Covariates map onto TiRex-2's `TimeseriesType` as follows: columns
    present only in `context_exog` (never known ahead) become `past_covariates`;
    columns present in `exog` (future-known) become `future_covariates`,
    built by concatenating their historical values (from `context_exog`, if
    present) with their future values (from `exog`) into a single
    `[context_length + steps]` stream, since TiRex-2's `future_covariates`
    tensor must span the context plus the forecast horizon. Covariates must
    be numeric; encode categoricals as numbers before passing them.

    The `tirex-2` package requires Python `>=3.11,<3.14` and is only tested
    on Linux and macOS (its `flashrnn`/`xlstm` dependencies target CUDA
    kernels; MPS is not supported, see `device` above). The pretrained
    weights on HuggingFace (`NX-AI/TiRex-2`) are a gated repository: users
    must accept the model terms and authenticate (`huggingface-cli login`
    or an `HF_TOKEN`/`hf_kwargs={"token": ...}`) before the first `predict`
    call can download them.

    References
    ----------
    .. [1] https://github.com/NX-AI/tirex-2

    .. [2] https://huggingface.co/NX-AI/TiRex-2

    .. [3] https://arxiv.org/abs/2607.01204

    """

    allow_exog: bool = True

    def __init__(
        self,
        model_id: str,
        *,
        model: Any | None = None,
        timeseries_cls: Any | None = None,
        context_length: int = 2048,
        device: str = "auto",
        hf_kwargs: dict[str, Any] | None = None,
        multivariate: bool = False,
        batch_size: int = 512,
        forecast_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialise the adapter.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID, e.g. `"NX-AI/TiRex-2"`.
        model : object, default None
            Pre-loaded `ForecastModel` instance. If `None`, the model is
            loaded lazily on the first call to `predict`.
        timeseries_cls : type, default None
            `tirex2.TimeseriesType` class used to build model inputs. If
            `None`, imported lazily from `tirex2` on the first call to
            `predict`. Intended for testing only.
        context_length : int, default 2048
            Maximum number of historical observations to retain as context.
            At `fit` time only the last `context_length` observations of
            `series` (and `exog`) are stored. At `predict` time, if
            `context` is longer than `context_length` it is trimmed to
            this length before inference; if it is shorter, all available
            observations are passed as-is. Must be a positive integer.
        device : str, default 'auto'
            Device placement for the model. `"auto"` selects the best
            available accelerator (CUDA > MPS > CPU), falling back from MPS
            to CPU with a warning since TiRex-2 does not support it. Also
            accepts explicit values such as `"cuda"` or `"cpu"`.
        hf_kwargs : dict, default None
            Additional keyword arguments forwarded to `tirex2.load_model`'s
            `hf_kwargs` (in turn forwarded to
            `huggingface_hub.snapshot_download`), e.g. to pass an access
            token for the gated model repo.
        multivariate : bool, default False
            If `True`, multiple series are stacked into a single joint
            multivariate forecast instead of being forecast independently.
        batch_size : int, default 512
            Maximum number of `TimeseriesType` entries forwarded to
            `ForecastModel.forecast` per call.
        forecast_kwargs : dict, default None
            Additional keyword arguments forwarded verbatim to
            `ForecastModel.forecast`.

        """

        if not isinstance(context_length, int) or context_length < 1:
            raise ValueError(
                f"`context_length` must be a positive integer. Got {context_length!r}."
            )
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(
                f"`batch_size` must be a positive integer. Got {batch_size!r}."
            )

        self.model_id        = model_id
        self._model          = model
        self._timeseries_cls = timeseries_cls
        self.context_        = None
        self.context_exog_   = None
        self.context_length  = context_length
        self.device          = device
        self.hf_kwargs       = dict(hf_kwargs) if hf_kwargs else {}
        self.multivariate    = multivariate
        self.batch_size      = batch_size
        self.forecast_kwargs = dict(forecast_kwargs) if forecast_kwargs else {}
        self.is_fitted       = False

    def get_params(self) -> dict:
        """
        Return the adapter's constructor parameters.

        Returns
        -------
        params : dict
            Keys: `model_id`, `context_length`, `device`, `hf_kwargs`,
            `multivariate`, `batch_size`, `forecast_kwargs`.

        """
        return {
            'model_id':        self.model_id,
            'context_length':  self.context_length,
            'device':          self.device,
            'hf_kwargs':       self.hf_kwargs or None,
            'multivariate':    self.multivariate,
            'batch_size':      self.batch_size,
            'forecast_kwargs': self.forecast_kwargs or None,
        }

    def set_params(self, **params) -> TiRexAdapter:
        """
        Set adapter parameters. Resets the model when `model_id`, `device`,
        or `hf_kwargs` changes, since those are baked into the loaded model.

        Parameters
        ----------
        **params :
            Valid keys: `model_id`, `context_length`, `device`, `hf_kwargs`,
            `multivariate`, `batch_size`, `forecast_kwargs`.

        Returns
        -------
        self : TiRexAdapter

        """

        valid = {
            'model_id', 'context_length', 'device', 'hf_kwargs',
            'multivariate', 'batch_size', 'forecast_kwargs',
        }
        invalid = set(params) - valid
        if invalid:
            raise ValueError(
                f"Invalid parameter(s) for TiRexAdapter: {sorted(invalid)}. "
                f"Valid parameters are: {sorted(valid)}."
            )

        model_reset_keys = {'model_id', 'device', 'hf_kwargs'}
        if params.keys() & model_reset_keys:
            self._model = None

        for key, value in params.items():
            if key == 'context_length':
                if not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"`context_length` must be a positive integer. Got {value!r}."
                    )
                self.context_length = value
            elif key == 'batch_size':
                if not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"`batch_size` must be a positive integer. Got {value!r}."
                    )
                self.batch_size = value
            elif key == 'hf_kwargs':
                self.hf_kwargs = dict(value) if value else {}
            elif key == 'forecast_kwargs':
                self.forecast_kwargs = dict(value) if value else {}
            else:
                setattr(self, key, value)

        return self

    def fit(
        self,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None],
    ) -> TiRexAdapter:
        """
        Store the training series and optional historical exogenous variables.
        No model training occurs since TiRex-2 is a zero-shot inference model.

        All input normalization and validation is performed upstream by
        `FoundationModel`; this method receives canonical dicts only.

        Parameters
        ----------
        context : dict pandas Series
            Normalized training series, one entry per series.
        context_exog : dict pandas DataFrame, pandas Series, or None
            Per-series historical exogenous variables (past covariates).

        Returns
        -------
        self : TiRexAdapter

        """

        self.context_ = context
        self.context_exog_ = context_exog
        self.is_fitted = True

        return self

    def predict(
        self,
        steps: int,
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        quantiles: list[float] | tuple[float] | None,
    ) -> dict[str, np.ndarray]:
        """
        Generate predictions using the TiRex-2 model.

        All input normalization, validation, and context trimming is
        performed upstream by `FoundationModel`; this method receives
        pre-processed dicts only.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        context : dict
            Per-series context windows (already trimmed to
            `context_length`).
        context_exog : dict
            Per-series past covariates (already trimmed).
        exog : dict
            Per-series future covariates for the forecast horizon.
        quantiles : list of float or None
            Quantile levels to return. If `None`, only the median (0.5) is
            produced. Levels outside TiRex-2's native quantile grid are
            obtained by linear interpolation.

        Returns
        -------
        predictions : dict
            Keys are series names. Each value is a 2-D array of shape
            `(steps, n_quantiles)`.

        Raises
        ------
        ValueError
            If `multivariate=True`, more than one series is being
            predicted, and `context_exog` or `exog` differ across series
            (TiRex-2 attaches a single shared covariate block to a joint
            multivariate group, so per-series exog cannot be represented),
            or if the series do not all share the same context length.

        """

        # NOTE: the model and TimeseriesType class are loaded lazily here so
        # that the adapter can be instantiated and fitted without requiring
        # tirex-2 to be installed.
        self._load_model()
        timeseries_cls = self._get_timeseries_cls()

        series_names_in = list(context.keys())
        query_levels = list(quantiles) if quantiles is not None else [0.5]

        use_joint = self.multivariate and len(series_names_in) > 1
        if use_joint:
            self._validate_shared_exog(series_names_in, context_exog, exog)
            timeseries_list = [
                self._build_joint_timeseries(
                    series_names_in, context, context_exog, exog, steps, timeseries_cls
                )
            ]
        else:
            timeseries_list = [
                self._build_series_timeseries(
                    series         = context[name],
                    context_exog   = context_exog[name] if context_exog is not None else None,
                    exog           = exog[name] if exog is not None else None,
                    steps          = steps,
                    timeseries_cls = timeseries_cls,
                )
                for name in series_names_in
            ]

        raw_forecasts = self._model.forecast(
            timeseries        = timeseries_list,
            prediction_length = steps,
            output_type       = "numpy",
            batch_size        = self.batch_size,
            **self.forecast_kwargs,
        )
        native_levels = self._native_quantile_levels()

        predictions: dict[str, np.ndarray] = {}
        if use_joint:
            arr = raw_forecasts[0]  # shape (V_t, Q, H)
            for i, name in enumerate(series_names_in):
                predictions[name] = self._select_quantiles(arr[i].T, native_levels, query_levels)
        else:
            for name, arr in zip(series_names_in, raw_forecasts):
                predictions[name] = self._select_quantiles(arr[0].T, native_levels, query_levels)

        return predictions

    def _load_model(self) -> None:
        """
        Load the TiRex-2 `ForecastModel` into `self._model` if not already set.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If `tirex-2` is not installed.

        Notes
        -----
        The model is imported lazily from `tirex2` and loaded via
        `tirex2.load_model`. `device="auto"` resolves to the best available
        accelerator, falling back from MPS to CPU with a warning (TiRex-2's
        recurrent kernels do not support MPS). This method is a no-op when
        `self._model` is already populated.

        """

        if self._model is not None:
            return
        try:
            from tirex2 import load_model
        except ImportError as exc:
            raise ImportError(
                "tirex-2 is required for TiRexAdapter. "
                "Install it with `pip install tirex-2`."
            ) from exc

        resolved_device = _resolve_torch_device(self.device)
        if resolved_device == "mps":
            warnings.warn(
                "MPS device is not supported by TiRex-2 (its recurrent "
                "kernels require CUDA). Falling back to CPU.",
                stacklevel=6,
            )
            resolved_device = "cpu"

        self._model = load_model(
            self.model_id, device=resolved_device, hf_kwargs=self.hf_kwargs or {}
        )

    def _get_timeseries_cls(self) -> Any:
        """
        Return `tirex2.TimeseriesType`, importing it lazily if not already
        cached or injected via the `timeseries_cls` constructor argument.

        Returns
        -------
        timeseries_cls : type

        Raises
        ------
        ImportError
            If `tirex-2` is not installed.

        """

        if self._timeseries_cls is not None:
            return self._timeseries_cls
        try:
            from tirex2 import TimeseriesType
        except ImportError as exc:
            raise ImportError(
                "tirex-2 is required for TiRexAdapter. "
                "Install it with `pip install tirex-2`."
            ) from exc
        self._timeseries_cls = TimeseriesType
        return self._timeseries_cls

    def _native_quantile_levels(self) -> list[float]:
        """
        Return the loaded model's native quantile levels as clean floats.

        Returns
        -------
        levels : list of float
            The checkpoint's fixed quantile grid (e.g. 9 levels for
            `NX-AI/TiRex-2`), rounded to 6 decimals to remove float32 noise.

        """

        q = self._model.quantiles
        if hasattr(q, "detach"):
            q = q.detach().cpu().numpy()
        else:
            q = np.asarray(q)

        return [round(float(x), 6) for x in q]

    @staticmethod
    def _select_quantiles(
        values: np.ndarray,
        native_levels: list[float],
        query_levels: list[float],
    ) -> np.ndarray:
        """
        Linearly interpolate a `(steps, n_native)` quantile array onto
        arbitrary `query_levels`, clamping at the edges of `native_levels`.

        Parameters
        ----------
        values : numpy ndarray
            Array of shape `(steps, n_native)` holding TiRex-2's native
            quantile forecast.
        native_levels : list of float
            TiRex-2's native quantile levels, matching `values`' last axis,
            sorted ascending.
        query_levels : list of float
            Quantile levels to produce.

        Returns
        -------
        result : numpy ndarray
            Array of shape `(steps, len(query_levels))`.

        """

        native = np.asarray(native_levels, dtype=float)
        result = np.empty((values.shape[0], len(query_levels)), dtype=np.float64)
        for i, level in enumerate(query_levels):
            exact = np.where(np.isclose(native, level, atol=1e-9))[0]
            if exact.size:
                result[:, i] = values[:, exact[0]]
            elif level <= native[0]:
                result[:, i] = values[:, 0]
            elif level >= native[-1]:
                result[:, i] = values[:, -1]
            else:
                hi = int(np.searchsorted(native, level))
                lo = hi - 1
                weight = (level - native[lo]) / (native[hi] - native[lo])
                result[:, i] = (1.0 - weight) * values[:, lo] + weight * values[:, hi]

        return result

    @staticmethod
    def _to_float_array(col_data: pd.Series) -> np.ndarray:
        """
        Convert a numeric or boolean covariate column to a `float32` array.

        Parameters
        ----------
        col_data : pandas Series
            A single covariate column.

        Returns
        -------
        col_array : numpy ndarray
            1-D `float32` array.

        Raises
        ------
        ValueError
            If the column is neither numeric nor boolean. TiRex-2 only
            conditions on numeric covariates; categoricals must be encoded
            as numbers.

        """

        if pd.api.types.is_numeric_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
            return col_data.astype(np.float32).to_numpy()

        raise ValueError(
            f"TiRexAdapter supports only numeric covariates. Column "
            f"{col_data.name!r} has dtype {col_data.dtype}. Encode categorical "
            f"covariates as numeric values before passing them."
        )

    @classmethod
    def _covariate_tensors(
        cls,
        context_length: int,
        steps: int,
        context_exog: pd.DataFrame | pd.Series | None,
        exog: pd.DataFrame | pd.Series | None,
    ) -> tuple[Any, Any]:
        """
        Build the `(past_covariates, future_covariates)` tensors expected by
        `TimeseriesType` for one series (or one shared covariate block in
        joint multivariate mode).

        Columns present only in `context_exog` become `past_covariates`
        (shape `[V_p, context_length]`). Columns present in `exog` become
        `future_covariates` (shape `[V_f, context_length + steps]`), built
        by concatenating their historical values (from `context_exog`, if
        present, NaN-filled otherwise) with their future values.

        Parameters
        ----------
        context_length : int
            Length of the context window.
        steps : int
            Number of forecast steps.
        context_exog : pandas DataFrame, pandas Series, default None
            Historical exogenous variables aligned to the context.
        exog : pandas DataFrame, pandas Series, default None
            Future-known exogenous variables covering the forecast horizon.

        Returns
        -------
        past_covariates : torch Tensor or None
        future_covariates : torch Tensor or None

        """

        import torch

        past_df = None
        if context_exog is not None:
            past_df = (
                context_exog if isinstance(context_exog, pd.DataFrame)
                else context_exog.to_frame()
            )
        future_df = None
        if exog is not None:
            future_df = exog if isinstance(exog, pd.DataFrame) else exog.to_frame()

        future_known_cols = list(future_df.columns) if future_df is not None else []
        past_only_cols = [
            col for col in (past_df.columns if past_df is not None else [])
            if col not in future_known_cols
        ]

        past_covariates = None
        if past_only_cols:
            arr = np.stack(
                [cls._to_float_array(past_df[col]) for col in past_only_cols]
            )
            past_covariates = torch.as_tensor(arr, dtype=torch.float32)

        future_covariates = None
        if future_known_cols:
            total_length = context_length + steps
            arr = np.full((len(future_known_cols), total_length), np.nan, dtype=np.float32)
            for j, col in enumerate(future_known_cols):
                future_values = cls._to_float_array(future_df[col])
                arr[j, context_length:context_length + future_values.shape[0]] = future_values
                if past_df is not None and col in past_df.columns:
                    past_values = cls._to_float_array(past_df[col])
                    arr[j, context_length - past_values.shape[0]:context_length] = past_values
            future_covariates = torch.as_tensor(arr, dtype=torch.float32)

        return past_covariates, future_covariates

    @classmethod
    def _build_series_timeseries(
        cls,
        series: pd.Series,
        context_exog: pd.DataFrame | pd.Series | None,
        exog: pd.DataFrame | pd.Series | None,
        steps: int,
        timeseries_cls: Any,
    ) -> Any:
        """
        Build a single-variate `TimeseriesType` for one series.

        Parameters
        ----------
        series : pandas Series
            The series' context window.
        context_exog : pandas DataFrame, pandas Series, default None
            Historical exogenous variables aligned to `series`.
        exog : pandas DataFrame, pandas Series, default None
            Future-known exogenous variables covering the forecast horizon.
        steps : int
            Number of forecast steps.
        timeseries_cls : type
            `tirex2.TimeseriesType` class (or a test double with the same
            `target`/`past_covariates`/`future_covariates` fields).

        Returns
        -------
        timeseries : tirex2.TimeseriesType
            `target` has shape `[1, context_length]`.

        """

        import torch

        target = torch.as_tensor(
            series.to_numpy(dtype=np.float32), dtype=torch.float32
        ).unsqueeze(0)
        past_covariates, future_covariates = cls._covariate_tensors(
            context_length = target.shape[-1],
            steps          = steps,
            context_exog   = context_exog,
            exog           = exog,
        )

        return timeseries_cls(
            target=target, past_covariates=past_covariates, future_covariates=future_covariates
        )

    @classmethod
    def _build_joint_timeseries(
        cls,
        series_names: list[str],
        context: dict[str, pd.Series],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        steps: int,
        timeseries_cls: Any,
    ) -> Any:
        """
        Build a single joint multivariate `TimeseriesType` stacking every
        series in `series_names`.

        Parameters
        ----------
        series_names : list of str
            Series order defining the stacking (variate) order.
        context : dict
            Per-series context windows. All must share the same length.
        context_exog : dict or None
            Per-series historical exogenous variables. Already validated
            (by `_validate_shared_exog`) to be identical across series.
        exog : dict or None
            Per-series future-known exogenous variables. Already validated
            to be identical across series.
        steps : int
            Number of forecast steps.
        timeseries_cls : type
            `tirex2.TimeseriesType` class (or a test double with the same
            `target`/`past_covariates`/`future_covariates` fields).

        Returns
        -------
        timeseries : tirex2.TimeseriesType
            `target` has shape `[len(series_names), context_length]`.

        Raises
        ------
        ValueError
            If series do not all share the same context length.

        """

        import torch

        lengths = {len(context[name]) for name in series_names}
        if len(lengths) > 1:
            raise ValueError(
                "`multivariate=True` requires all series to share the same "
                f"context length. Got lengths {sorted(lengths)}."
            )

        target = torch.as_tensor(
            np.stack([context[name].to_numpy(dtype=np.float32) for name in series_names]),
            dtype=torch.float32,
        )
        first = series_names[0]
        past_covariates, future_covariates = cls._covariate_tensors(
            context_length = target.shape[-1],
            steps          = steps,
            context_exog   = context_exog[first] if context_exog is not None else None,
            exog           = exog[first] if exog is not None else None,
        )

        return timeseries_cls(
            target=target, past_covariates=past_covariates, future_covariates=future_covariates
        )

    @staticmethod
    def _validate_shared_exog(
        series_names: list[str],
        context_exog: dict[str, pd.DataFrame | pd.Series | None] | None,
        exog: dict[str, pd.DataFrame | pd.Series | None] | None,
    ) -> None:
        """
        Validate that per-series exog is identical across all series, as
        required to attach a single shared covariate block in joint
        multivariate mode.

        Parameters
        ----------
        series_names : list of str
            Series being jointly forecast.
        context_exog : dict or None
            Per-series historical exogenous variables.
        exog : dict or None
            Per-series future-known exogenous variables.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `context_exog` or `exog` differ across any two series.

        """

        for label, values in (("context_exog", context_exog), ("exog", exog)):
            if values is None:
                continue
            first_name = series_names[0]
            first = values.get(first_name)
            first_df = None
            if first is not None:
                first_df = first if isinstance(first, pd.DataFrame) else first.to_frame()
            for name in series_names[1:]:
                other = values.get(name)
                other_df = None
                if other is not None:
                    other_df = other if isinstance(other, pd.DataFrame) else other.to_frame()
                mismatched = (
                    (first_df is None) != (other_df is None)
                    or (first_df is not None and not first_df.equals(other_df))
                )
                if mismatched:
                    raise ValueError(
                        f"`multivariate=True` requires `{label}` to be identical "
                        "across all series (a single shared covariate block), "
                        "because TiRex-2's TimeseriesType attaches one covariate "
                        f"tensor to the whole multivariate group. Series "
                        f"'{first_name}' and '{name}' have differing `{label}`. "
                        "Use `multivariate=False` (default) for per-series exog."
                    )


_ADAPTER_REGISTRY: dict[str, type] = {
    "amazon/chronos":    ChronosAdapter,
    "autogluon/chronos": ChronosAdapter,
    "google/timesfm":    TimesFMAdapter,
    "Salesforce/moirai": MoiraiAdapter,
    "soda-inria/tabicl": TabICLAdapter,
    "priorlabs/tabpfn":  TabPFNAdapter,
    "theforecastingcompany/t0": T0Adapter,
    "NX-AI/TiRex-2":     TiRexAdapter,
    # "ibm/TTM": TTMAdapter,
}


def _resolve_adapter(model_id: str) -> type:
    """
    Return the adapter class for *model_id* based on prefix matching.

    Parameters
    ----------
    model_id : str
        The model ID for which to find the adapter class.

    Returns
    -------
    adapter_cls : type
        The adapter class corresponding to the given model ID.

    """

    for prefix, cls in _ADAPTER_REGISTRY.items():
        if model_id.startswith(prefix):
            return cls
    
    raise ValueError(
        f"No adapter found for model '{model_id}'. "
        f"Registered prefixes: {list(_ADAPTER_REGISTRY)}."
    )
