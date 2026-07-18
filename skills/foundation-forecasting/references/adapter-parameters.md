# Foundation Adapter Parameters

`FoundationModel` resolves the adapter automatically from `model_id`. All keyword arguments passed to `FoundationModel(...)` beyond `model_id` are forwarded to the chosen adapter's `__init__`.

```python
from skforecast.foundation import FoundationModel

model = FoundationModel(
    model_id='autogluon/chronos-2-small',
    context_length=2048,
    device_map='auto',
)
```

## ChronosAdapter — Amazon Chronos-2

- **`model_id` prefix**: `autogluon/chronos`
- **`allow_exog`**: `True` (past and future covariates)
- **Quantiles**: any value in `(0, 1)`

| Parameter        | Type    | Default  | Description                                                                    |
|------------------|---------|----------|--------------------------------------------------------------------------------|
| `model_id`       | str     | —        | HuggingFace model ID (e.g. `autogluon/chronos-2-small`).                       |
| `pipeline`       | object  | `None`   | Pre-loaded `BaseChronosPipeline`. If `None`, loaded lazily on first `predict`. |
| `context_length` | int     | `8192`   | Max historical observations kept as context.                                   |
| `predict_kwargs` | dict    | `None`   | Extra kwargs forwarded to the pipeline's `predict_quantiles`.                  |
| `device_map`     | str     | `'auto'` | Device placement: `'auto'` (CUDA > MPS > CPU), `'cuda'`, `'mps'`, `'cpu'`.     |
| `torch_dtype`    | object  | `None`   | Torch dtype for `from_pretrained` (e.g. `torch.bfloat16`).                     |
| `cross_learning` | bool    | `False`  | If `True`, shares information across series in multi-series batches.           |

## TimesFMAdapter — Google TimesFM 2.5

- **`model_id` prefix**: `google/timesfm`
- **`allow_exog`**: `False`
- **Supported quantiles**: `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`

| Parameter                | Type | Default | Description                                                         |
|--------------------------|------|---------|---------------------------------------------------------------------|
| `model_id`               | str  | —       | HuggingFace model ID (e.g. `google/timesfm-2.5-200m-pytorch`).      |
| `model`                  | obj  | `None`  | Pre-loaded & compiled TimesFM model. If `None`, loaded lazily.      |
| `context_length`         | int  | `512`   | Max historical observations kept as context.                        |
| `max_horizon`            | int  | `512`   | Max forecast horizon. `predict(steps=...)` must be ≤ this.          |
| `forecast_config_kwargs` | dict | `None`  | Extra kwargs forwarded to `timesfm.ForecastConfig` at compile time. |

The model is compiled lazily for the exact requested `steps` (up to `max_horizon`) to avoid unnecessary decode iterations.

## MoiraiAdapter — Salesforce Moirai-2

- **`model_id` prefix**: `Salesforce/moirai`
- **`allow_exog`**: `False`
- **Supported quantiles**: `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`

| Parameter        | Type | Default  | Description                                                              |
|------------------|------|----------|--------------------------------------------------------------------------|
| `model_id`       | str  | —        | HuggingFace model ID (e.g. `Salesforce/moirai-2.0-R-small`).             |
| `module`         | obj  | `None`   | Pre-loaded `Moirai2Module`. If `None`, loaded lazily.                    |
| `context_length` | int  | `2048`   | Max historical observations kept as context.                             |
| `device`         | str  | `'auto'` | Device placement: `'auto'` (CUDA > MPS > CPU), `'cuda'`, `'mps'`, `'cpu'`. |

## TabICLAdapter — Soda-INRIA TabICL

- **`model_id` prefix**: `soda-inria/tabicl`
- **`allow_exog`**: `True` (past and future covariates)
- **Quantiles**: any value in `(0, 1)`

| Parameter            | Type  | Default  | Description                                                                      |
|----------------------|-------|----------|----------------------------------------------------------------------------------|
| `model_id`           | str   | —        | HuggingFace model ID (e.g. `soda-inria/tabicl`).                                 |
| `model`              | obj   | `None`   | Pre-instantiated `TabICLForecaster`. If `None`, created lazily on first predict. |
| `context_length`     | int   | `4096`   | Max historical observations kept as context.                                     |
| `point_estimate`     | str   | `'mean'` | Point forecast method: `'mean'` or `'median'`.                                   |
| `tabicl_config`      | dict  | `None`   | Extra kwargs forwarded to `TabICLRegressor` at inference time.                   |
| `temporal_features`  | list  | `None`   | `TimeTransform` instances applied before inference. `None` = TabICL defaults; `[]` = disable all. |

## TabPFNAdapter — Prior Labs TabPFN-TS

- **`model_id` prefix**: `priorlabs/tabpfn`
- **`allow_exog`**: `True` (known-future covariates; covariates without future values are discarded by the library)
- **Quantiles**: any value in `(0, 1)`

| Parameter             | Type  | Default    | Description                                                                      |
|-----------------------|-------|------------|----------------------------------------------------------------------------------|
| `model_id`            | str   | —          | Model ID (e.g. `priorlabs/tabpfn-ts`). Used only for adapter resolution.         |
| `model`               | obj   | `None`     | Pre-instantiated `TabPFNTSPipeline`. If `None`, created lazily on first predict. |
| `context_length`      | int   | `32768`    | Max historical observations kept as context. Lower (e.g. 4096) for faster inference. |
| `mode`                | str   | `'local'`  | `'local'` (on-device inference, CUDA > MPS > CPU) or `'client'` (Prior Labs cloud API, no GPU needed). |
| `point_estimate`      | str   | `'median'` | Ensemble aggregation for the point forecast: `'mean'`, `'median'` or `'mode'`.   |
| `tabpfn_model_config` | dict  | `None`     | Extra config forwarded to the underlying TabPFN regressor (e.g. `model_path`, `device`). |
| `temporal_features`   | list  | `None`     | `FeatureGenerator` instances applied before inference. `None` = TabPFN-TS defaults; `[]` = disable all. |

## T0Adapter — The Forecasting Company T0

- **`model_id` prefix**: `theforecastingcompany/t0`
- **`allow_exog`**: `True` (future-known covariates; historical and known-future values are concatenated into the `[context + horizon]` covariate stream)
- **Quantiles**: any value in `(0, 1)` (native levels `0.1, 0.25, 0.5, 0.75, 0.9`; other levels are produced by inference-time interpolation)

| Parameter        | Type   | Default  | Description                                                                |
|------------------|--------|----------|----------------------------------------------------------------------------|
| `model_id`       | str    | —        | HuggingFace model ID (e.g. `theforecastingcompany/t0-alpha`).             |
| `model`          | obj    | `None`   | Pre-loaded `T0Forecaster`. If `None`, loaded lazily on first `predict`.    |
| `context_length` | int    | `8192`   | Max historical observations kept as context.                               |
| `device_map`     | str    | `'auto'` | Device placement: `'auto'` (CUDA > MPS > CPU), `'cuda'`, `'mps'`, `'cpu'`. |
| `torch_dtype`    | object | `None`   | Torch dtype the loaded model is cast to (e.g. `torch.bfloat16`).           |

Point forecasts use the median (quantile `0.5`). Covariates must be numeric; encode categoricals as numbers before passing them. A series with no future exog is forecast without covariates.

## TiRexAdapter — NX-AI TiRex-2

- **`model_id` prefix**: `NX-AI/TiRex-2` (also matches `NX-AI/TiRex-2-gifteval-zs`, `NX-AI/TiRex-2-gifteval-pretrain`, `NX-AI/TiRex-2-fevbench`)
- **`allow_exog`**: `True` — columns present only in `context_exog` map to TiRex-2's `past_covariates` channel; columns present in `exog` (future-known) map to `future_covariates`, built by concatenating their historical and future values into one `[context_length + steps]` stream
- **Quantiles**: any value in `(0, 1)` (native levels `[0.1, 0.2, ..., 0.9]`; other levels are produced by linear interpolation, clamped at the edges)
- **Multivariate**: natively multivariate — `multivariate=True` stacks all series into one joint `TimeseriesType` (TiRex-2 attends across variates jointly); requires `exog` (if any) to be identical across all series, since one covariate block is shared by the whole group. Default `multivariate=False` forecasts each series independently.

| Parameter         | Type   | Default  | Description                                                                |
|-------------------|--------|----------|----------------------------------------------------------------------------|
| `model_id`        | str    | —        | HuggingFace model ID (e.g. `NX-AI/TiRex-2`).                              |
| `model`           | obj    | `None`   | Pre-loaded `ForecastModel`. If `None`, loaded lazily on first `predict`.   |
| `context_length`  | int    | `2048`   | Max historical observations kept as context (not enforced against the checkpoint's true capacity). |
| `device`          | str    | `'auto'` | Device placement: `'auto'` (CUDA > MPS > CPU, MPS falls back to CPU with a warning since TiRex-2 requires CUDA), `'cuda'`, `'cpu'`. |
| `hf_kwargs`       | dict   | `None`   | Forwarded to `tirex2.load_model`'s `hf_kwargs` (in turn to `huggingface_hub.snapshot_download`), e.g. an access token for the gated model repo. |
| `multivariate`    | bool   | `False`  | If `True`, jointly forecast multiple series (see above). Ignored in single-series mode. |
| `batch_size`      | int    | `512`    | Max `TimeseriesType` entries forwarded to `ForecastModel.forecast` per call. |
| `forecast_kwargs` | dict   | `None`   | Extra kwargs forwarded to `ForecastModel.forecast` (e.g. `tta_sign_flip`, `tta_diff`). |

The pretrained weights on HuggingFace (`NX-AI/TiRex-2`) are a gated repository: authenticate with `huggingface-cli login` or an `HF_TOKEN`/`hf_kwargs={"token": ...}` before the first `predict` call. The `tirex-2` package requires Python `>=3.11,<3.14` and is only tested on Linux and macOS.

## Common Behavior

All adapters implement the same minimal interface:

- `fit(series, exog=None)` — stores context and metadata; no training.
- `predict(steps, context, context_exog, exog, quantiles)` — returns a   `dict[str, np.ndarray]` of shape `(steps, n_quantiles)` keyed by series name.
- `get_params()` / `set_params(**kwargs)` — sklearn-style parameter access.

Backend libraries (`chronos-forecasting`, `timesfm`, `uni2ts`, `tabicl`, `tabpfn-time-series`, `tfc-t0`) are imported **lazily** inside the adapter method that needs them, so only the backend for the adapter you actually use needs to be installed.
