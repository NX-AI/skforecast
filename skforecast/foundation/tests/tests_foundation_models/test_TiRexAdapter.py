# Unit test TiRexAdapter
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.foundation._adapters import TiRexAdapter
from .fixtures_adapters import (
    y, exog, y_wide, y_dict, exog_shared,
    FakeTimeseriesType, FakeForecastModel,
    prepare_fit_args, prepare_predict_args
)


# ==============================================================================
# Tests TiRexAdapter.__init__
# ==============================================================================
def test_TiRexAdapter_init_default_params():
    """
    Test that default parameter values are set correctly and class-level
    attributes are properly initialised.
    """
    adapter = TiRexAdapter(model_id="NX-AI/TiRex-2")
    assert adapter.model_id == "NX-AI/TiRex-2"
    assert adapter.context_length == 2048
    assert adapter.device == "auto"
    assert adapter.hf_kwargs == {}
    assert adapter.multivariate is False
    assert adapter.batch_size == 512
    assert adapter.forecast_kwargs == {}
    assert adapter._model is None
    assert adapter._timeseries_cls is None
    assert adapter.context_ is None
    assert adapter.context_exog_ is None
    assert adapter.is_fitted is False
    assert TiRexAdapter.allow_exog is True


def test_TiRexAdapter_init_custom_params_stored():
    """
    Test that custom constructor parameters are stored correctly.
    """
    adapter = TiRexAdapter(
        model_id="NX-AI/TiRex-2-gifteval-zs",
        context_length=512,
        device="cpu",
        hf_kwargs={"token": "abc"},
        multivariate=True,
        batch_size=16,
        forecast_kwargs={"tta_sign_flip": True},
    )
    assert adapter.context_length == 512
    assert adapter.device == "cpu"
    assert adapter.hf_kwargs == {"token": "abc"}
    assert adapter.multivariate is True
    assert adapter.batch_size == 16
    assert adapter.forecast_kwargs == {"tta_sign_flip": True}


@pytest.mark.parametrize(
    "context_length",
    [0, -1, None, 1.5, "not_int"],
    ids=lambda cl: f"context_length={cl}"
)
def test_TiRexAdapter_init_ValueError_when_context_length_invalid(context_length):
    """
    Test that __init__ raises ValueError for non-positive-integer
    context_length values.
    """
    with pytest.raises(ValueError, match=re.escape("`context_length` must be a positive integer")):
        TiRexAdapter(model_id="NX-AI/TiRex-2", context_length=context_length)


@pytest.mark.parametrize(
    "batch_size",
    [0, -1, None, 1.5, "not_int"],
    ids=lambda bs: f"batch_size={bs}"
)
def test_TiRexAdapter_init_ValueError_when_batch_size_invalid(batch_size):
    """
    Test that __init__ raises ValueError for non-positive-integer
    batch_size values.
    """
    with pytest.raises(ValueError, match=re.escape("`batch_size` must be a positive integer")):
        TiRexAdapter(model_id="NX-AI/TiRex-2", batch_size=batch_size)


# ==============================================================================
# Tests TiRexAdapter.get_params / set_params
# ==============================================================================
def test_TiRexAdapter_get_params_returns_expected_keys_and_values():
    """
    Test that get_params returns all expected keys with the values set at
    construction, and that hf_kwargs/forecast_kwargs are None when empty.
    """
    adapter = TiRexAdapter(
        model_id="NX-AI/TiRex-2", context_length=512, multivariate=True
    )
    params = adapter.get_params()
    assert set(params.keys()) == {
        "model_id", "context_length", "device", "hf_kwargs",
        "multivariate", "batch_size", "forecast_kwargs",
    }
    assert params["model_id"] == "NX-AI/TiRex-2"
    assert params["context_length"] == 512
    assert params["multivariate"] is True
    assert params["hf_kwargs"] is None  # empty dict -> None
    assert params["forecast_kwargs"] is None  # empty dict -> None


@pytest.mark.parametrize(
    "params, match",
    [
        ({"context_length": 0}, "`context_length` must be a positive integer"),
        ({"context_length": -1}, "`context_length` must be a positive integer"),
        ({"batch_size": 0}, "`batch_size` must be a positive integer"),
        ({"unknown_param": 42}, "Invalid parameter"),
    ],
    ids=["context_length=0", "context_length=-1", "batch_size=0", "unknown_param"]
)
def test_TiRexAdapter_set_params_ValueError_when_invalid(params, match):
    """
    Test that set_params raises ValueError for invalid parameter values or
    unknown parameter names.
    """
    adapter = TiRexAdapter(model_id="NX-AI/TiRex-2")
    with pytest.raises(ValueError, match=re.escape(match)):
        adapter.set_params(**params)


@pytest.mark.parametrize(
    "param, value, resets_model",
    [
        ("model_id", "NX-AI/TiRex-2-fevbench", True),
        ("device", "cpu", True),
        ("hf_kwargs", {"token": "abc"}, True),
        ("multivariate", True, False),
        ("context_length", 128, False),
        ("batch_size", 8, False),
        ("forecast_kwargs", {"tta_diff": True}, False),
    ],
    ids=lambda x: str(x)
)
def test_TiRexAdapter_set_params_updates_and_resets_model(
    param, value, resets_model
):
    """
    Test that set_params updates the given parameter and resets _model
    only when model_id, device, or hf_kwargs changes. Returns self.
    """
    adapter = TiRexAdapter(
        model_id="NX-AI/TiRex-2", model=FakeForecastModel()
    )
    assert adapter._model is not None
    result = adapter.set_params(**{param: value})
    assert result is adapter
    if resets_model:
        assert adapter._model is None
    else:
        assert adapter._model is not None


def test_TiRexAdapter_set_params_empty_dicts_normalise_to_empty_dict():
    """
    Test that passing hf_kwargs=None/forecast_kwargs=None via set_params
    normalises the internal value to an empty dict.
    """
    adapter = TiRexAdapter(
        model_id="NX-AI/TiRex-2",
        hf_kwargs={"token": "abc"},
        forecast_kwargs={"tta_diff": True},
    )
    adapter.set_params(hf_kwargs=None, forecast_kwargs=None)
    assert adapter.hf_kwargs == {}
    assert adapter.forecast_kwargs == {}


# ==============================================================================
# Tests TiRexAdapter.fit
# ==============================================================================
def test_TiRexAdapter_fit_output_single_series():
    """
    Test fit on a single series: returns self, sets is_fitted=True, and
    stores history trimmed to context_length.
    """
    context_length = 10
    adapter = TiRexAdapter(model_id="NX-AI/TiRex-2", context_length=context_length)
    context, context_exog = prepare_fit_args(y, context_length=context_length)
    result = adapter.fit(context=context, context_exog=context_exog)

    assert result is adapter
    assert adapter.is_fitted is True
    hist = next(iter(adapter.context_.values()))
    assert len(hist) == context_length
    pd.testing.assert_series_equal(hist, y.iloc[-context_length:])


@pytest.mark.parametrize(
    "series_input",
    [y_wide, y_dict],
    ids=["wide_dataframe", "dict"]
)
def test_TiRexAdapter_fit_output_multi_series(series_input):
    """
    Test fit on multi-series input: sets is_fitted and stores a dict of
    Series keyed by series names.
    """
    context_length = 10
    adapter = TiRexAdapter(model_id="NX-AI/TiRex-2", context_length=context_length)
    context, context_exog = prepare_fit_args(series_input, context_length=context_length)
    adapter.fit(context=context, context_exog=context_exog)

    assert adapter.is_fitted is True
    assert set(adapter.context_.keys()) == {"s1", "s2"}
    for name, s in adapter.context_.items():
        assert isinstance(s, pd.Series)
        assert len(s) == context_length


def test_TiRexAdapter_fit_exog_stored():
    """
    Test that exog is stored in context_exog_, and that no exog maps to None.
    """
    adapter = TiRexAdapter(model_id="NX-AI/TiRex-2")

    context, context_exog = prepare_fit_args(y, exog=exog)
    adapter.fit(context=context, context_exog=context_exog)
    hist_exog = next(iter(adapter.context_exog_.values()))
    pd.testing.assert_frame_equal(hist_exog, exog)

    ctx2, ctx_exog2 = prepare_fit_args(y)
    adapter.fit(context=ctx2, context_exog=ctx_exog2)
    assert adapter.context_exog_ is None


# ==============================================================================
# Tests TiRexAdapter.predict — independent mode
# ==============================================================================
def test_TiRexAdapter_predict_point_forecast_single_series():
    """
    Test point forecast (quantiles=None) on a single series: returns the
    median (0.5) for every step.
    """
    fake_model = FakeForecastModel()
    adapter = TiRexAdapter(
        model_id="NX-AI/TiRex-2", model=fake_model, timeseries_cls=FakeTimeseriesType
    )
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=12)
    raw = adapter.predict(
        steps=12, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )

    assert isinstance(raw, dict)
    assert list(raw.keys()) == ["sales"]
    arr = raw["sales"]
    assert arr.shape == (12, 1)
    np.testing.assert_array_almost_equal(arr[:, 0], np.full(12, 0.5))


def test_TiRexAdapter_predict_quantile_forecast_native_levels():
    """
    Test quantile forecast at exactly TiRex-2's native levels: returned
    values equal the requested level with no interpolation error.
    """
    quantiles = [0.1, 0.5, 0.9]
    fake_model = FakeForecastModel()
    adapter = TiRexAdapter(
        model_id="NX-AI/TiRex-2", model=fake_model, timeseries_cls=FakeTimeseriesType
    )
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=5)
    raw = adapter.predict(
        steps=5, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=quantiles
    )

    arr = raw["sales"]
    assert arr.shape == (5, 3)
    for i, q in enumerate(quantiles):
        np.testing.assert_array_almost_equal(arr[:, i], np.full(5, q))


def test_TiRexAdapter_predict_quantile_interpolation_between_native_levels():
    """
    Test that a quantile level not present in the native grid is linearly
    interpolated between its two neighboring native levels.
    """
    fake_model = FakeForecastModel(quantiles=(0.1, 0.2, 0.5, 0.8, 0.9))
    adapter = TiRexAdapter(
        model_id="NX-AI/TiRex-2", model=fake_model, timeseries_cls=FakeTimeseriesType
    )
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=3)
    raw = adapter.predict(
        steps=3, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=[0.15]
    )
    # Halfway between native 0.1 and 0.2 -> 0.15
    np.testing.assert_array_almost_equal(raw["sales"][:, 0], np.full(3, 0.15))


def test_TiRexAdapter_predict_quantile_interpolation_clamps_outside_range():
    """
    Test that requesting quantiles outside the native grid's range clamps
    to the nearest native level instead of extrapolating.
    """
    fake_model = FakeForecastModel(quantiles=(0.1, 0.5, 0.9))
    adapter = TiRexAdapter(
        model_id="NX-AI/TiRex-2", model=fake_model, timeseries_cls=FakeTimeseriesType
    )
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=2)
    raw = adapter.predict(
        steps=2, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=[0.01, 0.99]
    )
    np.testing.assert_array_almost_equal(raw["sales"][:, 0], np.full(2, 0.1))
    np.testing.assert_array_almost_equal(raw["sales"][:, 1], np.full(2, 0.9))


@pytest.mark.parametrize(
    "series_input",
    [y_wide, y_dict],
    ids=["wide_dataframe", "dict"]
)
def test_TiRexAdapter_predict_independent_multi_series(series_input):
    """
    Test that multi-series prediction in independent mode (default) submits
    one single-variate TimeseriesType per series and returns one forecast
    per series.
    """
    fake_model = FakeForecastModel()
    adapter = TiRexAdapter(
        model_id="NX-AI/TiRex-2", model=fake_model, timeseries_cls=FakeTimeseriesType
    )
    ctx, ctx_exog = prepare_fit_args(series_input)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=4)
    raw = adapter.predict(
        steps=4, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )

    assert set(raw.keys()) == {"s1", "s2"}
    for name in ["s1", "s2"]:
        assert raw[name].shape == (4, 1)
        np.testing.assert_array_almost_equal(raw[name][:, 0], np.full(4, 0.5))

    # Independent mode -> two single-variate TimeseriesType entries submitted
    assert len(fake_model.last_timeseries) == 2
    for ts in fake_model.last_timeseries:
        assert ts.target.shape[0] == 1


def test_TiRexAdapter_predict_forwards_batch_size_and_forecast_kwargs():
    """
    Test that batch_size and forecast_kwargs are forwarded verbatim to
    ForecastModel.forecast.
    """
    fake_model = FakeForecastModel()
    adapter = TiRexAdapter(
        model_id="NX-AI/TiRex-2", model=fake_model, timeseries_cls=FakeTimeseriesType,
        batch_size=7, forecast_kwargs={"tta_sign_flip": True}
    )
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=3)
    adapter.predict(
        steps=3, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )
    assert fake_model.last_batch_size == 7
    assert fake_model.last_kwargs.get("tta_sign_flip") is True
    assert fake_model.last_prediction_length == 3


# ==============================================================================
# Tests TiRexAdapter.predict — joint multivariate mode
# ==============================================================================
def test_TiRexAdapter_predict_joint_multivariate_stacks_series():
    """
    Test that multivariate=True stacks all series into a single joint
    TimeseriesType and splits the shared forecast back per series.
    """
    fake_model = FakeForecastModel()
    adapter = TiRexAdapter(
        model_id="NX-AI/TiRex-2", model=fake_model, timeseries_cls=FakeTimeseriesType,
        multivariate=True
    )
    ctx, ctx_exog = prepare_fit_args(y_dict)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=4)
    raw = adapter.predict(
        steps=4, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=[0.1, 0.5, 0.9]
    )

    assert set(raw.keys()) == {"s1", "s2"}
    for name in ["s1", "s2"]:
        assert raw[name].shape == (4, 3)

    # Joint mode -> exactly one multivariate TimeseriesType submitted
    assert len(fake_model.last_timeseries) == 1
    assert fake_model.last_timeseries[0].target.shape == (2, 30)


def test_TiRexAdapter_predict_multivariate_ignored_for_single_series():
    """
    Test that multivariate=True has no effect when only one series is
    being predicted (falls back to independent single-variate mode).
    """
    fake_model = FakeForecastModel()
    adapter = TiRexAdapter(
        model_id="NX-AI/TiRex-2", model=fake_model, timeseries_cls=FakeTimeseriesType,
        multivariate=True
    )
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=3)
    adapter.predict(
        steps=3, context=ctx_p, context_exog=ctx_exog_p,
        exog=exog_p, quantiles=None
    )
    assert len(fake_model.last_timeseries) == 1
    assert fake_model.last_timeseries[0].target.shape[0] == 1


def test_TiRexAdapter_predict_joint_multivariate_raises_on_mismatched_context_length():
    """
    Test that joint multivariate mode raises ValueError when series have
    different context lengths (cannot stack into one tensor).
    """
    fake_model = FakeForecastModel()
    adapter = TiRexAdapter(
        model_id="NX-AI/TiRex-2", model=fake_model, timeseries_cls=FakeTimeseriesType,
        multivariate=True
    )
    context = {
        "s1": y_dict["s1"],
        "s2": y_dict["s2"].iloc[:-5],
    }
    ctx_exog = {"s1": None, "s2": None}

    with pytest.raises(ValueError, match=re.escape("same context length")):
        adapter.predict(
            steps=3, context=context, context_exog=ctx_exog, exog=None, quantiles=None
        )


def test_TiRexAdapter_predict_joint_multivariate_raises_on_mismatched_exog():
    """
    Test that joint multivariate mode raises ValueError when per-series
    exog differs across series, since TiRex-2 attaches a single shared
    covariate block to the whole multivariate group.
    """
    fake_model = FakeForecastModel()
    adapter = TiRexAdapter(
        model_id="NX-AI/TiRex-2", model=fake_model, timeseries_cls=FakeTimeseriesType,
        multivariate=True
    )
    context_exog = {
        "s1": pd.DataFrame({"feat": np.zeros(30)}, index=y_dict["s1"].index),
        "s2": pd.DataFrame({"feat": np.ones(30)}, index=y_dict["s2"].index),
    }

    with pytest.raises(ValueError, match=re.escape("multivariate=True` requires `context_exog`")):
        adapter.predict(
            steps=3, context=y_dict, context_exog=context_exog, exog=None, quantiles=None
        )


def test_TiRexAdapter_predict_joint_multivariate_allows_identical_shared_exog():
    """
    Test that joint multivariate mode succeeds when per-series exog is
    identical across all series (a genuinely shared covariate block).
    """
    fake_model = FakeForecastModel()
    adapter = TiRexAdapter(
        model_id="NX-AI/TiRex-2", model=fake_model, timeseries_cls=FakeTimeseriesType,
        multivariate=True
    )
    context_exog = {
        "s1": exog_shared.copy(),
        "s2": exog_shared.copy(),
    }

    raw = adapter.predict(
        steps=3, context=y_dict, context_exog=context_exog, exog=None, quantiles=None
    )
    assert set(raw.keys()) == {"s1", "s2"}
    assert len(fake_model.last_timeseries) == 1


# ==============================================================================
# Tests TiRexAdapter.predict — covariate forwarding
# ==============================================================================
def test_TiRexAdapter_predict_past_only_covariate_maps_to_past_covariates():
    """
    Test that a covariate present only in context_exog (never known ahead)
    is forwarded as past_covariates, sized to context_length only.
    """
    fake_model = FakeForecastModel()
    adapter = TiRexAdapter(
        model_id="NX-AI/TiRex-2", model=fake_model, timeseries_cls=FakeTimeseriesType
    )
    ctx, ctx_exog = prepare_fit_args(y, exog=exog)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=6)
    adapter.predict(
        steps=6, context=ctx_p, context_exog=ctx_exog_p, exog=exog_p, quantiles=None
    )

    ts = fake_model.last_timeseries[0]
    assert ts.past_covariates is not None
    assert ts.past_covariates.shape == (2, len(ctx_p["sales"]))
    assert ts.future_covariates is None


def test_TiRexAdapter_predict_future_known_covariate_spans_context_plus_horizon():
    """
    Test that a covariate known into the future is forwarded as
    future_covariates, concatenating its historical values (context_exog)
    with its future values (exog) into one [context_length + steps] stream.
    """
    fake_model = FakeForecastModel()
    adapter = TiRexAdapter(
        model_id="NX-AI/TiRex-2", model=fake_model, timeseries_cls=FakeTimeseriesType
    )
    ctx_exog = pd.DataFrame({"promo": np.zeros(50)}, index=y.index)
    ctx, ctx_exog_dict = prepare_fit_args(y, exog=ctx_exog)
    adapter.fit(context=ctx, context_exog=ctx_exog_dict)

    future_idx = pd.date_range(y.index[-1] + y.index.freq, periods=6, freq=y.index.freq)
    future = pd.DataFrame({"promo": np.ones(6)}, index=future_idx)
    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=6, exog=future)
    adapter.predict(
        steps=6, context=ctx_p, context_exog=ctx_exog_p, exog=exog_p, quantiles=None
    )

    ts = fake_model.last_timeseries[0]
    assert ts.past_covariates is None
    assert ts.future_covariates is not None
    context_length = len(ctx_p["sales"])
    assert ts.future_covariates.shape == (1, context_length + 6)
    values = ts.future_covariates.numpy()[0]
    np.testing.assert_array_almost_equal(values[:context_length], np.zeros(context_length))
    np.testing.assert_array_almost_equal(values[context_length:], np.ones(6))


def test_TiRexAdapter_predict_no_exog_no_covariates():
    """
    Test that when neither context_exog nor exog is provided, both
    past_covariates and future_covariates are None.
    """
    fake_model = FakeForecastModel()
    adapter = TiRexAdapter(
        model_id="NX-AI/TiRex-2", model=fake_model, timeseries_cls=FakeTimeseriesType
    )
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=5)
    adapter.predict(
        steps=5, context=ctx_p, context_exog=ctx_exog_p, exog=exog_p, quantiles=None
    )
    ts = fake_model.last_timeseries[0]
    assert ts.past_covariates is None
    assert ts.future_covariates is None


def test_TiRexAdapter_predict_non_numeric_covariate_raises_ValueError():
    """
    Test that a non-numeric covariate column raises ValueError, since
    TiRex-2 only conditions on numeric covariates.
    """
    fake_model = FakeForecastModel()
    adapter = TiRexAdapter(
        model_id="NX-AI/TiRex-2", model=fake_model, timeseries_cls=FakeTimeseriesType
    )
    bad_exog = pd.DataFrame({"weather": ["sunny"] * 50}, index=y.index)
    ctx, ctx_exog = prepare_fit_args(y, exog=bad_exog)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=3)
    with pytest.raises(ValueError, match=re.escape("supports only numeric covariates")):
        adapter.predict(
            steps=3, context=ctx_p, context_exog=ctx_exog_p, exog=exog_p, quantiles=None
        )


# ==============================================================================
# Tests TiRexAdapter._load_model
# ==============================================================================
def test_TiRexAdapter_load_model_ImportError_when_tirex2_not_installed():
    """
    Test that predict raises a clear ImportError when tirex-2 is not
    installed and no model was injected.
    """
    adapter = TiRexAdapter(model_id="NX-AI/TiRex-2")
    ctx, ctx_exog = prepare_fit_args(y)
    adapter.fit(context=ctx, context_exog=ctx_exog)

    ctx_p, ctx_exog_p, exog_p = prepare_predict_args(adapter, steps=3)
    with pytest.raises(ImportError, match=re.escape("tirex-2 is required")):
        adapter.predict(
            steps=3, context=ctx_p, context_exog=ctx_exog_p, exog=exog_p, quantiles=None
        )
