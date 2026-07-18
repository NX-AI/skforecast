# TiRex-2 adapter — source analysis (Step 1)

Findings from reading the actual current source of both codebases (skforecast fork
after sync to `upstream/master` @ v0.23.0+18, and `NX-AI/tirex-2` @ v0.1.1, cloned
directly rather than trusted from the README/GitHub UI). This supersedes the
speculative framing in the original brief wherever the two disagree.

## 1. Shared vs. per-variate covariates in `TimeseriesType`

`TimeseriesType` (`src/tirex2/model/types.py`) is:

```python
@dataclass
class TimeseriesType:
    target: torch.Tensor             # [V_t, T]
    past_covariates: torch.Tensor | None    # [V_p, T]
    future_covariates: torch.Tensor | None  # [V_f, >=T+H]
```

Confirmed: covariates are **one shared block per `TimeseriesType` instance**, not
per-target-variate. If `target` stacks several series into one multivariate group
(`V_t > 1`), there is exactly one `past_covariates` tensor and one
`future_covariates` tensor for the *whole* group — there is no mechanism to say
"variate 0 gets covariate A, variate 1 gets covariate B."

This means skforecast's per-series `context_exog`/`exog` dicts (where each series
can have its own distinct column set) cannot be attached correctly once multiple
series are stacked into one joint multivariate `TimeseriesType`. Consequently the
adapter must default to **independent mode**: one `TimeseriesType` per series
(`target` shape `[1, T]`), submitted together as a list to `model.forecast(...)` for
batched compute but *not* jointly attended. Joint multivariate mode
(`multivariate=True`, opt-in, analogous to Chronos's `cross_learning`) stacks
several series into a single `TimeseriesType` and is only well-defined when either
(a) no exog is used, or (b) exog is a single shared DataFrame broadcast to every
series (same columns, same values) — skforecast already represents this case
distinctly from a per-series dict, so it's a cheap check. If per-series exog with
differing schemas is supplied together with `multivariate=True`, the adapter raises
rather than silently dropping or misattributing covariates.

A second, more precise discrepancy surfaced here (relevant to the adapter's
input-building code, not just this question): `future_covariates` is not a
horizon-only tensor — its declared shape is `[V_f, >=T+H]`, i.e. it must span the
**context plus the horizon**, concatenated. This matches skforecast's existing
`T0Adapter._build_future_covariates` pattern (context values + future values
concatenated into one `[..., context_length + steps]` array) much more closely than
`ChronosAdapter._build_chronos_input`, where `future_covariates` is a
`prediction_length`-only dict. TiRex-2 additionally keeps a genuinely separate
`past_covariates` channel (columns only ever observed historically, never
known ahead) — a distinction neither T0 nor Chronos draws in quite the same way
(Chronos's `past_covariates`/`future_covariates` are two independently-windowed
dicts, not one context+horizon-spanning stream). So `TiRexAdapter`'s input builder
is a hybrid: columns present only in `context_exog` map to `past_covariates`
(context-length only, Chronos-style); columns present in `exog` (future-known) map
to `future_covariates` as `context_exog[col] concat exog[col]` (context+horizon,
T0-style), NaN-filled where a series lacks historical values for that column.

## 2. Quantile handling

`ForecastModel.forecast()` (`src/tirex2/api_adapter/forecast.py`) takes no
`quantile_levels` argument at all. The returned tensor's quantile axis is always
the checkpoint's own fixed grid, read from the model's `quantiles` buffer
(`TiRex2.quantiles`, set at construction from `model-config.yaml`). There is no way
to request arbitrary quantiles natively — this puts TiRex-2 in the same category as
TimesFM/Moirai (fixed grid) rather than Chronos/TabICL (arbitrary quantiles), which
contradicts nothing in the brief but does resolve its open question.

The exact grid could not be verified against the real `NX-AI/TiRex-2` checkpoint
config, because the weights are gated on HuggingFace and downloading
`model-config.yaml` requires an accepted-terms HF token, which is out of scope for
this analysis. However:
- The README's usage example annotates the output shape as
  `(n_targets, 9 quantiles, prediction_length)`.
- The repo's own test fixture (`conftest.py`, `_build_model`) constructs a
  full-size model with `QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`
  — evenly spaced deciles, matching the 9-quantile claim exactly and consistent
  with the convention skforecast already uses for TimesFM/Moirai's
  `SUPPORTED_QUANTILES`.
- `quantiles` is a genuine constructor argument of `TiRex2`, so in principle a
  different checkpoint variant (e.g. `-gifteval-zs`) could ship a different grid.
  The adapter should therefore read `model.quantiles` at load time rather than
  hardcoding `[0.1 ... 0.9]`, even though the default checkpoint almost certainly
  uses that grid.

The package does ship a private linear-interpolation helper
(`_interpolate_quantile_levels` in `api_adapter/forecast.py`, edge-clamped, used
only internally for FEV-format output) but it is not exported
(`tirex2.__all__ == ["load_model", "ForecastModel", "TimeseriesType", "TiRex2"]`).
`TiRexAdapter` should implement its own equivalent linear interpolation (same
edge-clamping behavior) over `model.quantiles` rather than importing a private
symbol, and document in the docstring that requested quantiles outside the native
grid are interpolated, not natively modeled — mirroring how TimesFM/Moirai document
their *lack* of interpolation, but doing the opposite (TiRex-2 interpolates instead
of raising) since NX-AI's own FEV integration establishes interpolation as the
intended way to serve arbitrary quantiles from this model.

## 3. Other discrepancies found

- **License**: confirmed directly — `LICENSE` is the standard Apache License 2.0
  text, and `NOTICE` reads `Copyright 2026 NXAI GmbH`, licensed under Apache-2.0.
  No `license_accepted`-style gate is needed for the code, unlike TiRex v1's
  Community License. This part of the brief's premise holds.
- **HF weights are still gated**, independent of the code license: `NX-AI/TiRex-2`
  on HuggingFace requires either `huggingface-cli login` or an `HF_TOKEN` with
  accepted terms (documented in the tirex-2 README itself). This is a runtime
  concern for users loading real weights, not a packaging/license concern.
  skforecast's existing pattern — lazy-import inside `predict`, let the backend
  library raise its own error — already surfaces this adequately (huggingface_hub's
  `snapshot_download` raises a clear error on an unauthenticated gated request), so
  no special-cased gate is needed in `TiRexAdapter`, only a docstring note.
- **Streaming is not part of this package.** The brief's premise ("operates in a
  streaming fashion as new observations arrive") describes **TiRex-2 Pro**, a
  separate closed offering explicitly called out in the README: *"This repository
  is our open-source release. Our pro version extends TiRex-2 with streaming,
  hardware-optimized inference..., finetuning, and classification & regression
  support."* The open-source `tirex-2` package we are integrating has no
  incremental/streaming API at all — not even one that's simply unused by
  skforecast's batch-call model. There is nothing to reconcile here; the design
  question dissolves rather than needing an answer.
- **Packaging convention was misremembered in the brief.** None of the existing
  foundation-model backends (`chronos-forecasting`, `timesfm`, `uni2ts`, `tabicl`,
  `tabpfn-time-series`, `tfc-t0`) are declared anywhere in `pyproject.toml`'s
  `[project.optional-dependencies]` — grepped and confirmed absent. They are only
  ever documented as manual `pip install <package>` instructions (in the user
  guide, the `FoundationModel` docstring's References section, and the
  `foundation-forecasting` skill). `tirex-2` should follow the same real
  convention: a documented install instruction, not a new pyproject.toml extra
  with no precedent among its six siblings.
- **Testing convention needs no HF token.** Every existing adapter test file uses a
  hand-written `Fake*` stand-in for the real backend (`FakePipeline`,
  `FakeT0Forecaster`, `FakeMoirai2Forecast`, ...) — none of them touch the network
  or download real weights. `TiRexAdapter`'s tests will follow the same pattern
  with a `FakeForecastModel`, so the HF-gating concern raised in the brief does not
  actually affect CI at all.
- **Model IDs**: confirmed real HF repo IDs referenced in the tirex-2 repo's own
  examples: `NX-AI/TiRex-2` (primary), `NX-AI/TiRex-2-gifteval-pretrain`,
  `NX-AI/TiRex-2-gifteval-zs`, `NX-AI/TiRex-2-fevbench`. All four share the
  `NX-AI/TiRex-2` prefix, so a single `_ADAPTER_REGISTRY` entry
  (`"NX-AI/TiRex-2": TiRexAdapter`) resolves all of them via the existing
  `startswith`-prefix mechanism — no special-casing required. (One inconsistent,
  unhyphenated `NX-AI/TiRex2` default appears in
  `examples/fevbench/submission.py`; it looks like a stale typo rather than a
  second real naming convention and is not being registered.)
- **Tighter platform/version constraint than existing adapters**: `tirex-2`
  requires Python `>=3.11,<3.14` and depends on `flashrnn`/`xlstm`, which the
  README says are "currently only tested on Linux and macOS" (no Windows). This is
  materially narrower than skforecast's other foundation-model backends and is
  worth calling out explicitly in the adapter docstring / install docs so it isn't
  a surprise.
- `model.predict()`/`_predict_once` already logs a warning and truncates when
  `prediction_length` exceeds the checkpoint's configured `future_len`, so unlike
  `TimesFMAdapter` the new adapter does not need to enforce its own `max_horizon`
  ValueError — the underlying library's own behavior is sufficient and simpler to
  rely on.

## 4. Refined questions for issue #1190

1. *Original framing*: "TiRex needs a `license_accepted` gate because of its
   Community License." *Refined*: TiRex-2's code is Apache-2.0 (verified directly
   against `LICENSE`/`NOTICE`), so no `license_accepted` gate is needed for the
   package itself. Its pretrained weights on HuggingFace (`NX-AI/TiRex-2`) remain a
   *gated* repo requiring an accepted-terms HF token, independent of the code
   license. Given skforecast's existing adapters already rely on the backend
   library's own error surfacing for missing installs/auth (no adapter does
   upfront credential checks today), is that pattern considered sufficient for a
   gated-weights-but-permissively-licensed-code backend, or would the maintainers
   want an explicit, adapter-level check/informative message for this specific
   combination — since it hasn't come up among the existing six adapters
   (Chronos-2, TimesFM, Moirai-2, TabICL, TabPFN-TS, T0)?
2. *Original framing*: "Does TiRex-2's native multivariate/covariate support map
   cleanly onto skforecast's per-series exog model?" *Refined*: TiRex-2's
   `TimeseriesType` carries exactly one shared covariate block per (possibly
   multivariate) group — it has no mechanism to assign different covariates to
   different target variates within one joint multivariate call. Since
   skforecast's exog model is fundamentally per-series (a dict keyed by series
   name, each with potentially distinct columns), we plan to ship joint
   multivariate forecasting as an opt-in (`multivariate=True`) that only accepts
   exog when it is either absent or a single schema shared across all series,
   raising otherwise, with independent per-series forecasting remaining the
   default. Does that match how skforecast's maintainers would want a new adapter
   to expose a genuinely joint-multivariate-capable backend, given none of the
   existing six adapters have had to make this particular trade-off?
