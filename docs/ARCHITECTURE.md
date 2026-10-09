# Architecture

## Entry point

`main.py` dispatches one of four modes to `scripts/`:

| mode | script | what it does |
|---|---|---|
| `prep` | `scripts/prep.py` | load CSV -> clean monthly panel, split chronologically |
| `train` | `scripts/train.py` | build datasets, train TFT, save checkpoints + history |
| `eval` | `scripts/eval.py` | load newest checkpoint, score holdout, write reports/plots |
| `smoke` | `scripts/smoke.py` | tiny 2-epoch end-to-end run into `tft_results/smoke/` |

## Packages

- `config/` - `base.yaml` (paths, splits, hyperparameters, TFT variable groups) exposed as constants.
- `data/` - `loader.py` (find CSV, `load_panel`, `split_panel`), `preprocessing.py`
  (monthly grid, gap handling, lags/rolling/seasonality features), `dataset.py`
  (pytorch-forecasting `TimeSeriesDataSet` builder with per-series normalizer).
- `models/tft.py` - TFT construction, `WeightedQuantileLoss` (median emphasised),
  training loop wiring.
- `training/` - `trainer.py` (Lightning trainer), `callbacks.py` (live per-epoch
  MAE/RMSE reporter + history persistence), `logger.py`.
- `evaluation/` - `metrics.py` (reg metrics, aligned baselines, skill), `evaluate.py`
  (holdout scoring + interval calibration), `forecasting.py` (future months),
  `per_commodity.py`.
- `reporting/` - `report_writer.py` (final text report), `plots.py` (all PNG output).
- `utils/` - `compat.py` (pytorch-forecasting/lightning version shims, de-normalisation),
  `helpers.py` (formatting, determinism flags, device pick).

## Data flow

```
WFP csv -> preprocess (monthly grid + features)
        -> chronological split (train < 2024 < test 2025)
        -> TimeSeriesDataSet (24-month encoder, 3-month decoder)
        -> TFT training (checkpoints, per-epoch history)
        -> evaluate (predictions, baselines, skill, plots, final report)
```

## Evaluation rules

- Holdout windows are never touched during training or model selection.
- Baselines score the exact same windows and rows as the TFT (aligned baselines).
- Metric of record: pooled MAE over all 3 horizons vs the persistence baseline.

## Tests

`tests/` covers pure logic only (no CSV, no GPU): metric helpers, formatting helpers,
compat de-normalisation and chronological split boundaries. Run with `python -m pytest`
(`[tool.pytest.ini_options]` in `pyproject.toml` adds `tests/` to the path).
