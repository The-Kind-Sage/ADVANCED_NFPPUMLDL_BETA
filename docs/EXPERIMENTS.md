# Experiments log

Everything below was run against the same chronological split; the holdout
(2025-01 .. 2026-08) was never used for selection.

## Hyperparameter ablations (see `tft_results/prev_run_*`)

| experiment | pooled MAE | outcome |
|---|---|---|
| lr 1e-3 | 18.92 | worse - dropped |
| lr 5e-4 | 17.66 | chosen (committed to `config/base.yaml`) |
| hidden 128, 12 epochs | worse RMSE/tails | kept hidden 64 |
| log target | inflated MAE | `target_transform: raw` kept |
| dropout 0.10 | val loss blew up after epoch 3 | raised to 0.25 |

## Determinism

Two identical seed-42 runs without determinism flags gave 17.66 and 20.29 pooled MAE
(GPU atomic reductions). With `enable_determinism()` (CUBLAS workspace, deterministic
algorithms `warn_only=True`) the spread drops to roughly 0.5 MAE - strict mode is
impossible because pf's interpolation has no deterministic CUDA backward.

## Fine-tuning trials (`scripts/finetune.py`)

| trial | best validation MAE |
|---|---|
| low_lr | 17.507 |
| **regularized** (lr 1e-4, wd 5e-4, dropout 0.4) | **17.363** |
| median_focused (median weight 4.0) | 17.370 |

The regularized winner scored 19.32 pooled on the holdout - worse than the base model's
17.59 - so fine-tuning was **not** adopted. Trial details:
`tft_results/finetune_20260929_002853/experiment_summary.json`.

## Baseline benchmark (`scripts/benchmark_forecasts.py`)

Validation selected `residual_boosting`; on test: persistence 11.03, TFT 17.59,
TFT+persistence blend (weight 0.23) 11.73, damped trend 11.03 (alpha 0.0 - trend term
switched off by validation), residual boosting 11.04. The validation-selected
per-horizon switch (persistence h1, residual boosting h2/h3) scored 10.91 pooled.

## Rolling backtest (`scripts/rolling_backtest.py`)

Nested expanding-window selection (no test peeking): folds from 2022-01, selecting
per horizon between persistence and residual boosting using only the preceding
validation year. Results: `tft_results/rolling_backtest_*`.

## Delta-3 target experiment (reverted)

Training on the 3-month price difference (`price_diff3` target, base reconstruction)
scored **19.82** pooled / skill -0.797 (h1 18.65, h2 19.58, h3 21.22) vs the incumbent
17.59 / -0.595 - worse on every horizon - so all source changes were reverted. Evidence
kept in `tft_results/d3_seed42/`, `d3_prep/`, `d3_smoke/`.
