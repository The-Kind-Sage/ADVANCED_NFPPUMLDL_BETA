# Reproducing

## Environment

- Python 3.13, Windows, RTX 4050 6 GB
- `torch 2.14.0+cu126`, `pytorch-forecasting 1.8.0`, `lightning 2.6.6`
- Install: `pip install -e .` (or `pip install -r requirements.txt`)

## Pipeline

```bash
python main.py --mode prep     # clean panel -> tft_results/processed_monthly_panel.csv
python main.py --mode train    # ~4 epochs, checkpoints + training_history.csv
python main.py --mode eval     # holdout metrics, plots, final report (no training)
python main.py --mode smoke    # 2-epoch sanity run into tft_results/smoke/
```

Expected `eval` headline: pooled MAE 17.59, skill -0.595, n = 40,491
(`tft_results/metrics.json`).

## Splits (fixed in `config/base.yaml`)

- train: 2001-04 .. 2023-12
- validation: 2024-01 .. 2024-12
- holdout: 2025-01 .. 2026-08 (never used for selection)

## Determinism

`utils.enable_determinism()` sets `CUBLAS_WORKSPACE_CONFIG`, deterministic algorithms
(`warn_only=True`), fixed cudnn flags and TF32. `warn_only` is required: pf's
`TimeDistributedInterpolation` has no deterministic CUDA backward. Residual run-to-run
spread is ~0.5 pooled MAE (vs ~2.6 without the flags). Exact bit-reproducibility across
driver/library versions is not guaranteed.

## Experiments

```bash
python scripts/benchmark_forecasts.py   # baseline strategies on validation + test
python scripts/rolling_backtest.py      # expanding-window per-horizon selection
python scripts/finetune.py              # 3 short fine-tuning trials
```

Each writes a timestamped folder under `tft_results/`. Checkpoints and the processed
panel CSV are git-ignored (regenerate with `prep`/`train`); metrics, reports and plots
are committed.

## Tests

```bash
python -m pytest
```
