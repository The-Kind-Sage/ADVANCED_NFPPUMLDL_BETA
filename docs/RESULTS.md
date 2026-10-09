# Results

Final numbers for the best checkpoint (`epoch=2-step=645-v5.ckpt`), holdout window
2025-01-15 .. 2026-08-15, 13,497 forecast windows x 3 horizons = 40,491 predictions,
prices in NPR.

## Headline

| model | pooled MAE | RMSE | MAPE | R2 |
|---|---|---|---|---|
| **TFT** | **17.59** | 31.38 | 12.10% | 0.9411 |
| persistence | 11.03 | 25.13 | 8.50% | 0.9622 |
| seasonal naive | 21.40 | 37.77 | 13.12% | 0.9147 |
| rolling one-step (upper bar) | 8.13 | 20.40 | 6.21% | 0.9751 |

## Per horizon

| model | h1 | h2 | h3 |
|---|---|---|---|
| TFT | 15.97 | 18.24 | 18.56 |
| persistence | 7.34 | 11.27 | 14.47 |
| seasonal naive | 20.73 | 21.50 | 21.95 |
| skill vs persistence | -1.176 | -0.618 | -0.283 |

Pooled skill vs persistence: **-0.595** (lower is better for MAE; negative skill means
persistence still wins pooled). The TFT gap shrinks with horizon: h3 skill (-0.283) is
the closest.

## Interval calibration (nominal 80%)

| metric | value |
|---|---|
| coverage (pooled) | 0.655 |
| mean width | 42.09 NPR |
| quantile crossings | 0.35% of cells |

Coverage below 0.80 - intervals are optimistic; crossing rate is negligible.

## Row-level naive reference (different scoring unit)

| baseline | MAE |
|---|---|
| last value | 8.12 |
| seasonal 12 | 20.81 |

All artifacts: `tft_results/metrics.json`, `tft_results/final_report.txt`,
`tft_results/test_predictions.npz`, plots alongside them.
