# NFPPUMLDL
Nepal Food Price Prediction Using Machine Learning Models and Deep Learning

Forecasts monthly food prices for Nepal with a Temporal Fusion Transformer trained on
WFP market-price data (2001-2026). The model predicts the next 3 months ahead for every
market, commodity and unit, together with 80% prediction intervals.

## Setup

```bash
pip install -e .
```

## Usage

```bash
python main.py --mode prep     # clean and split the monthly price panel
python main.py --mode train    # train the TFT (checkpoints + training history)
python main.py --mode eval     # evaluate the best checkpoint on the holdout window
python main.py --mode smoke    # tiny 2-epoch end-to-end check
```

Extra experiment scripts: `scripts/benchmark_forecasts.py` (five baseline strategies),
`scripts/rolling_backtest.py` (expanding-window backtest) and `scripts/finetune.py`
(short fine-tuning trials).
