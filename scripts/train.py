"""Train mode - FULL run: train TFT + evaluate + forecast."""
from data import load_panel, split_panel, build_datasets
from models.tft import train_tft
from evaluation import evaluate, future_forecast, per_commodity_breakdown
from reporting import write_final_report, print_final_summary
from config import OUT_DIR


def run_train():
    df = load_panel()
    train_df, val_hist, test_hist, val_idx, test_idx = split_panel(df)
    training, validation, test = build_datasets(train_df, val_hist, test_hist, val_idx, test_idx)
    best, _, history = train_tft(training, validation, OUT_DIR,
                                 ref_median=float(df["price"].median()))
    metrics = evaluate(best, test, df, OUT_DIR)
    commodity_rows = per_commodity_breakdown(best, training, test_hist, test_idx, OUT_DIR)
    future_forecast(best, training, df, OUT_DIR)
    path = write_final_report(OUT_DIR, "train", history, metrics, commodity_rows)
    print_final_summary(metrics)
    print(f"\nDONE. All outputs in ./{OUT_DIR}/")
    print(f"Final report : {path}")
    print("Best-model comparison for your report: TFT overall_pooled vs naive references in metrics.json")


if __name__ == "__main__":
    run_train()
