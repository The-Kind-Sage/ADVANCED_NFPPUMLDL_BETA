"""Smoke mode - tiny 2-epoch test that TFT runs end-to-end. Writes to tft_results/smoke/."""
import os
from data import load_panel, split_panel, build_datasets
from models.tft import train_tft
from evaluation import evaluate
from reporting import write_final_report, print_final_summary
from config import OUT_DIR, SMOKE_SERIES, SMOKE_EPOCHS


def run_smoke():
    df = load_panel()
    print("\n*** SMOKE MODE: tiny subset, 2 epochs, just to verify TFT runs ***")
    # own subfolder: a smoke run must never overwrite root evaluation reports
    out_dir = os.path.join(OUT_DIR, "smoke")
    os.makedirs(out_dir, exist_ok=True)
    top = df.groupby("series_id").size().sort_values(ascending=False).head(SMOKE_SERIES).index
    df = df[df["series_id"].isin(top)].copy()
    train_df, val_hist, test_hist, val_idx, test_idx = split_panel(df)
    training, validation, test = build_datasets(
        train_df, val_hist, test_hist, val_idx, test_idx, max_encoder=12, max_pred=2)
    best, _, history = train_tft(training, validation, out_dir, max_epochs=SMOKE_EPOCHS,
                                 batch=256, hidden=16, patience=2,
                                 ref_median=float(df["price"].median()))
    metrics = evaluate(best, test, df, out_dir)
    path = write_final_report(
        out_dir, "smoke", history, metrics,
        config=dict(lookback=12, horizon=2, batch=256, hidden=16,
                    max_epochs=SMOKE_EPOCHS, patience=2))
    print_final_summary(metrics)
    print(f"\nsaved {path}")
    print("\nSMOKE OK. Next: python -m scripts.train")


if __name__ == "__main__":
    run_smoke()
