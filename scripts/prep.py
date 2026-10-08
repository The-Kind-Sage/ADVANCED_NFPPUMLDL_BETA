"""Prep mode - EDA + preprocessing check (no torch needed)."""
import os
import json
from data import load_panel, split_panel
from evaluation.metrics import naive_baselines
from config import OUT_DIR, TEST_START


def run_prep():
    df = load_panel()
    split_panel(df)  # prints the chronological split report
    df.to_csv(os.path.join(OUT_DIR, "processed_monthly_panel.csv"), index=False)
    print(f"\nsaved {OUT_DIR}/processed_monthly_panel.csv")
    print("Naive one-step references on test rows:")
    print(json.dumps(naive_baselines(df, TEST_START), indent=2))
    print("\nPREP OK. Next: python -m scripts.smoke")


if __name__ == "__main__":
    run_prep()
