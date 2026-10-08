"""CLI dispatcher for Nepal Food Price Forecasting with TFT."""
import argparse
import importlib


def main():
    ap = argparse.ArgumentParser(description="Nepal Food Price Forecasting with TFT")
    ap.add_argument("--mode", default="train", choices=["prep", "smoke", "train", "eval"],
                    help="prep=EDA only, smoke=2-epoch test, train=full run, "
                         "eval=re-score newest checkpoint without training")
    args = ap.parse_args()
    mod = importlib.import_module(f"scripts.{args.mode}")
    getattr(mod, f"run_{args.mode}")()


if __name__ == "__main__":
    main()
