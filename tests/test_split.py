import pandas as pd

from data import chronological_split


def make_frame():
    dates = pd.date_range("2021-01-15", periods=60, freq="MS")
    return pd.DataFrame({"date": dates, "time_idx": list(range(60)),
                         "series_id": ["s1"] * 60, "price": [10.0] * 60})


def test_split_sizes_and_boundaries():
    df = make_frame()
    train, val_hist, test_hist, val_idx, test_idx = chronological_split(
        df, "2024-01-15", "2025-01-15")
    assert len(train) == 36
    assert len(val_hist) == 48
    assert len(test_hist) == 60
    assert val_idx == 36
    assert test_idx == 48
    assert train["date"].max() < pd.Timestamp("2024-01-15")
    assert val_hist["date"].max() < pd.Timestamp("2025-01-15")
    assert test_hist["date"].max() == df["date"].max()
