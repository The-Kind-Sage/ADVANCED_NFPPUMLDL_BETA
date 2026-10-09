import numpy as np
import pandas as pd

from evaluation.metrics import reg_metrics, naive_baselines, aligned_baselines


def test_reg_metrics_known_values():
    m = reg_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    assert abs(m["MAE"] - 2.0 / 3.0) < 1e-12
    assert abs(m["RMSE"] - np.sqrt(4.0 / 3.0)) < 1e-12
    assert m["n"] == 3


def test_reg_metrics_masks_non_finite():
    m = reg_metrics([1.0, np.nan, 3.0], [1.0, 2.0, 4.0])
    assert m["n"] == 2
    assert abs(m["MAE"] - 0.5) < 1e-12


def test_naive_baselines_values():
    dates = pd.date_range("2024-01-15", periods=14, freq="MS")
    prices = [10.0 + i for i in range(14)]
    df = pd.DataFrame({"series_id": ["s"] * 14, "date": dates, "price": prices})
    out = naive_baselines(df, "2025-01-15")
    assert out["naive_last_value"]["n"] == 2
    assert abs(out["naive_last_value"]["MAE"] - 1.0) < 1e-12
    assert abs(out["naive_seasonal_12"]["MAE"] - 12.0) < 1e-12


def test_aligned_baselines_same_windows():
    df = pd.DataFrame({"series_id": ["s1"] * 15,
                       "time_idx": list(range(15)),
                       "price": [10.0 + i for i in range(15)]})
    index = pd.DataFrame({"series_id": ["s1", "s1", "s1"], "time_idx": [3, 4, 14]})
    out = aligned_baselines(df, index, horizon=3)
    pers, seas, roll = out["persistence"], out["seasonal_naive"], out["rolling_one_step"]
    np.testing.assert_allclose(pers[0], [12.0, 12.0, 12.0])
    np.testing.assert_allclose(pers[1], [13.0, 13.0, 13.0])
    assert np.isnan(seas[0]).all() and np.isnan(seas[1]).all()
    np.testing.assert_allclose(seas[2], [12.0, 13.0, 14.0])
    np.testing.assert_allclose(roll[0], [12.0, 13.0, 14.0])
    np.testing.assert_allclose(roll[1], [13.0, 14.0, 15.0])
