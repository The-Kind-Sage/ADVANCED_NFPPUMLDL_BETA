from utils.helpers import _num, _cm_first, _cm_lr, _epoch_line


def test_num_formats_and_missing_values():
    assert _num(None) == "-"
    assert _num(float("nan")) == "-"
    assert _num(float("inf")) == "-"
    assert _num("abc") == "-"
    assert _num(3.14159, 2) == "3.14"
    assert _num(7) == "7.0000"


def test_cm_first_exact_then_prefix():
    assert _cm_first({"train_loss": 1.5}, "train_loss") == 1.5
    assert _cm_first({"val_loss_total": 3.0}, "val_loss") == 3.0
    assert _cm_first({}, "missing") is None


def test_cm_lr():
    assert _cm_lr({"lr-AdamW": 0.001}) == 0.001
    assert _cm_lr({"learning_rate": 0.01}) == 0.01
    assert _cm_lr({"val_loss": 1.0}) is None


def test_epoch_line_content():
    row = {"epoch": 3, "train_loss": 1.0, "val_loss": 0.9, "val_MAE": 5.2,
           "val_RMSE": 7.4, "lr": 0.001, "seconds": 12.3, "is_best": True,
           "best_epoch": 3, "stale_epochs": 0}
    line = _epoch_line(row, 4, 6)
    assert "[EPOCH 03/04]" in line
    assert "*new best*" in line
    assert "best@3 stale=0/6" in line
    assert "12.3s" in line
