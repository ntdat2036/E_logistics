"""test_split_cv.py - Kiểm thử các bộ chia Cross-Validation thuần NumPy (Mục 9.1).
1. Expanding-window (TimeSeriesSplit): Không rò rỉ (mọi chỉ số val > mọi chỉ số train trong từng fold).
2. StratifiedKFold: Giữ nguyên tỷ lệ nhãn dương tương đồng giữa các fold.
"""
import numpy as np
import pytest
from src.model import TimeSeriesSplit, StratifiedKFold


def test_time_series_split_no_leakage():
    """Kiểm tra expanding-window không rò rỉ: val luôn diễn ra sau train."""
    n = 1000
    X = np.arange(n).reshape(-1, 1)
    cv = TimeSeriesSplit(n_splits=4)

    splits = list(cv.split(X))
    assert len(splits) == 4, "Phải có đúng 4 fold"

    prev_train_len = 0
    for fold, (train_idx, val_idx) in enumerate(splits):
        assert len(train_idx) > prev_train_len, "Train set phải mở rộng dần qua các fold"
        prev_train_len = len(train_idx)

        assert len(set(train_idx) & set(val_idx)) == 0, f"Fold {fold}: Train và Val bị giao nhau!"
        assert train_idx.max() < val_idx.min(), f"Fold {fold}: Có chỉ mục train >= chỉ mục val (rò rỉ tương lai)!"


def test_stratified_kfold_preserves_ratio():
    """Kiểm tra StratifiedKFold giữ nguyên tỷ lệ nhãn dương ~10% ở mọi fold."""
    rng = np.random.default_rng(42)
    y = np.zeros(1000, dtype=int)
    y[:100] = 1  # 10% nhãn 1
    rng.shuffle(y)
    X = np.zeros((1000, 2))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))
    assert len(splits) == 5

    base_rate = y.mean()
    for fold, (train_idx, val_idx) in enumerate(splits):
        assert len(set(train_idx) & set(val_idx)) == 0
        val_rate = y[val_idx].mean()
        train_rate = y[train_idx].mean()

        assert abs(val_rate - base_rate) < 0.02, f"Fold {fold}: Tỷ lệ val ({val_rate}) lệch xa base ({base_rate})"
        assert abs(train_rate - base_rate) < 0.02, f"Fold {fold}: Tỷ lệ train ({train_rate}) lệch xa base ({base_rate})"

