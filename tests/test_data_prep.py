"""test_data_prep.py - Kiểm thử bộ tiền xử lý dữ liệu và kiểm tra tính toàn vẹn (Mục 9.1).
1. Phễu lọc: 99,441 -> 95,180 -> 88,526.
2. Nhãn là nhị phân 0 hoặc 1.
3. Không đặc trưng nào bị NaN sau khi áp dụng fill_train_median.
4. Mã băm SHA256 của 5 file trong data/raw/ không đổi.
"""
import os
import pytest
import numpy as np
from src.data_prep import (
    compute_raw_sha256,
    load_raw,
    build_order_table,
    add_features,
    select_cohort,
    fill_train_median,
    time_split,
    FEATURES,
    RAW_FILES,
)

EXPECTED_SHA256 = {
    'olist_customers_dataset.csv': '983a422239e1712ded753b3bf9ecf47dc73f144d306029dcfa99e70a226883d2',
    'olist_order_items_dataset.csv': '0bc4d068c4fe38cbb01bd90e8746e3c613fe7b4baef75fab7b0e329701c3e279',
    'olist_orders_dataset.csv': '8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf',
    'olist_products_dataset.csv': '3e6569628a17fbc75fd206ee357b59e20364b9afa90f5b6cd5b4d624c58aa9cc',
    'olist_sellers_dataset.csv': '1f643d2b950373b85735e7794b20986f528d7a000432e7c6f9bcbb44d0846a0e',
}


def test_raw_files_sha256_unmodified():
    """Quy tắc R1: 5 file gốc không bị sửa, ghi đè."""
    current_hashes = compute_raw_sha256('data/raw')
    assert len(current_hashes) == 5, "Phải có đủ 5 tập tin CSV trong data/raw/"
    for fname, exp_hash in EXPECTED_SHA256.items():
        assert fname in current_hashes, f"Thiếu tập tin {fname}"
        assert current_hashes[fname].lower() == exp_hash.lower(), (
            f"Tập tin {fname} đã bị thay đổi! Hash hiện tại: {current_hashes[fname]}"
        )


def test_funnel_and_cohort():
    """Kiểm tra phễu lọc 99,441 -> 95,180 -> 88,526 và split."""
    raw = load_raw('data/raw')
    assert len(raw['orders']) == 99441, "Số đơn thô phải là 99,441"

    d_order = build_order_table(raw)
    assert len(d_order) == 95180, f"Số đơn sau khi lọc delivered + 1 seller + đủ timestamp phải là 95,180, thực tế {len(d_order)}"

    d_feat = add_features(d_order)
    d_cohort = select_cohort(d_feat)
    assert len(d_cohort) == 88526, f"Cohort phải là 88,526, thực tế {len(d_cohort)}"

    tr_fit, val_idx, test_idx = time_split(d_cohort)
    assert len(tr_fit) == 55808, f"Train fit phải là 55,808, thực tế {len(tr_fit)}"
    assert len(val_idx) == 13953, f"Val phải là 13,953, thực tế {len(val_idx)}"
    assert len(test_idx) == 18765, f"Test phải là 18,765, thực tế {len(test_idx)}"


def test_labels_are_binary():
    """Kiểm tra nhãn late_handover và late_delivery chỉ có 0 và 1."""
    raw = load_raw('data/raw')
    d_order = build_order_table(raw)

    assert set(d_order['late_handover'].unique()).issubset({0, 1})
    assert set(d_order['late_delivery'].unique()).issubset({0, 1})
    assert set(d_order['ts_anomaly'].unique()).issubset({0, 1})


def test_features_no_nan_after_fill():
    """Kiểm tra không đặc trưng nào bị NaN sau khi điền trung vị tập train."""
    raw = load_raw('data/raw')
    d_order = build_order_table(raw)
    d_feat = add_features(d_order)
    d_cohort = select_cohort(d_feat)

    tr_fit, _, _ = time_split(d_cohort)
    d_filled = fill_train_median(d_cohort, tr_fit)

    for f in FEATURES:
        assert f in d_filled.columns, f"Thiếu đặc trưng {f}"
        nan_cnt = d_filled[f].isnull().sum()
        assert nan_cnt == 0, f"Đặc trưng {f} có {nan_cnt} giá trị NaN"

