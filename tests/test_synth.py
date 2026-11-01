"""test_synth.py - Kiểm thử module sinh dữ liệu giả định (Mục 4.7, Mục 10 và ASSUMPTIONS.md).
1. Tính tất định: SEED = 42 sinh ra dữ liệu giống hệt nhau ở các lần chạy.
2. Toàn bộ các cột giả lập đều có tiền tố 'sim_'.
3. Tuyệt đối không có cột nào có tiền tố 'sim_' xuất hiện trong danh sách FEATURES của src/data_prep.py (chống ô nhiễm mô hình ML).
4. Dữ liệu gốc trong data/raw/ không bị thay đổi.
"""
import os
import pandas as pd
import pytest
from src.synth import generate_simulated_data, SEED
from src.data_prep import FEATURES, compute_raw_sha256
from tests.test_data_prep import EXPECTED_SHA256


def test_synth_determinism_and_prefix(tmp_path):
    """Kiểm tra tính tất định và tiền tố sim_."""
    out_dir1 = str(tmp_path / 'sim1')
    sku1, ord1 = generate_simulated_data(raw_dir='data/raw', out_dir=out_dir1, seed=SEED)

    # Kiểm tra tiền tố sim_ ở bảng SKU
    for col in sku1.columns:
        if col not in ['product_id', 'product_category_name']:
            assert col.startswith('sim_'), f"Cột {col} trong sku_inventory thiếu tiền tố 'sim_'!"

    # Kiểm tra tiền tố sim_ ở bảng Orders
    for col in ord1.columns:
        if col not in ['order_id']:
            assert col.startswith('sim_'), f"Cột {col} trong orders_simulated thiếu tiền tố 'sim_'!"

    # Kiểm tra tính tất định (seed 42)
    out_dir2 = str(tmp_path / 'sim2')
    sku2, ord2 = generate_simulated_data(raw_dir='data/raw', out_dir=out_dir2, seed=SEED)

    pd.testing.assert_frame_equal(sku1, sku2)
    pd.testing.assert_frame_equal(ord1, ord2)


def test_no_sim_columns_in_ml_features():
    """Quy tắc R4 & Điều cấm 7: Tuyệt đối không đưa cột sim_ vào mô hình học máy."""
    for f in FEATURES:
        assert not f.startswith('sim_'), (
            f"VI PHẠM ĐIỀU CẤM 7: Cột giả lập {f} xuất hiện trong FEATURES huấn luyện của mô hình!"
        )


def test_raw_csv_unmodified_after_synth():
    """Kiểm tra dữ liệu gốc data/raw/ vẫn nguyên vẹn 100%."""
    current_hashes = compute_raw_sha256('data/raw')
    for fname, exp_hash in EXPECTED_SHA256.items():
        assert current_hashes[fname].lower() == exp_hash.lower()

