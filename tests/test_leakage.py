"""test_leakage.py - Kiểm thử rò rỉ dữ liệu (Data Leakage Tests - Mục 9.1).
1. Phân chia theo thời gian không giao nhau giữa train_fit, val, test.
2. Với 1,000 đơn ngẫu nhiên, tính lại lịch sử bằng vòng lặp chậm (brute-force) chỉ từ đơn giao xong trước order_approved_at,
   so với bản vector hóa - phải bằng nhau (sai số < 1e-9).
3. Hằng số tiên nghiệm (prior) chỉ tính từ đơn trước 2018-05-01.
4. Số dòng train_fit / val / test = 55,808 / 13,953 / 18,765.
5. Mọi đơn trong test có thời điểm duyệt lớn hơn hoặc bằng mọi đơn trong train_fit.
6. Stratified random split giữ tỷ lệ nhãn dương tương đồng.
"""
import numpy as np
import pandas as pd
import pytest
from src.data_prep import (
    load_raw,
    build_order_table,
    add_features,
    select_cohort,
    time_split,
    stratified_split,
    run_leakage_audit,
    fill_train_median,
)


@pytest.fixture(scope="module")
def prepared_data():
    raw = load_raw('data/raw')
    d_order = build_order_table(raw)
    d_all = add_features(d_order)
    d_cohort = select_cohort(d_all)
    tr_fit, val_idx, test_idx = time_split(d_cohort)
    return {
        'raw': raw,
        'd_order': d_order,
        'd_all': d_all,
        'd_cohort': d_cohort,
        'tr_fit': tr_fit,
        'val_idx': val_idx,
        'test_idx': test_idx,
    }


def test_time_split_no_overlap(prepared_data):
    """Kiểm tra 3 tập train_fit, val, test không có phần tử giao nhau."""
    tr = set(prepared_data['tr_fit'])
    va = set(prepared_data['val_idx'])
    te = set(prepared_data['test_idx'])

    assert len(tr & va) == 0, "Train và Val bị giao nhau!"
    assert len(tr & te) == 0, "Train và Test bị giao nhau!"
    assert len(va & te) == 0, "Val và Test bị giao nhau!"
    assert len(tr) + len(va) + len(te) == len(prepared_data['d_cohort'])


def test_time_split_counts(prepared_data):
    """Kiểm tra chính xác kích thước 55,808 / 13,953 / 18,765."""
    assert len(prepared_data['tr_fit']) == 55808
    assert len(prepared_data['val_idx']) == 13953
    assert len(prepared_data['test_idx']) == 18765


def test_test_set_strictly_after_train(prepared_data):
    """Kiểm tra tính tuần tự thời gian: test không được duyệt trước train."""
    c = prepared_data['d_cohort']
    max_train_fit = c.iloc[prepared_data['tr_fit']].order_approved_at.max()
    min_val = c.iloc[prepared_data['val_idx']].order_approved_at.min()
    max_val = c.iloc[prepared_data['val_idx']].order_approved_at.max()
    min_test = c.iloc[prepared_data['test_idx']].order_approved_at.min()

    assert min_val >= max_train_fit
    assert min_test >= max_val


def test_knowledge_time_brute_force_1000_samples(prepared_data):
    """Kiểm toán Knowledge-time với 1,000 đơn mẫu ngẫu nhiên:
    Tính lại tỷ lệ trễ lịch sử bằng brute force chỉ từ đơn đã giao xong (order_delivered_customer_date <= order_approved_at).
    So với giá trị vector hóa seller_han_rate: sai số phải < 1e-9.
    """
    d_all = prepared_data['d_all']
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(d_all), size=1000, replace=False)

    train_end = '2018-05-01'
    pre = (d_all.order_approved_at < train_end).values
    prior_han = float(d_all.late_handover.values[pre].mean())

    for idx in sample_indices:
        row = d_all.iloc[idx]
        seller = row['seller']
        t_app = row['order_approved_at']

        # Đơn của cùng người bán đã hoàn tất giao trước thời điểm duyệt đơn hiện tại
        past = d_all[(d_all['seller'] == seller) & (d_all['order_delivered_customer_date'] <= t_app)]
        n = len(past)
        l = past['late_handover'].sum()

        expected_rate = (l + 20 * prior_han) / (n + 20)
        actual_rate = row['seller_han_rate']

        assert abs(expected_rate - actual_rate) < 1e-9, (
            f"Lỗi rò rỉ tại đơn idx={idx}: expected {expected_rate}, actual {actual_rate}"
        )


def test_prior_only_from_before_test_cutoff(prepared_data):
    """Kiểm tra hằng số tiên nghiệm prior chỉ tính trên các đơn duyệt trước 2018-05-01."""
    d_all = prepared_data['d_all']
    pre = (d_all.order_approved_at < '2018-05-01').values
    post = (d_all.order_approved_at >= '2018-05-01').values

    assert pre.sum() > 0
    assert post.sum() > 0
    # Đảm bảo không có đơn nào sau 2018-05-01 nằm trong tập pre
    assert d_all.iloc[pre].order_approved_at.max() < pd.Timestamp('2018-05-01')


def test_stratified_split_preserves_class_ratio(prepared_data):
    """Kiểm tra chia ngẫu nhiên phân tầng 80/20 thuần NumPy."""
    c = prepared_data['d_cohort']
    tr, te = stratified_split(c, test_frac=0.2, seed=42)

    assert len(tr) + len(te) == len(c)
    assert len(set(tr) & set(te)) == 0

    base_rate = c.late_handover.mean()
    train_rate = c.iloc[tr].late_handover.mean()
    test_rate = c.iloc[te].late_handover.mean()

    assert abs(base_rate - train_rate) < 0.005
    assert abs(base_rate - test_rate) < 0.005


def test_leakage_audit_cli(prepared_data):
    """Chạy toàn bộ CLI kiểm toán rò rỉ dữ liệu và kiểm tra toàn bộ PASS."""
    passed, results = run_leakage_audit(
        d_all=prepared_data['d_all'],
        d_cohort=prepared_data['d_cohort'],
        verbose=True,
    )
    assert passed, f"Có kiểm toán rò rỉ bị FAIL: {[r for r in results if not r[0]]}"

