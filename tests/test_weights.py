"""test_weights.py - Kiểm thử lưu và nạp mô hình cây quyết định (Mục 9.1).
Lưu -> nạp -> dự đoán trên tập dữ liệu thử nghiệm phải cho kết quả giống hệt (sai số tuyệt đối = 0.0).
"""
import os
import shutil
import numpy as np
import pytest
from src.model import DecisionTree
from src.weights import save_tree, load_tree, verify_roundtrip, predict_tree_dict


def test_tree_save_and_load_roundtrip(tmp_path):
    """Kiểm tra khứ hồi lưu và nạp DecisionTree cho dự đoán trùng khớp tuyệt đối (0.0)."""
    weights_dir = str(tmp_path / 'weights_test')

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(800, 6))
    y = (X[:, 0] + X[:, 1] > 0.5).astype(int)
    feature_names = [f'feat_{i}' for i in range(6)]

    tree = DecisionTree(max_depth=4, min_samples_leaf=20, n_bins=32, random_state=42).fit(X, y)
    thresholds = {'t1': 0.7, 't2': 0.3, 'prior': 0.25}
    best_params = {'max_depth': 4, 'min_samples_leaf': 20, 'n_bins': 32}

    # Lưu
    save_tree(tree, feature_names, thresholds, best_params, weights_dir=weights_dir)

    # Nạp và kiểm tra khứ hồi
    assert verify_roundtrip(tree, X, weights_dir=weights_dir)

    # Kiểm tra dự đoán từ dict JSON
    loaded_tree, tree_dict, loaded_thr, loaded_bp = load_tree(weights_dir=weights_dir)
    assert loaded_thr['t1'] == 0.7
    assert loaded_bp['max_depth'] == 4

    p_orig = tree.predict_proba(X)
    p_json = np.array([predict_tree_dict(tree_dict, row) for row in X])
    np.testing.assert_array_equal(p_orig, p_json)

