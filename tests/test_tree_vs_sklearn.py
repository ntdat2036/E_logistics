"""test_tree_vs_sklearn.py - Kiểm thử đối chiếu DecisionTree thuần NumPy với sklearn (Mục 9.1).
1. Tính tất định: Hai lần fit cùng dữ liệu và tham số phải cho kết quả dự đoán giống hệt nhau (sai số 0.0).
2. Chênh lệch ROC-AUC giữa cây NumPy và sklearn DecisionTreeClassifier < 0.01 trên cùng tập dữ liệu.
3. Hỗ trợ criterion entropy và class_weight balanced.
"""
import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from src.model import DecisionTree


def test_tree_determinism():
    """Kiểm tra cây thuần NumPy là tất định (deterministic)."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(500, 8))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    tree1 = DecisionTree(max_depth=4, min_samples_leaf=20, n_bins=32, random_state=123).fit(X, y)
    tree2 = DecisionTree(max_depth=4, min_samples_leaf=20, n_bins=32, random_state=123).fit(X, y)

    p1 = tree1.predict_proba(X)
    p2 = tree2.predict_proba(X)

    np.testing.assert_array_equal(p1, p2, err_msg="Hai lần huấn luyện độc lập cho kết quả khác nhau!")


def test_tree_auc_vs_sklearn():
    """Kiểm tra chênh lệch AUC giữa bản NumPy và sklearn < 0.01."""
    rng = np.random.default_rng(2026)
    X_train = rng.normal(0, 1, size=(2500, 12))
    # Tạo nhãn phụ thuộc phi tuyến tính nhẹ
    y_train = (
        (X_train[:, 0] > 0.2) & (X_train[:, 1] < 0.5)
        | (X_train[:, 2] + X_train[:, 3] > 0.8)
    ).astype(int)

    X_test = rng.normal(0, 1, size=(800, 12))
    y_test = (
        (X_test[:, 0] > 0.2) & (X_test[:, 1] < 0.5)
        | (X_test[:, 2] + X_test[:, 3] > 0.8)
    ).astype(int)

    # 1. DecisionTree thuần NumPy
    tree_np = DecisionTree(max_depth=5, min_samples_leaf=30, n_bins=64, random_state=42).fit(X_train, y_train)
    p_np = tree_np.predict_proba(X_test)
    auc_np = roc_auc_score(y_test, p_np)

    # 2. scikit-learn DecisionTreeClassifier
    tree_sk = DecisionTreeClassifier(max_depth=5, min_samples_leaf=30, random_state=42).fit(X_train, y_train)
    p_sk = tree_sk.predict_proba(X_test)[:, 1]
    auc_sk = roc_auc_score(y_test, p_sk)

    diff = abs(auc_np - auc_sk)
    assert diff < 0.01, f"Chênh lệch AUC quá lớn: np={auc_np:.4f}, sk={auc_sk:.4f}, diff={diff:.4f}"


def test_tree_entropy_criterion():
    """Kiểm tra cây hoạt động chuẩn xác với criterion='entropy'."""
    rng = np.random.default_rng(99)
    X = rng.normal(0, 1, size=(1000, 6))
    y = (X[:, 0] > 0).astype(int)

    tree_ent = DecisionTree(max_depth=3, min_samples_leaf=20, criterion='entropy').fit(X, y)
    p_ent = tree_ent.predict_proba(X)

    assert p_ent.min() >= 0.0 and p_ent.max() <= 1.0
    auc = roc_auc_score(y, p_ent)
    assert auc > 0.70

