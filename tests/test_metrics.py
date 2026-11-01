"""test_metrics.py - Đối chiếu các hàm đo lường thuần NumPy với scikit-learn (Mục 9.1).
So sánh accuracy, precision, recall, f1, roc_auc, pr_auc (average precision), brier score trên:
- 5 mảng kiểm thử nhỏ (edge cases, perfect, all zeros, all ones, imbalanced).
- 1 mảng ngẫu nhiên lớn có giá trị trùng lặp (ties).
"""
import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score as sk_acc,
    precision_score as sk_prec,
    recall_score as sk_rec,
    f1_score as sk_f1,
    roc_auc_score as sk_roc,
    average_precision_score as sk_pr,
    brier_score_loss as sk_brier,
    confusion_matrix as sk_cm,
)
from src.model import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

# 5 mảng nhỏ
SMALL_TEST_CASES = [
    # Case 1: Cân bằng, dự đoán chuẩn
    (np.array([0, 1, 0, 1, 1, 0]), np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])),
    # Case 2: Dự đoán ngược
    (np.array([1, 1, 0, 0]), np.array([0.1, 0.2, 0.8, 0.9])),
    # Case 3: Nhiều điểm số trùng lặp (ties)
    (np.array([0, 1, 1, 0, 1, 0, 0, 1]), np.array([0.5, 0.5, 0.8, 0.2, 0.8, 0.5, 0.2, 0.5])),
    # Case 4: Mất cân bằng lớp nặng (10% dương)
    (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]), np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.9])),
    # Case 5: Toàn bộ cùng một điểm số
    (np.array([0, 1, 0, 1]), np.array([0.4, 0.4, 0.4, 0.4])),
]


@pytest.mark.parametrize("y_true, y_score", SMALL_TEST_CASES)
def test_metrics_on_small_cases(y_true, y_score):
    y_pred = (y_score >= 0.5).astype(int)

    # Accuracy
    assert abs(accuracy_score(y_true, y_pred) - sk_acc(y_true, y_pred)) < 1e-9

    # Confusion matrix
    np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), sk_cm(y_true, y_pred))

    # Precision, Recall, F1
    assert abs(precision_score(y_true, y_pred, zero_division=0) - sk_prec(y_true, y_pred, zero_division=0)) < 1e-9
    assert abs(recall_score(y_true, y_pred, zero_division=0) - sk_rec(y_true, y_pred, zero_division=0)) < 1e-9
    assert abs(f1_score(y_true, y_pred, zero_division=0) - sk_f1(y_true, y_pred, zero_division=0)) < 1e-9

    # Brier
    assert abs(brier_score_loss(y_true, y_score) - sk_brier(y_true, y_score)) < 1e-9

    # ROC-AUC & PR-AUC
    assert abs(roc_auc_score(y_true, y_score) - sk_roc(y_true, y_score)) < 1e-9
    assert abs(average_precision_score(y_true, y_score) - sk_pr(y_true, y_score)) < 1e-9


def test_metrics_on_large_random_array():
    """Kiểm tra trên mảng ngẫu nhiên 1,000 phần tử có nhiều giá trị trùng lặp."""
    rng = np.random.default_rng(2026)
    y_true = rng.integers(0, 2, size=1000)
    # Lượng ties lớn
    y_score = rng.choice(np.linspace(0.0, 1.0, 25), size=1000)
    y_pred = (y_score >= 0.5).astype(int)

    assert abs(accuracy_score(y_true, y_pred) - sk_acc(y_true, y_pred)) < 1e-9
    assert abs(precision_score(y_true, y_pred) - sk_prec(y_true, y_pred)) < 1e-9
    assert abs(recall_score(y_true, y_pred) - sk_rec(y_true, y_pred)) < 1e-9
    assert abs(f1_score(y_true, y_pred) - sk_f1(y_true, y_pred)) < 1e-9
    assert abs(brier_score_loss(y_true, y_score) - sk_brier(y_true, y_score)) < 1e-9
    assert abs(roc_auc_score(y_true, y_score) - sk_roc(y_true, y_score)) < 1e-9
    assert abs(average_precision_score(y_true, y_score) - sk_pr(y_true, y_score)) < 1e-9

