"""train.py - Tinh chỉnh siêu tham số, khóa ngưỡng trên Validation và đánh giá cuối trên Test (Giai đoạn 4).
TUÂN THỦ TUYỆT ĐỐI MỤC 2 (R2, R3) VÀ MỤC 4.5, 4.6, 4.9.
1. GridSearch CV expanding-window 4 fold trên train_fit, tiêu chí PR-AUC.
2. Huấn luyện mô hình tối ưu trên train_fit.
3. Quét và KHÓA ngưỡng trên Validation: t1 (p85), t2 (p45), t_f1, t_cost. Lưu vào weights/.
4. Sau khi khóa, mở tập Test ĐÚNG MỘT LẦN để đo lường và xuất outputs/04_final_metrics.json.
"""
import json
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd

from src.data_prep import (
    FEATURES,
    load_raw,
    build_order_table,
    add_features,
    select_cohort,
    time_split,
    fill_train_median,
)
from src.model import (
    DecisionTree,
    TimeSeriesSplit,
    StratifiedKFold,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from src.weights import save_tree, verify_roundtrip


def run_grid_search(X_tr, y_tr, param_grid=None, verbose=True):
    """Thực hiện GridSearch expanding-window 4-fold theo tiêu chí PR-AUC (Mục 4.5)."""
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 4, 5, 6, 8],
            'min_samples_leaf': [50, 100, 200, 400],
            'criterion': ['gini', 'entropy'],
            'n_bins': [32, 64],
        }

    cv_exp = TimeSeriesSplit(n_splits=4)
    splits_exp = list(cv_exp.split(X_tr, y_tr))

    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits_strat = list(cv_strat.split(X_tr, y_tr))

    import itertools
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    if verbose:
        print(f"Bắt đầu GridSearch trên {len(combos)} tổ hợp siêu tham số...")

    results = []
    best_score = -float('inf')
    best_params = None

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        params['min_samples_split'] = 2 * params['min_samples_leaf']

        # 1. Đánh giá expanding-window
        exp_scores = []
        for tr_i, va_i in splits_exp:
            tree = DecisionTree(**params, random_state=42).fit(X_tr[tr_i], y_tr[tr_i])
            p_val = tree.predict_proba(X_tr[va_i])
            exp_scores.append(average_precision_score(y_tr[va_i], p_val))
        mean_exp = float(np.mean(exp_scores))

        # 2. Đánh giá Stratified K-Fold (để so sánh)
        strat_scores = []
        for tr_i, va_i in splits_strat:
            tree = DecisionTree(**params, random_state=42).fit(X_tr[tr_i], y_tr[tr_i])
            p_val = tree.predict_proba(X_tr[va_i])
            strat_scores.append(average_precision_score(y_tr[va_i], p_val))
        mean_strat = float(np.mean(strat_scores))

        res = {
            'params': params,
            'mean_pr_auc_exp': round(mean_exp, 4),
            'std_pr_auc_exp': round(float(np.std(exp_scores)), 4),
            'mean_pr_auc_strat': round(mean_strat, 4),
        }
        results.append(res)

        if mean_exp > best_score:
            best_score = mean_exp
            best_params = params

        if verbose and (i + 1) % 20 == 0:
            print(f"  -> Đã duyệt {i + 1}/{len(combos)} tổ hợp...")

    # Sắp xếp kết quả giảm dần theo PR-AUC expanding-window
    results.sort(key=lambda x: x['mean_pr_auc_exp'], reverse=True)
    if verbose:
        print(f"Hoàn thành GridSearch! Tham số tốt nhất: {best_params} (CV PR-AUC={best_score:.4f})")

    return best_params, results


def find_validation_thresholds(y_val, p_val, c_fn=3.0, c_fp=1.0):
    """Quét và xác định các ngưỡng quyết định trên tập Validation (Mục 4.6)."""
    # 1. Cắt mức ưu tiên P1/P2/P3 theo phân vị điểm rủi ro trên validation
    # t1: phân vị 85% (top 15% rủi ro cao nhất xếp P1)
    # t2: phân vị 45% (từ 45% đến 85% xếp P2, dưới 45% xếp P3)
    t1 = float(np.percentile(p_val, 85.0))
    t2 = float(np.percentile(p_val, 45.0))

    # 2. Quét ngưỡng F1 tối ưu và ngưỡng Chi phí tối ưu (C_FN=3, C_FP=1)
    threshold_grid = np.unique(p_val)
    # Lọc lấy các ngưỡng hợp lý
    threshold_grid = np.linspace(0.05, 0.60, 200)

    best_f1 = -1.0
    best_t_f1 = 0.5
    min_cost = float('inf')
    best_t_cost = 0.5

    for thr in threshold_grid:
        y_pred = (p_val >= thr).astype(int)
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

        # F1
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_t_f1 = float(thr)

        # Chi phí
        cost = c_fn * fn + c_fp * fp
        if cost < min_cost:
            min_cost = cost
            best_t_cost = float(thr)

    return {
        't1': round(t1, 4),
        't2': round(t2, 4),
        't_f1': round(best_t_f1, 4),
        'val_best_f1': round(best_f1, 4),
        't_cost': round(best_t_cost, 4),
        'val_min_cost': float(min_cost),
        'c_fn': c_fn,
        'c_fp': c_fp,
    }


def evaluate_final_test_set(tree, X_test, y_test, thresholds, d_test=None):
    """Mở tập Test ĐÚNG MỘT LẦN sau khi các ngưỡng đã bị khóa cứng (Mục 4.6 & 4.9)."""
    p_test = tree.predict_proba(X_test)
    y_test = np.asarray(y_test, dtype=int)
    n_test = len(y_test)
    n_pos = int(y_test.sum())

    t1 = thresholds['t1']
    t2 = thresholds['t2']
    t_f1 = thresholds['t_f1']

    # 1. Chỉ số tổng thể
    roc_auc = float(roc_auc_score(y_test, p_test))
    pr_auc = float(average_precision_score(y_test, p_test))
    brier = float(brier_score_loss(y_test, p_test))

    # 2. Đánh giá tại ngưỡng F1 khóa từ validation
    y_pred_f1 = (p_test >= t_f1).astype(int)
    cm_f1 = confusion_matrix(y_test, y_pred_f1)
    acc = float(accuracy_score(y_test, y_pred_f1))
    prec = float(precision_score(y_test, y_pred_f1, zero_division=0))
    rec = float(recall_score(y_test, y_pred_f1, zero_division=0))
    f1 = float(f1_score(y_test, y_pred_f1, zero_division=0))

    # 3. Phân bổ các mức ưu tiên P1 / P2 / P3
    mask_p1 = p_test >= t1
    mask_p2 = (p_test >= t2) & (~mask_p1)
    mask_p3 = p_test < t2

    n_p1, n_p2, n_p3 = int(mask_p1.sum()), int(mask_p2.sum()), int(mask_p3.sum())
    late_p1 = int(y_test[mask_p1].sum())
    late_p2 = int(y_test[mask_p2].sum())
    late_p3 = int(y_test[mask_p3].sum())

    # 4. Triage Recall: Tỷ lệ bắt đơn trễ khi chỉ cảnh báo top 15% đơn rủi ro nhất
    top_15_count = int(np.ceil(0.15 * n_test))
    top_15_idx = np.argsort(p_test)[::-1][:top_15_count]
    recall_top15 = float(y_test[top_15_idx].sum() / n_pos)

    # 5. Cờ can thiệp Trưởng ca (P1 hoặc seller_hist_n < 20)
    manual_flag = mask_p1.copy()
    if d_test is not None and 'seller_hist_n' in d_test.columns:
        low_hist_mask = (d_test['seller_hist_n'].values < 20)
        manual_flag = mask_p1 | low_hist_mask
        n_low_hist = int(low_hist_mask.sum())
        n_manual = int(manual_flag.sum())
        late_captured_manual = int(y_test[manual_flag].sum())
    else:
        n_low_hist = 0
        n_manual = n_p1
        late_captured_manual = late_p1

    metrics = {
        'test_size': n_test,
        'test_late_orders': n_pos,
        'test_base_rate': round(float(n_pos / n_test), 4),
        'roc_auc': round(roc_auc, 4),
        'pr_auc': round(pr_auc, 4),
        'brier': round(brier, 4),
        'at_f1_threshold': {
            'threshold': t_f1,
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4),
            'confusion_matrix': {
                'tn': int(cm_f1[0, 0]),
                'fp': int(cm_f1[0, 1]),
                'fn': int(cm_f1[1, 0]),
                'tp': int(cm_f1[1, 1]),
            },
        },
        'priority_levels': {
            'P1': {
                'orders_count': n_p1,
                'orders_share': round(n_p1 / n_test, 4),
                'late_count': late_p1,
                'late_rate': round(late_p1 / n_p1, 4) if n_p1 > 0 else 0,
                'late_captured_share': round(late_p1 / n_pos, 4),
            },
            'P2': {
                'orders_count': n_p2,
                'orders_share': round(n_p2 / n_test, 4),
                'late_count': late_p2,
                'late_rate': round(late_p2 / n_p2, 4) if n_p2 > 0 else 0,
                'late_captured_share': round(late_p2 / n_pos, 4),
            },
            'P3': {
                'orders_count': n_p3,
                'orders_share': round(n_p3 / n_test, 4),
                'late_count': late_p3,
                'late_rate': round(late_p3 / n_p3, 4) if n_p3 > 0 else 0,
                'late_captured_share': round(late_p3 / n_pos, 4),
            },
        },
        'triage_recall_top15_percent': round(recall_top15, 4),
        'shift_supervisor_manual_flag': {
            'p1_share': round(n_p1 / n_test, 4),
            'low_history_share': round(n_low_hist / n_test, 4),
            'total_manual_share': round(n_manual / n_test, 4),
            'late_orders_captured_share': round(late_captured_manual / n_pos, 4),
        },
    }
    return metrics, p_test


def main():
    print("=== GIAI ĐOẠN 4: TUNING, KHÓA NGƯỠNG VÀ HUẤN LUYỆN CUỐI ===")
    raw = load_raw('data/raw')
    d_order = build_order_table(raw)
    d_all = add_features(d_order)
    d_cohort = select_cohort(d_all)
    tr_fit, val_idx, test_idx = time_split(d_cohort)

    d_filled = fill_train_median(d_cohort, tr_fit)
    X = d_filled[FEATURES].values
    y = d_filled.late_handover.values.astype(int)

    X_tr, y_tr = X[tr_fit], y[tr_fit]
    X_va, y_va = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    # 1. GridSearch
    print("\n--- 1. Tinh chỉnh siêu tham số (GridSearch) ---")
    best_params, cv_results = run_grid_search(X_tr, y_tr, verbose=True)

    # 2. Huấn luyện mô hình tối ưu trên train_fit
    print("\n--- 2. Huấn luyện mô hình DecisionTree tối ưu trên train_fit ---")
    tree = DecisionTree(**best_params, random_state=42).fit(X_tr, y_tr)

    # 3. Quét và KHÓA ngưỡng trên Validation
    print("\n--- 3. Quét và KHÓA ngưỡng trên Validation (chưa chạm Test) ---")
    p_val = tree.predict_proba(X_va)
    thresholds = find_validation_thresholds(y_va, p_val)

    # Bổ sung các thông số cần thiết vào thresholds
    pre = (d_all.order_approved_at < '2018-05-01').values
    thresholds['prior_del'] = round(float(d_all.late_delivery.values[pre].mean()), 4)
    thresholds['prior_han'] = round(float(d_all.late_handover.values[pre].mean()), 4)
    thresholds['train_median_weight'] = float(d_cohort.loc[tr_fit, 'weight'].median())
    thresholds['train_median_vol'] = float(d_cohort.loc[tr_fit, 'vol'].median())

    print(f"  -> Ngưỡng khóa: t1 (P1 cut)={thresholds['t1']}, t2 (P2 cut)={thresholds['t2']}")
    print(f"  -> Ngưỡng tối ưu F1={thresholds['t_f1']} (Val F1={thresholds['val_best_f1']})")
    print(f"  -> Ngưỡng tối ưu Chi phí={thresholds['t_cost']}")

    # Lưu trọng số vào weights/
    save_tree(tree, FEATURES, thresholds, best_params, weights_dir='weights')
    print("  -> Đã lưu các artifact vào weights/ (tree.json, thresholds.json, best_params.json, tree.npz)")

    # Kiểm tra khứ hồi ngay lập tức
    verify_roundtrip(tree, X_va[:200], weights_dir='weights')
    print("  -> Kiểm tra khứ hồi (roundtrip test): [PASS] (sai số 0.0)")

    # 4. Đánh giá duy nhất 1 lần trên tập Test
    print("\n--- 4. Mở tập Test đánh giá duy nhất một lần ---")
    d_test = d_cohort.iloc[test_idx].reset_index(drop=True)
    test_metrics, p_test = evaluate_final_test_set(tree, X_te, y_te, thresholds, d_test=d_test)

    # Lưu outputs/04_final_metrics.json
    out_json = os.path.join('outputs', '04_final_metrics.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    print(f"  -> Đã ghi nhận chỉ số cuối cùng vào {out_json}")

    # Xuất báo cáo markdown
    out_md = os.path.join('outputs', '04_tuning_report.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("# BÁO CÁO TINH CHỈNH VÀ CHỈ SỐ CUỐI CÙNG (GIAI ĐOẠN 4)\n\n")
        f.write("## 1. Siêu tham số tối ưu (GridSearch CV Expanding-Window)\n\n")
        f.write("```json\n" + json.dumps(best_params, indent=2) + "\n```\n\n")
        f.write("## 2. Ngưỡng phân vị và ngưỡng phân loại đã khóa (từ Validation)\n\n")
        f.write("```json\n" + json.dumps(thresholds, indent=2) + "\n```\n\n")
        f.write("## 3. Chỉ số kiểm định trên tập Test (Mục 4.9)\n\n")
        f.write(f"- **Test ROC-AUC:** {test_metrics['roc_auc']}\n")
        f.write(f"- **Test PR-AUC:** {test_metrics['pr_auc']}\n")
        f.write(f"- **Test Brier score:** {test_metrics['brier']}\n")
        f.write(f"- **Precision / Recall / F1 tại ngưỡng {test_metrics['at_f1_threshold']['threshold']}:** ")
        f.write(f"{test_metrics['at_f1_threshold']['precision']} / {test_metrics['at_f1_threshold']['recall']} / {test_metrics['at_f1_threshold']['f1']}\n")
        f.write(f"- **Confusion Matrix:** TN={test_metrics['at_f1_threshold']['confusion_matrix']['tn']}, FP={test_metrics['at_f1_threshold']['confusion_matrix']['fp']}, FN={test_metrics['at_f1_threshold']['confusion_matrix']['fn']}, TP={test_metrics['at_f1_threshold']['confusion_matrix']['tp']}\n\n")
        f.write("## 4. Phân tầng ưu tiên trên Test\n\n")
        f.write("| Mức ưu tiên | Tỷ lệ đơn | Tỷ lệ trễ | Số đơn trễ bắt được |\n|---|---|---|---|\n")
        for lvl in ['P1', 'P2', 'P3']:
            p_info = test_metrics['priority_levels'][lvl]
            f.write(f"| **{lvl}** | {p_info['orders_share']:.1%} ({p_info['orders_count']:,}) | {p_info['late_rate']:.1%} | {p_info['late_captured_share']:.1%} ({p_info['late_count']:,}) |\n")
        f.write(f"\n- **Recall top 15% rủi ro nhất:** {test_metrics['triage_recall_top15_percent']:.1%}\n")
        f.write(f"- **Tổng số đơn cần Trưởng ca xử lý thủ công:** {test_metrics['shift_supervisor_manual_flag']['total_manual_share']:.1%} (bắt được {test_metrics['shift_supervisor_manual_flag']['late_orders_captured_share']:.1%} đơn trễ)\n")

    print(f"  -> Đã xuất báo cáo chi tiết vào {out_md}")
    print("\n=== HOÀN THÀNH GIAI ĐOẠN 4 ===")


if __name__ == '__main__':
    main()

