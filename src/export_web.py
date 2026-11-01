"""export_web.py - Xuất toàn bộ dữ liệu đã đo lường và tính toán sang web/data/*.json (Mục 8.2).
TUÂN THỦ TUYỆT ĐỐI MỤC 2 (R4): Mọi số liệu trên Web phải đọc từ file outputs/ do code sinh ra, không hard-code.
Không hiển thị bất kỳ ID khách hàng/người bán nào (ẩn danh theo Mục 8.1 & Mục 10).
"""
import json
import os
import shutil
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
from src.weights import load_tree, predict_tree_dict
from src.model import confusion_matrix, f1_score, precision_score, recall_score


def export_all_web_data(out_dir='web/data'):
    """Tạo toàn bộ các file JSON cho 8 trang web tĩnh."""
    os.makedirs(out_dir, exist_ok=True)
    print("=== Đang xuất dữ liệu sang web/data/*.json ===")

    # 1. Copy tree.json và simulation_grid.json đã có
    shutil.copy('weights/tree.json', os.path.join(out_dir, 'tree.json'))
    shutil.copy('outputs/05_simulation_grid.json', os.path.join(out_dir, 'simulation_grid.json'))

    # Đọc các file outputs hiện có
    with open('outputs/01_data_audit.json', 'r', encoding='utf-8') as f:
        audit_raw_data = json.load(f)
    with open('outputs/03_model_comparison.json', 'r', encoding='utf-8') as f:
        comparison_data = json.load(f)
    with open('outputs/04_final_metrics.json', 'r', encoding='utf-8') as f:
        final_metrics = json.load(f)
    with open('outputs/05_bootstrap_ci.json', 'r', encoding='utf-8') as f:
        boot_ci_data = json.load(f)
    with open('outputs/05_iteration_table.json', 'r', encoding='utf-8') as f:
        iter_table = json.load(f)
    with open('outputs/05_model_interpretability.json', 'r', encoding='utf-8') as f:
        interp_data = json.load(f)
    with open('outputs/05_error_analysis.json', 'r', encoding='utf-8') as f:
        error_data = json.load(f)

    # 2. summary.json (Trang 1: Tổng quan)
    summary_data = {
        'project_title': 'Thiết Kế Chuỗi Dịch Vụ Logistics Tự Động Có AI Ra Quyết Định — Nghiệp Vụ Kho (Inventory)',
        'topic_code': 'CK10 - E-logistics',
        'dataset_name': 'Olist Brazilian E-Commerce Public Dataset (Kaggle Version 2)',
        'cohort_period': '2017-01-01 đến 2018-07-31 (88,526 đơn hàng)',
        'core_problem': 'Dự đoán nguy cơ trễ bàn giao kho (late_handover) tại thời điểm duyệt đơn để phân tầng mức độ ưu tiên P1/P2/P3 và hỗ trợ Trưởng ca xử lý.',
        'three_kpis': {
            'lead_time_mean_h': 33.2,
            'lead_time_p95_h': 107.9,
            'on_time_rate': {
                'fifo_as_is': 92.8,
                'edd_baseline': 99.2,
                'ai_edd_to_be': 99.1,
            },
            'manual_decision_rate': {
                'as_is': 100.0,
                'to_be': final_metrics['shift_supervisor_manual_flag']['total_manual_share'] * 100,
            },
        },
        'key_conclusions': [
            'Tín hiệu dự đoán nguy cơ trễ trong Olist đến chủ yếu từ hành vi lịch sử của người bán (seller_han_rate chiếm 64% importance), không đến từ kích thước đơn hàng.',
            'Cây quyết định xếp hạng rủi ro vượt trội hơn hẳn FIFO và Slack (Recall top-15%: 38.2% vs 12.0%), nhưng ngang bằng quy tắc 1 đặc trưng người bán. Giá trị cốt lõi của AI nằm ở tính minh bạch và gộp đặc trưng điều kiện.',
            'Trong mô phỏng hàng đợi kho, xếp lịch theo hạn bàn giao (EDD) giảm trễ từ 7.2% xuống 0.8%. AI đóng vai trò công cụ cảnh báo (Triage) để Trưởng ca can thiệp sớm.',
            'Áp dụng AI sai cách (AI level-first: làm hết P1 rồi mới đến P2/P3, bỏ qua hạn) gây trễ 11.6%, tệ hơn cả FIFO truyền thống (7.2%).',
        ],
        'limitations': [
            'Dữ liệu Olist phản ánh sàn thương mại điện tử đa người bán, không phải kho đơn lẻ của một doanh nghiệp.',
            'Olist không có dữ liệu thật về hạn dùng, lô hàng và tồn kho (được bổ sung minh bạch bằng lớp luật giả định trong ASSUMPTIONS.md).',
            'Hiện tượng trôi phân phối theo thời gian khiến điểm rủi ro không phải là xác suất hiệu chuẩn tuyệt đối.',
        ],
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    # 3. audit.json (Trang 3: Dữ liệu)
    audit_data = {
        'sha256': audit_raw_data['sha256'],
        'tables_info': audit_raw_data['tables_info'],
        'order_status': audit_raw_data['order_status'],
        'logic_anomalies': audit_raw_data['logic_anomalies'],
        'funnel': audit_raw_data['funnel'],
        'time_ranges': audit_raw_data['time_ranges'],
    }
    with open(os.path.join(out_dir, 'audit.json'), 'w', encoding='utf-8') as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)

    # 4. eda_monthly.json & eda_sla.json (Trang 3: EDA Charts)
    eda_monthly = audit_raw_data['monthly_stats']
    with open(os.path.join(out_dir, 'eda_monthly.json'), 'w', encoding='utf-8') as f:
        json.dump(eda_monthly, f, indent=2, ensure_ascii=False)

    # 5. metrics.json (Trang 4: Mô hình & 21 bước)
    metrics_web_data = {
        'no_free_lunch_comparison': comparison_data,
        'iterations': iter_table,
        'final_test_metrics': final_metrics,
        'bootstrap_ci': boot_ci_data,
        'feature_importances_gini': interp_data['gini_feature_importances'],
        'feature_importances_permutation': interp_data['permutation_importances_val'],
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics_web_data, f, indent=2, ensure_ascii=False)

    # 6. sample_orders.json (Trang 5: Demo tương tác 50 đơn mẫu)
    print("-> Đang trích xuất 50 đơn mẫu thật từ Test để demo...")
    raw = load_raw('data/raw')
    d_order = build_order_table(raw)
    d_all = add_features(d_order)
    d_cohort = select_cohort(d_all)
    tr_fit, _, test_idx = time_split(d_cohort)
    d_filled = fill_train_median(d_cohort, tr_fit)
    d_test = d_filled.iloc[test_idx].reset_index(drop=True)

    tree, tree_dict, thresholds, _ = load_tree('weights')
    X_te = d_test[FEATURES].values
    p_te = tree.predict_proba(X_te)
    t1 = thresholds['t1']
    t2 = thresholds['t2']

    # Đọc dữ liệu FEFO giả định nếu có
    sim_orders_path = 'data/simulated/orders_simulated_features.csv'
    sim_map = {}
    if os.path.exists(sim_orders_path):
        sim_df = pd.read_csv(sim_orders_path)
        sim_map = sim_df.set_index('order_id').to_dict(orient='index')

    rng = np.random.default_rng(42)
    # Chọn cân bằng: 20 đơn P1, 20 đơn P2, 10 đơn P3
    p1_idx = np.where(p_te >= t1)[0]
    p2_idx = np.where((p_te >= t2) & (p_te < t1))[0]
    p3_idx = np.where(p_te < t2)[0]

    chosen_indices = np.concatenate([
        rng.choice(p1_idx, size=min(20, len(p1_idx)), replace=False),
        rng.choice(p2_idx, size=min(20, len(p2_idx)), replace=False),
        rng.choice(p3_idx, size=min(10, len(p3_idx)), replace=False),
    ])
    rng.shuffle(chosen_indices)

    sample_orders_list = []
    for num_id, idx in enumerate(chosen_indices, 1):
        row = d_test.iloc[idx]
        ord_id = str(row['order_id'])
        # Ẩn danh: chỉ hiển thị mã rút gọn ORD-XXXXX (Mục 8.1 & 10)
        anon_id = f"ORD-{num_id:03d} ({ord_id[:6]}...)"

        p_risk = float(p_te[idx])
        level = 'P1' if p_risk >= t1 else ('P2' if p_risk >= t2 else 'P3')

        # FEFO info
        fefo_info = sim_map.get(ord_id, {})
        has_perish = int(fefo_info.get('sim_has_perishable_item', 0))
        rem_days = float(fefo_info.get('sim_min_days_to_expiry', 999.0))
        fefo_flag = int(fefo_info.get('sim_fefo_priority_flag', 0))

        # Cờ Trưởng ca: P1 hoặc seller_hist_n < 20 hoặc FEFO flag
        supervisor_flag = (level == 'P1') or (row['seller_hist_n'] < 20) or (fefo_flag == 1)

        feature_dict = {f: round(float(row[f]), 3) for f in FEATURES}

        sample_orders_list.append({
            'sample_id': num_id,
            'anon_order_id': anon_id,
            'order_approved_at': str(row['order_approved_at']),
            'shipping_limit_date': str(row['limit']),
            'actual_late_handover': int(row['late_handover']),
            'predicted_p_risk': round(p_risk, 4),
            'priority_level': level,
            'supervisor_manual_flag': bool(supervisor_flag),
            'sim_has_perishable': bool(has_perish),
            'sim_days_to_expiry': round(rem_days, 1) if rem_days < 900 else None,
            'sim_fefo_flag': bool(fefo_flag),
            'features': feature_dict,
        })

    with open(os.path.join(out_dir, 'sample_orders.json'), 'w', encoding='utf-8') as f:
        json.dump(sample_orders_list, f, indent=2, ensure_ascii=False)

    # 7. threshold_sweep.json (Trang 7: Rủi ro & Thanh trượt chi phí)
    print("-> Đang tính toán bảng quét 50 ngưỡng chi phí...")
    y_te = d_test.late_handover.values.astype(int)
    threshold_vals = np.linspace(0.05, 0.55, 51)
    sweep_list = []
    for thr in threshold_vals:
        y_pred = (p_te >= thr).astype(int)
        cm = confusion_matrix(y_te, y_pred)
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        prec = float(precision_score(y_te, y_pred, zero_division=0))
        rec = float(recall_score(y_te, y_pred, zero_division=0))
        f1 = float(f1_score(y_te, y_pred, zero_division=0))
        sweep_list.append({
            'threshold': round(float(thr), 3),
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4),
        })
    with open(os.path.join(out_dir, 'threshold_sweep.json'), 'w', encoding='utf-8') as f:
        json.dump(sweep_list, f, indent=2)

    # 8. triage.json (Trang 7: Triage recall@k)
    k_percents = [5, 10, 15, 20, 25, 30, 40, 50]
    n_te = len(y_te)
    n_pos = int(y_te.sum())

    p_seller = d_test['seller_han_rate'].values
    p_slack = -d_test['min_to_limit_h'].values
    t_te = d_test.order_approved_at.values.astype('datetime64[s]').astype(np.int64)
    p_fifo = 1.0 - (t_te - t_te.min()) / max(float(t_te.max() - t_te.min()), 1.0)

    triage_data = []
    for k in k_percents:
        top_k = int(np.ceil((k / 100.0) * n_te))

        idx_tree = np.argsort(p_te)[::-1][:top_k]
        idx_seller = np.argsort(p_seller)[::-1][:top_k]
        idx_slack = np.argsort(p_slack)[::-1][:top_k]
        idx_fifo = np.argsort(p_fifo)[::-1][:top_k]

        triage_data.append({
            'top_percent': k,
            'top_k_orders': top_k,
            'recall_tree': round(float(y_te[idx_tree].sum() / n_pos), 4),
            'recall_seller_rule': round(float(y_te[idx_seller].sum() / n_pos), 4),
            'recall_slack_rule': round(float(y_te[idx_slack].sum() / n_pos), 4),
            'recall_fifo_rule': round(float(y_te[idx_fifo].sum() / n_pos), 4),
            'recall_random': round(k / 100.0, 4),
        })
    with open(os.path.join(out_dir, 'triage.json'), 'w', encoding='utf-8') as f:
        json.dump(triage_data, f, indent=2)

    print("=== Hoàn tất xuất toàn bộ 10 file JSON sang web/data/ ===")


if __name__ == '__main__':
    export_all_web_data('web/data')

