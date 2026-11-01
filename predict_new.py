"""predict_new.py - Nhận CSV đơn mới -> dự đoán điểm rủi ro, mức ưu tiên P1/P2/P3 và cờ Trưởng ca.
Sử dụng mô hình cây quyết định đã huấn luyện và khóa cứng trong weights/.
TUÂN THỦ TUYỆT ĐỐI MỤC 2 (R2): Thuần NumPy / JSON, không import scikit-learn.
"""
import argparse
import json
import os
import sys

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
from src.weights import predict_tree_dict


def predict_orders_dataframe(df, weights_dir='weights'):
    """Dự đoán cho một DataFrame đơn hàng đã có đủ 20 đặc trưng FEATURES."""
    tree_path = os.path.join(weights_dir, 'tree.json')
    thr_path = os.path.join(weights_dir, 'thresholds.json')

    if not os.path.exists(tree_path) or not os.path.exists(thr_path):
        raise FileNotFoundError(f"Không tìm thấy mô hình trong {weights_dir}! Hãy chạy train.py trước.")

    with open(tree_path, 'r', encoding='utf-8') as f:
        tree_dict = json.load(f)
    with open(thr_path, 'r', encoding='utf-8') as f:
        thresholds = json.load(f)

    feature_names = tree_dict['feature_names']
    t1 = thresholds['t1']
    t2 = thresholds['t2']

    # Kiểm tra cột thiếu
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Dữ liệu thiếu các cột đặc trưng bắt buộc: {missing}")

    X = df[feature_names].values.astype(float)
    p_risks = np.array([predict_tree_dict(tree_dict, row) for row in X])

    priority_levels = np.where(p_risks >= t1, 'P1', np.where(p_risks >= t2, 'P2', 'P3'))

    # Cờ can thiệp thủ công Trưởng ca (P1 hoặc seller_hist_n < 20 hoặc sim_fefo_priority_flag == 1)
    low_hist = df['seller_hist_n'].values < 20 if 'seller_hist_n' in df.columns else False
    fefo_flag = df['sim_fefo_priority_flag'].values == 1 if 'sim_fefo_priority_flag' in df.columns else False
    manual_flags = (priority_levels == 'P1') | low_hist | fefo_flag

    res_df = df.copy()
    res_df['predicted_risk_score'] = np.round(p_risks, 4)
    res_df['priority_level'] = priority_levels
    res_df['supervisor_manual_flag'] = manual_flags

    return res_df


def main():
    parser = argparse.ArgumentParser(description="Dự đoán mức ưu tiên và rủi ro trễ bàn giao cho đơn hàng mới.")
    parser.add_argument('csv_path', nargs='?', default=None, help="Đường dẫn file CSV chứa các đơn hàng mới")
    parser.add_argument('--output', '-o', default=None, help="Đường dẫn file CSV xuất kết quả")
    args = parser.parse_args()

    if args.csv_path is not None and os.path.exists(args.csv_path):
        df_new = pd.read_csv(args.csv_path)
        print(f"Đang đọc dữ liệu từ: {args.csv_path} ({len(df_new)} dòng)...")
    else:
        print("Không có file CSV truyền vào -> Đang đọc 10 đơn mẫu từ web/data/sample_orders.json...")
        sample_path = 'web/data/sample_orders.json'
        with open(sample_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)[:10]
        # Chuyển đổi thành DataFrame
        rows = []
        for s in samples:
            r = {'order_id': s['anon_order_id']}
            r.update(s['features'])
            if 'sim_fefo_flag' in s:
                r['sim_fefo_priority_flag'] = 1 if s['sim_fefo_flag'] else 0
            rows.append(r)
        df_new = pd.DataFrame(rows)

    res = predict_orders_dataframe(df_new, weights_dir='weights')

    display_cols = ['order_id', 'predicted_risk_score', 'priority_level', 'supervisor_manual_flag']
    avail_cols = [c for c in display_cols if c in res.columns]

    print("\n========================= KẾT QUẢ DỰ ĐOÁN ĐƠN HÀNG MỚI =========================")
    print(res[avail_cols].to_string(index=False))
    print("=================================================================================\n")

    if args.output:
        res.to_csv(args.output, index=False)
        print(f"Đã lưu kết quả dự đoán chi tiết vào: {args.output}")


if __name__ == '__main__':
    main()
