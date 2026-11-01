"""test_web_parity.py - Kiểm thử tương thích suy luận giữa Python và JavaScript (Mục 9.1).
Lấy 200 đơn ngẫu nhiên từ Test: so sánh điểm rủi ro tính từ Python và từ logic JavaScript (web/js/tree.js).
Độ chênh lệch bắt buộc < 1e-9 (thực tế sai số = 0.0 nhờ quy ước x[f] < thr).
"""
import json
import os
import shutil
import subprocess
import numpy as np
import pytest
from src.weights import load_tree, predict_tree_dict
from src.data_prep import load_raw, build_order_table, add_features, select_cohort, time_split, fill_train_median, FEATURES


def test_tree_json_parity_200_samples():
    """Kiểm tra 200 mẫu ngẫu nhiên: logic duyệt JSON khớp 100% với predict_proba."""
    tree, tree_dict, thresholds, _ = load_tree('weights')

    raw = load_raw('data/raw')
    c = select_cohort(add_features(build_order_table(raw)))
    tr_fit, _, test_idx = time_split(c)
    c = fill_train_median(c, tr_fit)
    X_test = c.iloc[test_idx][FEATURES].values

    rng = np.random.default_rng(2026)
    sample_idx = rng.choice(len(X_test), size=200, replace=False)
    X_sample = X_test[sample_idx]

    # Dự đoán bằng đối tượng Python
    p_py = tree.predict_proba(X_sample)

    # Dự đoán bằng logic JSON (như JavaScript)
    p_json = np.array([predict_tree_dict(tree_dict, row) for row in X_sample])

    diff = np.max(np.abs(p_py - p_json))
    assert diff < 1e-9, f"Sai số giữa Python và JSON vượt ngưỡng: {diff}"
    assert diff == 0.0, f"Sai số phải bằng 0.0 tuyệt đối, thực tế {diff}"


def test_nodejs_tree_js_parity_if_node_available():
    """Nếu môi trường có cài đặt Node.js, chạy trực tiếp file web/js/tree.js để so sánh."""
    node_bin = shutil.which('node')
    if not node_bin:
        pytest.skip("Node.js không có sẵn trong môi trường Windows PATH, bỏ qua gọi subprocess node.")

    tree, tree_dict, _, _ = load_tree('weights')
    raw = load_raw('data/raw')
    c = select_cohort(add_features(build_order_table(raw)))
    tr_fit, _, test_idx = time_split(c)
    c = fill_train_median(c, tr_fit)
    X_test = c.iloc[test_idx][FEATURES].values[:50]

    p_py = tree.predict_proba(X_test)

    # Gọi Node.js
    js_code = f"""
    const {{ predictTree }} = require('./web/js/tree.js');
    const model = {json.dumps(tree_dict)};
    const X = {json.dumps(X_test.tolist())};
    const res = X.map(x => predictTree(model, x).p);
    console.log(JSON.stringify(res));
    """
    proc = subprocess.run([node_bin, '-e', js_code], capture_output=True, text=True, check=True)
    p_js = np.array(json.loads(proc.stdout.strip()))

    diff = np.max(np.abs(p_py - p_js))
    assert diff < 1e-9, f"Sai số giữa Python và Node.js: {diff}"

