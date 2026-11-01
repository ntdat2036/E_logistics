"""weights.py - Quản lý lưu và nạp cấu trúc cây, siêu tham số và ngưỡng (Mục 5, Mục 8.3, Phụ lục D).
TUÂN THỦ TUYỆT ĐỐI MỤC 2 (R2): KHÔNG import scikit-learn trong src/.
"""
import json
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
from src.model import DecisionTree, BinMapper


def export_tree_dict(tree, feature_names, t1, t2):
    """Xuất cây sang định dạng dict / JSON theo Phụ lục D.
    Quy ước rẽ chuẩn: x[f] < thr -> đi trái (thr là giá trị thực tế, không phải chỉ số bin).
    """
    nodes = []
    for i in range(len(tree.feat)):
        leaf = tree.feat[i] < 0
        nodes.append({
            'f': int(tree.feat[i]),
            'thr': None if leaf else tree.threshold_value(i),
            'l': int(tree.left[i]),
            'r': int(tree.right[i]),
            'p': float(tree.val[i]),
            'n': int(tree.cnt[i]),
        })
    return {
        'feature_names': list(feature_names),
        'nodes': nodes,
        't1': float(t1),
        't2': float(t2),
        'rule': 'x[f] < thr -> left',
    }


def predict_tree_dict(model_dict, x):
    """Suy luận từ dict JSON: duyệt cây từ gốc theo quy ước x[f] < thr -> rẽ trái."""
    nodes = model_dict['nodes']
    i = 0
    while nodes[i]['f'] >= 0:
        f = nodes[i]['f']
        thr = nodes[i]['thr']
        i = nodes[i]['l'] if x[f] < thr else nodes[i]['r']
    return float(nodes[i]['p'])


def save_tree(tree, feature_names, thresholds, best_params, weights_dir='weights'):
    """Lưu toàn bộ mô hình cây, ngưỡng và tham số vào weights/."""
    os.makedirs(weights_dir, exist_ok=True)
    t1 = thresholds.get('t1', 0.15)
    t2 = thresholds.get('t2', 0.06)

    # 1. tree.json
    tree_dict = export_tree_dict(tree, feature_names, t1, t2)
    with open(os.path.join(weights_dir, 'tree.json'), 'w', encoding='utf-8') as f:
        json.dump(tree_dict, f, indent=2, ensure_ascii=False)

    # 2. thresholds.json
    with open(os.path.join(weights_dir, 'thresholds.json'), 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)

    # 3. best_params.json
    with open(os.path.join(weights_dir, 'best_params.json'), 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)

    # 4. feature_names.json
    with open(os.path.join(weights_dir, 'feature_names.json'), 'w', encoding='utf-8') as f:
        json.dump(list(feature_names), f, indent=2, ensure_ascii=False)

    # 5. tree.npz (lưu trữ mảng số nguyên vẹn cho Python)
    edge_arrays = {f'edge_{j}': tree.bm_.edges_[j] for j in range(len(tree.bm_.edges_))}
    np.savez_compressed(
        os.path.join(weights_dir, 'tree.npz'),
        feat=tree.feat,
        thr=tree.thr,
        left=tree.left,
        right=tree.right,
        val=tree.val,
        cnt=tree.cnt,
        feature_importances=getattr(tree, 'feature_importances_', np.zeros(len(feature_names))),
        n_features=len(feature_names),
        **edge_arrays,
    )

    # 6. README.md
    with open(os.path.join(weights_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write("# Trọng số và Ngưỡng Mô hình (Weights & Thresholds)\n\n")
        f.write("Thư mục chứa toàn bộ mô hình và ngưỡng phân định đã được huấn luyện và khóa cứng:\n\n")
        f.write("- `tree.json`: Cấu trúc cây quyết định dạng JSON dùng cho JavaScript trên web và predict_new.py.\n")
        f.write("- `thresholds.json`: Các ngưỡng phân vị (t1, t2), ngưỡng tối ưu F1, prior và trung vị điền thiếu.\n")
        f.write("- `best_params.json`: Bộ siêu tham số tốt nhất qua GridSearch CV expanding-window.\n")
        f.write("- `feature_names.json`: Danh sách 20 đặc trưng đầu vào.\n")
        f.write("- `tree.npz`: Định dạng nén nhị phân NumPy phục vụ tải nhanh trong môi trường Python.\n")

    return tree_dict


def load_tree(weights_dir='weights'):
    """Nạp lại mô hình cây từ weights/ và tái tạo đối tượng DecisionTree hoàn chỉnh."""
    npz_path = os.path.join(weights_dir, 'tree.npz')
    json_path = os.path.join(weights_dir, 'tree.json')
    thr_path = os.path.join(weights_dir, 'thresholds.json')
    bp_path = os.path.join(weights_dir, 'best_params.json')

    with open(thr_path, 'r', encoding='utf-8') as f:
        thresholds = json.load(f)

    best_params = {}
    if os.path.exists(bp_path):
        with open(bp_path, 'r', encoding='utf-8') as f:
            best_params = json.load(f)

    with open(json_path, 'r', encoding='utf-8') as f:
        tree_dict = json.load(f)

    # Tái tạo đối tượng DecisionTree từ file npz
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        tree = DecisionTree(**best_params)
        tree.feat = data['feat']
        tree.thr = data['thr']
        tree.left = data['left']
        tree.right = data['right']
        tree.val = data['val']
        tree.cnt = data['cnt']
        tree.feature_importances_ = data['feature_importances']

        bm = BinMapper(best_params.get('n_bins', 64))
        n_feats = int(data['n_features'])
        bm.edges_ = [data[f'edge_{j}'] for j in range(n_feats)]
        tree.bm_ = bm
    else:
        # Tái tạo nhẹ từ tree_dict
        tree = None

    return tree, tree_dict, thresholds, best_params


def verify_roundtrip(tree, X_sample, weights_dir='weights'):
    """Kiểm tra khứ hồi (roundtrip test): Lưu -> nạp lại -> so sánh dự đoán trên X_sample (sai số phải = 0.0)."""
    p_orig = tree.predict_proba(X_sample)

    loaded_tree, tree_dict, _, _ = load_tree(weights_dir)
    p_loaded = loaded_tree.predict_proba(X_sample)
    p_dict = np.array([predict_tree_dict(tree_dict, row) for row in X_sample])

    diff_loaded = np.max(np.abs(p_orig - p_loaded))
    diff_dict = np.max(np.abs(p_orig - p_dict))

    assert diff_loaded == 0.0, f"Roundtrip thất bại trên loaded_tree! max_diff={diff_loaded}"
    assert diff_dict == 0.0, f"Roundtrip thất bại trên predict_tree_dict! max_diff={diff_dict}"
    return True

