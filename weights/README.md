# Trọng số và Ngưỡng Mô hình (Weights & Thresholds)

Thư mục chứa toàn bộ mô hình và ngưỡng phân định đã được huấn luyện và khóa cứng:

- `tree.json`: Cấu trúc cây quyết định định dạng JSON dùng cho JavaScript trên Web và script `predict_new.py`. Ngưỡng phân nhánh là giá trị thực (`threshold_value`), quy ước rẽ: `x[f] < thr -> left`.
- `thresholds.json`: Các ngưỡng phân vị khóa trên tập Validation (`t1 = 0.1883`, `t2 = 0.0651`), ngưỡng tối ưu $F_1$ (`t_f1 = 0.1799`), hằng số tiên nghiệm `prior_late_handover`, trung vị điền thiếu của tập train `train_medians`, và ngưỡng chi phí tối ưu `optimal_cost_threshold`.
- `best_params.json`: Bộ siêu tham số tốt nhất qua GridSearch CV expanding-window (`max_depth: 6, min_samples_leaf: 100, criterion: 'gini', n_bins: 32, min_samples_split: 200`).
- `feature_names.json`: Danh sách 20 đặc trưng đầu vào chuẩn hóa.
- `tree.npz`: Định dạng nén nhị phân NumPy phục vụ tải nhanh trong môi trường Python.
