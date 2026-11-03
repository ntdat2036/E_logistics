# Trọng số và Ngưỡng Mô hình (Weights & Thresholds)

Thư mục chứa toàn bộ mô hình và ngưỡng phân định đã được huấn luyện và khóa cứng:

- `tree.json`: Cấu trúc cây quyết định dạng JSON dùng cho JavaScript trên web và predict_new.py.
- `thresholds.json`: Các ngưỡng phân vị (t1, t2), ngưỡng tối ưu F1, prior và trung vị điền thiếu.
- `best_params.json`: Bộ siêu tham số tốt nhất qua GridSearch CV expanding-window.
- `feature_names.json`: Danh sách 20 đặc trưng đầu vào.
- `tree.npz`: Định dạng nén nhị phân NumPy phục vụ tải nhanh trong môi trường Python.
