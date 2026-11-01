# Đồ án CK10: Thiết Kế Chuỗi Dịch Vụ Logistics Tự Động Có AI Ra Quyết Định — Nghiệp Vụ Kho (Inventory)

**Môn học:** E-logistics  
**Sản phẩm:** Repo code (NumPy thuần) + Notebook + Website trình bày / mô phỏng tương tác  
**Tập dữ liệu:** Olist Brazilian E-Commerce bản gốc (Kaggle `olistbr/brazilian-ecommerce`, Version 2)

---

## 1. Giới thiệu Đề tài

Dự án tập trung vào nghiệp vụ Kho (Inventory) trong chuỗi dịch vụ E-logistics:
- **Bài toán cốt lõi:** Dự đoán rủi ro trễ bàn giao đơn hàng cho đơn vị vận tải (`late_handover`) tại thời điểm duyệt đơn (`order_approved_at`).
- **Ra quyết định tự động:** Phân loại đơn hàng theo 3 mức ưu tiên (**P1 - Cao**, **P2 - Trung bình**, **P3 - Thấp**).
- **Hỗ trợ con người:** Đưa ra cơ chế cảnh báo Trưởng ca (Triage) đối với các đơn hàng rủi ro cao hoặc người bán chưa đủ độ tin cậy lịch sử, kết hợp quy tắc FEFO giả định.
- **Mô phỏng As-Is vs To-Be:** So sánh hiệu quả vận hành giữa chính sách truyền thống (FIFO), chính sách theo hạn (EDD) và chính sách có AI ưu tiên (AI-EDD) trên 18,765 đơn hàng thực tế.

---

## 2. Cấu trúc Thư mục Dự án

```
ck10-olist-priority/
├── HUONG_DAN.md                  # Hướng dẫn chi tiết đồ án
├── README.md                     # Tài liệu tổng quan dự án
├── AI_USAGE_LOG.md               # Nhật ký minh bạch việc sử dụng trợ lý AI
├── requirements.txt              # Danh sách thư viện và phiên bản
├── .gitignore                    # Cấu hình bỏ qua dữ liệu thô và file tạm
├── data/
│   └── raw/                      # 5 file CSV gốc Olist (không commit)
├── src/                          # Mã nguồn chính (ML thuần NumPy)
│   ├── __init__.py
│   ├── data_prep.py              # Đọc, tiền xử lý, trích xuất đặc trưng
│   ├── model.py                  # Mô hình cây quyết định, RF, LogR thuần NumPy
│   ├── evaluate.py               # Đánh giá chỉ số, bootstrap, phân tích lỗi
│   ├── simulate.py               # Mô phỏng hàng đợi kho As-Is vs To-Be
│   ├── weights.py                # Lưu và nạp trọng số mô hình
│   └── export_web.py             # Xuất dữ liệu JSON phục vụ web
├── run_pipeline.py               # Pipeline thực thi 21 bước tuần tự
├── predict_new.py                # Suy luận đơn hàng mới từ file CSV
├── build_notebook.py             # Sinh và thực thi notebook.ipynb tự động
├── notebook.ipynb                # Báo cáo Jupyter Notebook
├── tests/                        # Bộ kiểm thử đơn vị và kiểm toán rò rỉ dữ liệu
├── weights/                      # Trọng số và ngưỡng cắt đã huấn luyện
├── outputs/                      # Báo cáo JSON, số liệu EDA và đồ thị sinh tự động
└── web/                          # Website báo cáo và demo tương tác (GitHub Pages)
```

---

## 3. Quy tắc Kỹ thuật Bắt buộc (Mục 2 & Mục 10)

1. **Dữ liệu gốc nguyên vẹn:** Không sửa, không ghi đè, không xóa trực tiếp trên file CSV trong `data/raw/`.
2. **Không rò rỉ dữ liệu:** Chỉ sử dụng thông tin biết được tại `order_approved_at`. Đặc trưng lịch sử chỉ tổng hợp từ các đơn đã hoàn tất giao hàng trước thời điểm duyệt đơn.
3. **ML thuần NumPy:** Lõi mô hình được cài đặt bằng NumPy thuần túy trong `src/`. Thư viện `scikit-learn` chỉ được dùng trong `tests/` để đối chiếu độ chuẩn xác.
4. **Chia tập theo thời gian:** Dùng phân chia theo chuỗi thời gian (`time_split`) làm kết quả chính nhằm phản ánh đúng điều kiện vận hành thực tế.
5. **Báo cáo trung thực:** Công bố đầy đủ các hạn chế và số liệu thực nghiệm thực tế từ `outputs/`.

---

## 4. Hướng dẫn Cài đặt & Chạy Thử

### Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### Chạy kiểm thử
```bash
pytest
```

