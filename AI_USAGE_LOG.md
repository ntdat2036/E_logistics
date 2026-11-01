# NHẬT KÝ SỬ DỤNG TRỢ LÝ AI (AI USAGE LOG)

Tài liệu này ghi lại một cách minh bạch toàn bộ các hoạt động, quyết định kỹ thuật và hỗ trợ từ AI (Antigravity Assistant) trong quá trình thực hiện Đồ án CK10 (E-logistics), tuân thủ tuyệt đối quy định về liêm chính học thuật (Mục 2 - R5).

---

## 1. Nguyên tắc sử dụng AI trong dự án
- **Quy tắc bắt buộc:**
  - Không bịa đặt tài liệu tham khảo, số liệu thực nghiệm.
  - Mã nguồn Machine Learning lõi (`src/model.py`) được lập trình thuần túy bằng NumPy, không dùng `scikit-learn` trong `src/`.
  - Mọi con số trong báo cáo, bảng biểu và website demo được đọc từ kết quả tính toán thực tế (`outputs/*.json`).
  - Dữ liệu gốc Olist giữ nguyên vẹn, không chỉnh sửa trực tiếp.
  - Phân chia tập dữ liệu theo thứ tự thời gian (`time_split`) để tránh rò rỉ thông tin tương lai.
  - Toàn bộ kết quả thực nghiệm (kể cả kết quả kém hoặc hạn chế) được báo cáo trung thực.

---

## 2. Nhật ký công việc theo từng giai đoạn

### Giai đoạn 0: Khởi tạo cấu trúc dự án & Thiết lập môi trường
- **Thời gian:** 2026-11-01
- **Người giám sát / Phê duyệt:** Nhóm sinh viên CK10
- **Nội dung thực hiện bởi AI:**
  1. Đọc và phân tích toàn bộ yêu cầu trong `HUONG_DAN.md` (bao gồm các quy tắc Mục 2, cấu trúc Mục 5, các giai đoạn Mục 6, và các điều cấm Mục 10).
  2. Khảo sát dữ liệu gốc ban đầu trong thư mục `dataset/`, kiểm tra số lượng dòng và mã băm SHA256 để đảm bảo tính toàn vẹn.
  3. Lên kế hoạch chi tiết cho Giai đoạn 0 và trình nhóm phê duyệt trước khi thao tác.
  4. Tạo cấu trúc cây thư mục chuẩn: `data/raw/`, `src/`, `tests/`, `weights/`, `outputs/`, `web/`.
  5. Di chuyển 5 file CSV gốc vào `data/raw/`, thiết lập file `data/raw/README.md` lưu ý bản quyền và cấm commit dữ liệu thô.
  6. Thiết lập `.gitignore` bảo vệ dữ liệu gốc và các file tạm.
  7. Thiết lập `requirements.txt` ghi rõ phiên bản và ghi chú cấm import `sklearn` ngoài `tests/`.
  8. Khởi tạo `src/__init__.py`, `README.md` khung dự án.
  9. Kiểm tra môi trường thực thi: cài đặt `pytest` và chạy kiểm thử nghiệm thu Giai đoạn 0.
- **Trạng thái:** Hoàn thành nghiệm thu Giai đoạn 0.

### Giai đoạn 1: Dữ liệu (Các bước 1–6)
- **Thời gian:** 2026-11-01
- **Người giám sát / Phê duyệt:** Nhóm sinh viên CK10
- **Nội dung thực hiện bởi AI:**
  1. Cài đặt module `src/data_prep.py` theo đúng mã nguồn tham chiếu Phụ lục A và các yêu cầu kiểm toán:
     - Xây dựng hàm `load_raw` đọc 5 file CSV gốc.
     - Hàm `build_order_table`: Tổng hợp bảng cấp đơn hàng, lọc đơn `delivered`, 1 người bán, đủ 3 mốc thời gian; tính 2 nhãn thật (`late_handover` và `late_delivery`); gắn cờ lỗi logic `ts_anomaly` (không xóa dòng).
     - Hàm `add_features`: Trích xuất 20 đặc trưng (thời hạn, đơn hàng, địa lý, thời gian, tải hệ thống 24h, lịch sử seller/tuyến với kiến trúc chống rò rỉ knowledge-time).
     - Hàm `select_cohort`: Lọc cửa sổ 2017-01-01 đến 2018-07-31 (loại bỏ 2016 quá ít đơn và tháng 08/2018 bị kiểm duyệt phải).
     - Hàm `fill_train_median`: Điền giá trị khuyết thiếu chỉ dựa trên trung vị tập train_fit.
     - Hàm `time_split`: Chia chuỗi thời gian thành train_fit (55,808 đơn), validation (13,953 đơn), test (18,765 đơn).
  2. Thực hiện kiểm toán dữ liệu `audit_raw`:
     - Kiểm tra SHA256 của 5 file gốc: xác nhận dữ liệu thô nguyên vẹn 100%.
     - Kiểm tra phân phối kích thước và null khớp hoàn toàn với bảng Mục 3.
     - Ghi nhận 166 đơn có ngày giao vận tải trước ngày đặt, 23 đơn nhận trước ngày giao vận tải; 117 đơn mang cờ `ts_anomaly` trong cohort.
     - Xuất báo cáo kiểm toán đầy đủ ra `outputs/01_data_audit.json` và `outputs/01_data_audit.md`.
  3. Vẽ và xuất 4 biểu đồ EDA (`outputs/fig_*.png`):
     - `fig_orders_monthly.png`: Số lượng đơn theo tháng (làm nổi bật cohort).
     - `fig_delay_shock.png`: Tỷ lệ trễ bàn giao và trễ giao khách theo tháng (thể hiện cú sốc Black Friday 11/2017, biến động vận tải 02-03/2018, và hiện tượng tụt dốc giả do kiểm duyệt phải tháng 08/2018).
     - `fig_sla_dist.png`: Phân phối `min_to_limit_h` và `sla_days`.
     - `fig_delay_by_sla.png`: Tỷ lệ trễ theo các nhóm SLA cam kết.
  4. Viết và thực thi bộ kiểm thử `tests/test_data_prep.py`:
     - Xác nhận phễu lọc 99,441 → 95,180 → 88,526.
     - Xác nhận nhãn nhị phân và các đặc trưng không có NaN sau khi điền trung vị train.
     - Xác nhận 4/4 bài kiểm thử vượt qua.
- **Trạng thái:** Hoàn thành nghiệm thu Giai đoạn 1.


