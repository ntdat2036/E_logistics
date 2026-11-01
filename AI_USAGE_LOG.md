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

### Giai đoạn 2: Đặc trưng và Chia dữ liệu (Các bước 7–10)
- **Thời gian:** 2026-11-01
- **Người giám sát / Phê duyệt:** Nhóm sinh viên CK10
- **Nội dung thực hiện bởi AI:**
  1. Bổ sung hàm `stratified_split` thuần NumPy trong `src/data_prep.py` để chia dữ liệu ngẫu nhiên phân tầng 80/20 (phục vụ đối chiếu kết quả "lạc quan" theo Mục 4.4 so với chia theo chuỗi thời gian).
  2. Xây dựng hàm `run_leakage_audit` kiểm toán rò rỉ dữ liệu in ra màn hình terminal theo chuẩn `[PASS]`/`[FAIL]`:
     - Kiểm tra không trùng lặp chỉ mục giữa train_fit, val, test.
     - Kiểm tra tính tuần tự thời gian: test hoàn toàn sau train.
     - Kiểm tra Knowledge-Time: so sánh tỷ lệ trễ lịch sử với vòng lặp brute-force trên 1,000 đơn mẫu ngẫu nhiên (chênh lệch tuyệt đối = 0.00e+00).
     - Kiểm tra hằng số tiên nghiệm prior chỉ tính từ các đơn trước mốc 2018-05-01.
     - Kiểm tra giá trị trung vị điền thiếu chỉ tính từ tập train_fit.
     - Kiểm tra sự cô lập độc lập của tập validation và test.
  3. Xây dựng bộ kiểm thử `tests/test_leakage.py` gồm 7 bài test chuyên biệt kiểm tra toàn diện các nguy cơ rò rỉ thông tin tương lai.
  4. Thực thi kiểm thử toàn bộ dự án qua `pytest`: 11/11 bài test vượt qua (100% PASS).
- **Trạng thái:** Hoàn thành nghiệm thu Giai đoạn 2.

### Giai đoạn 3: Mô hình và Baseline (Các bước 11–13)
- **Thời gian:** 2026-11-01
- **Người giám sát / Phê duyệt:** Nhóm sinh viên CK10
- **Nội dung thực hiện bởi AI:**
  1. Xây dựng module học máy lõi `src/model.py` thuần NumPy 100% (tuân thủ nghiêm ngặt R2, không import scikit-learn):
     - `BinMapper`: Ánh xạ quantile bins (32 hoặc 64).
     - `DecisionTree`: Cây quyết định phân loại nhị phân hỗ trợ Gini và Entropy, `min_samples_split`, `max_features` ('sqrt', int, float), `class_weight`, xuất cấu trúc nút và ngưỡng giá trị thật `threshold_value`, tính `feature_importances_`.
     - `RandomForest`: Rừng ngẫu nhiên thuần NumPy (bagging ensemble, `max_features='sqrt'`).
     - `LogisticRegression`: Hồi quy logistic tối ưu hóa bằng Gradient Descent thuần NumPy, chuẩn hóa L2, hỗ trợ `class_weight='balanced'`, theo dõi `loss_history_`.
     - `StandardScaler`: Chuẩn hóa $Z$-score (fit trên train, transform val/test).
     - Toàn bộ bộ đo lường metrics thuần NumPy: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `brier_score_loss`, `roc_auc_score` (tính qua xếp hạng Mann-Whitney U có xử lý ties), `average_precision_score` (PR-AUC).
     - Bộ chia cross-validation thuần NumPy: `TimeSeriesSplit` (expanding-window) và `StratifiedKFold`.
     - `GridSearch` thuần NumPy.
  2. Xây dựng module đánh giá `src/evaluate.py`:
     - Chạy so sánh 7 mô hình No Free Lunch (Mục 4.5): Dummy, FIFO, Slack, Seller History, Tree Order Only, Logistic Regression, Decision Tree lõi, Random Forest.
     - Xuất kết quả so sánh ra `outputs/03_model_comparison.json` và `outputs/03_model_comparison.md`.
     - Kết quả đo trên test: Cây lõi ROC-AUC = 0.6666 (khớp 0.667 tham chiếu), PR-AUC = 0.1541 (khớp 0.154); Quy tắc lịch sử người bán = 0.6712 (khớp 0.671); FIFO = 0.4646; Slack = 0.4866; Cây không lịch sử = 0.5183.
     - Thực hiện phân tích Bias - Variance: Quét `max_depth` từ 1 đến 12, xuất đồ thị `outputs/fig_bias_variance.png` và dữ liệu `outputs/03_bias_variance.json`.
  3. Viết và thực thi 2 bộ kiểm thử:
     - `tests/test_metrics.py`: Đối chiếu các hàm metric thuần NumPy với `scikit-learn` trên 5 mảng nhỏ và mảng ngẫu nhiên (chênh lệch < 1e-9).
     - `tests/test_tree_vs_sklearn.py`: Đối chiếu cây thuần NumPy với `DecisionTreeClassifier` của sklearn (chênh lệch AUC < 0.01; chứng minh tính tất định).
  4. Chạy toàn bộ test suite dự án qua `pytest`: 20/20 bài test vượt qua (100% PASS).
- **Trạng thái:** Hoàn thành nghiệm thu Giai đoạn 3.




