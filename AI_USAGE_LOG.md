# NHẬT KÝ SỬ DỤNG TRỢ LÝ AI (AI USAGE LOG)


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
  1. Đọc và phân tích toàn bộ yêu cầu trong `tài liệu hướng dẫn đồ án` (bao gồm các quy tắc Mục 2, cấu trúc Mục 5, các giai đoạn Mục 6, và các điều cấm Mục 10).
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

### Giai đoạn 4: Tuning, CV, Huấn luyện cuối (Các bước 14–17)
- **Thời gian:** 2026-11-01
- **Người giám sát / Phê duyệt:** Nhóm sinh viên CK10
- **Nội dung thực hiện bởi AI:**
  1. Xây dựng module `src/weights.py`:
     - `export_tree_dict` & `predict_tree_dict`: Xuất cấu trúc cây ra JSON theo Phụ lục D, quy ước rẽ chuẩn `x[f] < thr -> left` với ngưỡng thực tế `threshold_value`.
     - `save_tree`: Lưu trữ cấu trúc cây (`weights/tree.json`), các ngưỡng đã khóa (`weights/thresholds.json`), siêu tham số (`weights/best_params.json`), danh sách đặc trưng (`weights/feature_names.json`), và mảng nhị phân `weights/tree.npz`.
     - `load_tree`: Nạp lại mô hình cây hoàn chỉnh trong môi trường Python.
     - `verify_roundtrip`: Kiểm tra khứ hồi (dự đoán trước và sau khi lưu khớp tuyệt đối, sai số 0.0).
  2. Viết module thực thi huấn luyện và tuning `src/train.py`:
     - Chạy GridSearch trên 80 tổ hợp siêu tham số qua Expanding-window 4-fold và Stratified 5-fold (tiêu chí PR-AUC).
     - Tìm ra siêu tham số tối ưu: `{'max_depth': 6, 'min_samples_leaf': 100, 'criterion': 'gini', 'n_bins': 32, 'min_samples_split': 200}` với CV PR-AUC = 0.2021.
     - Quét và KHÓA cứng các ngưỡng trên tập Validation: `t1 = 0.1883` (phân vị 85%), `t2 = 0.0651` (phân vị 45%), ngưỡng tối ưu F1 `t_f1 = 0.1799`.
     - Lưu toàn bộ các ngưỡng vào `weights/thresholds.json`.
  3. Mở tập Test DUY NHẤT 1 LẦN sau khi các ngưỡng đã khóa hoàn toàn:
     - Đạt Test ROC-AUC = 0.6553, PR-AUC = 0.1548 (khớp số liệu tham chiếu ~0.667 / 0.154 ở Mục 4.9).
     - Brier score = 0.0711.
     - Tại ngưỡng F1: Precision = 0.1859, Recall = 0.3887, F1 = 0.2515 (TN=14,976, FP=2,387, FN=857, TP=545).
     - Phân tầng P1 (15.6% đơn, trễ 18.59%, bắt 38.87% đơn trễ); P2 (50.41% đơn, trễ 5.91%); P3 (33.97% đơn, trễ 4.68%).
     - Recall top 15% rủi ro nhất = 38.16%.
     - Tổng số đơn cảnh báo Trưởng ca = 41.06% (bắt được 57.49% đơn trễ).
     - Xuất báo cáo chi tiết vào `outputs/04_final_metrics.json` và `outputs/04_tuning_report.md`.
  4. Viết và thực thi các bài kiểm thử:
     - `tests/test_split_cv.py`: Đảm bảo expanding-window không rò rỉ và stratified k-fold giữ nguyên tỷ lệ nhãn.
     - `tests/test_weights.py`: Kiểm tra khứ hồi lưu/nạp đạt sai số 0.0.
  5. Chạy toàn bộ test suite dự án qua `pytest`: 23/23 bài test vượt qua (100% PASS).
- **Trạng thái:** Hoàn thành nghiệm thu Giai đoạn 4.

### Giai đoạn 5: Thống kê, Lỗi, Giải thích và Mô phỏng (Các bước 18–21)
- **Thời gian:** 2026-11-01
- **Người giám sát / Phê duyệt:** Nhóm sinh viên CK10
- **Nội dung thực hiện bởi AI:**
  1. Cài đặt module mô phỏng hàng đợi kho [`src/simulate.py`](src/simulate.py) theo đúng Phụ lục C và Mục 7:
     - 18,765 đơn hàng thật trong test, thời gian xử lý $6 + 3 \times (\text{n\_items} - 1)$ phút.
     - Ca làm việc 08:00 - 17:00, Thứ 2 đến Thứ 7.
     - So sánh 4 chính sách: FIFO (As-Is), EDD (baseline theo hạn), AI-EDD (To-Be) và AI Level-first (làm hết P1 rồi P2 rồi P3, bỏ qua hạn).
     - Quét toàn bộ lưới độ nhạy $K \in \{3,4,5,6\} \times \text{time\_factor} \in \{0.8, 0.9, 1.0\}$, lưu vào `outputs/05_simulation_grid.json`.
     - Kết quả tại $K=3$, 100% thời gian: FIFO trễ 7.2%, EDD trễ 0.8%, AI-EDD trễ 0.9%, AI Level-first trễ 11.6%.
  2. Viết module thống kê, phân tích lỗi và giải thích [`src/statistical_analysis.py`](src/statistical_analysis.py):
     - **Bước 18:** Bootstrap 1,000 lần tính khoảng tin cậy 95% của ROC-AUC ([0.6395, 0.6714]) và PR-AUC ([0.1420, 0.1699]); Paired bootstrap hiệu Tree - Seller và Tree - FIFO; đo lường ROC-AUC theo từng tháng test (05, 06, 07/2018).
     - **Bước 19:** Phân tích 857 đơn trễ bị bỏ sót (FN) theo bang và lịch sử người bán; khảo sát đơn hàng sát ngưỡng $t_1 \pm 0.03$.
     - **Bước 20:** Xuất cây dạng text (`outputs/05_tree_structure.txt`), trích xuất quyết định chi tiết 5 đơn mẫu, tính Gini importance và Permutation importance trên validation.
     - **Bước 21:** Xây dựng bảng lặp 5 Iterations (`outputs/05_iteration_table.json`) và báo cáo tổng hợp `outputs/05_phase5_report.md`.
  3. Viết bộ kiểm thử [`tests/test_simulate.py`](tests/test_simulate.py): Kiểm tra tính bảo toàn (`done >= t_arr`), ca làm việc hợp lệ và tính tuần tự khi $K=1$.
  4. Chạy toàn bộ test suite dự án qua `pytest`: 26/26 bài test vượt qua (100% PASS).
- **Trạng thái:** Hoàn thành nghiệm thu Giai đoạn 5 (Chưa làm web, tuân thủ đúng yêu cầu).

### Bổ sung Dữ liệu Giả định Kho & Minh bạch Giả định (Mục 4.7 & ASSUMPTIONS.md)
- **Thời gian:** 2026-11-01
- **Người giám sát / Phê duyệt:** Nhóm sinh viên CK10
- **Nội dung thực hiện bởi AI:**
  1. Soạn thảo tài liệu minh bạch hóa [`ASSUMPTIONS.md`](ASSUMPTIONS.md):
     - Ghi nhận chi tiết các hạn chế của Olist (thiếu hạn dùng, lô hàng, tồn kho, chi phí).
     - Trình bày bảng giả định có kiểm soát: phân nhóm `sim_shelf_life_days` (180–365 ngày cho thực phẩm, mỹ phẩm), tồn an toàn $SS = z \cdot \sigma \cdot \sqrt{L}$, tồn kho hiện tại $SOH \sim \text{Poisson}(\mu \times 7) + SS$, ca làm việc 08:00–17:00, $K=3$, chi phí phạt trễ 10% giá trị đơn, chi phí lưu kho 0.05 R$/ngày.
     - Cam kết ranh giới: 100% cột giả lập mang tiền tố `sim_`, lưu tại `data/simulated/`, tuyệt đối không đưa vào mô hình học máy.
  2. Viết script tạo dữ liệu [`src/synth.py`](src/synth.py) với `SEED = 42`:
     - Xuất `data/simulated/sku_inventory_simulated.csv` (32,951 SKU) gồm thống kê nhu cầu, tồn an toàn, tồn hiện tại và cờ nguy cơ đứt gãy.
     - Xuất `data/simulated/orders_simulated_features.csv` (98,666 đơn) gồm ngày nhận hàng, ngày hết hạn, cờ quy tắc FEFO (`sim_fefo_priority_flag` nếu hạn còn $\le 3$ ngày) và chi phí phạt trễ giả định.
  3. Viết bộ kiểm thử [`tests/test_synth.py`](tests/test_synth.py):
     - Kiểm tra tính tất định của seed 42.
     - Kiểm tra mọi cột giả lập bắt buộc có tiền tố `sim_`.
     - Kiểm tra danh sách `FEATURES` trong `src/data_prep.py` không chứa bất kỳ cột `sim_` nào.
     - Kiểm tra 5 file CSV gốc trong `data/raw/` nguyên vẹn 100%.
  4. Cập nhật `.gitignore` để quản lý `data/simulated/*.csv`.
  5. Chạy toàn bộ test suite dự án qua `pytest`: 29/29 bài test vượt qua (100% PASS).
### Giai đoạn 6: Xuất Dữ Liệu & Xây Dựng Giao Diện Web (Mục 8 & Phụ lục D, E)
- **Thời gian:** 2026-11-01
- **Người giám sát / Phê duyệt:** Nhóm sinh viên CK10 (Người dùng phê duyệt tiếp tục Giai đoạn 6)
- **Nội dung thực hiện bởi AI:**
  1. Xây dựng module xuất dữ liệu web [`src/export_web.py`](src/export_web.py):
     - Xuất toàn bộ 9 file JSON vào `web/data/`: `summary.json`, `audit.json`, `eda_monthly.json`, `metrics.json`, `sample_orders.json` (50 đơn mẫu test, ẩn danh ID), `simulation_grid.json`, `threshold_sweep.json`, `tree.json` (quy ước Phụ lục D), `triage.json`.
     - Tuyệt đối không hard-code số liệu vào HTML.
  2. Xây dựng script suy luận CLI độc lập [`predict_new.py`](predict_new.py) hỗ trợ đọc CSV đơn mới hoặc chạy 10 mẫu mặc định, đưa ra điểm rủi ro, phân loại P1/P2/P3 và cờ Trưởng ca.
  3. Xây dựng công cụ suy luận cây quyết định trên trình duyệt bằng JavaScript thuần [`web/js/tree.js`](web/js/tree.js) theo Phụ lục E:
     - Tuân thủ quy ước rẽ nhánh nghiêm ngặt: `x[f] < thr -> left` (dấu `<`, không dùng `<=`).
     - Ghi nhận đầy đủ chuỗi vết đường đi của quyết định (decision path trace) phục vụ giải thích mô hình.
  4. Tải và đóng gói thư viện Chart.js v4.4.1 UMD offline vào [`web/vendor/chart.umd.js`](web/vendor/chart.umd.js), không phụ thuộc CDN bên ngoài.
  5. Xây dựng mã định kiểu CSS hiện đại, chuẩn tiếp cận và responsive [`web/css/styles.css`](web/css/styles.css).
  6. Xây dựng bộ điều khiển giao diện SPA [`web/js/app.js`](web/js/app.js) tích hợp Chart.js, bộ lọc mô phỏng hàng đợi tương tác, thanh trượt chi phí $C_{FN}/C_{FP}$, và kiểm tra đơn mẫu trực tiếp.
  7. Thiết kế giao diện Web hoàn chỉnh 8 tab [`web/index.html`](web/index.html) với sơ đồ Swimlane SVG trực quan quy trình As-Is vs To-Be, bảng biểu và tuyên bố minh bạch đạo đức.
  8. Viết bộ kiểm thử đối chiếu [`tests/test_web_parity.py`](tests/test_web_parity.py) kiểm tra sự trùng khớp 100% giữa dự đoán cây trên Python và cấu trúc JSON (200 mẫu ngẫu nhiên, sai số tuyệt đối = 0.0).
  9. Chạy toàn bộ test suite dự án qua `pytest`: 30 passed, 1 skipped (Node.js fallback).
- **Trạng thái:** Hoàn thành nghiệm thu Giai đoạn 6.

---

### Giai đoạn 7: Tài Liệu Hóa, Tái Lập Pipeline & Rà Soát Cuối Cùng
- **Thời gian:** 2026-11-02
- **Người giám sát / Phê duyệt:** Nhóm sinh viên CK10
- **Nội dung thực hiện bởi AI:**
  1. Viết script thực thi tự động toàn bộ 21 bước tuần tự [`run_pipeline.py`](run_pipeline.py):
     - Tự động gọi lần lượt Data prep & Audit (1-6) $\rightarrow$ Leakage Audit (7-10) $\rightarrow$ Synth inventory & FEFO $\rightarrow$ Evaluate 7 baselines (11-13) $\rightarrow$ Hyperparameter GridSearch & Khóa ngưỡng (14-17) $\rightarrow$ Bootstrap CI, Error analysis, Explainability (18-20) $\rightarrow$ Queue Simulation As-Is vs To-Be (21) $\rightarrow$ Export Web Assets $\rightarrow$ Chạy Pytest toàn diện.
     - In bảng tổng kết nghiệm thu so khớp tự động với các chỉ số mục tiêu trong Mục 4.9.
  2. Viết script tạo báo cáo Jupyter Notebook tự động [`build_notebook.py`](build_notebook.py) bằng thư viện `nbformat`.
  3. Hoàn thiện tài liệu [`README.md`](README.md) chi tiết:
     - Tóm tắt đề tài, cấu trúc thư mục, quy tắc bắt buộc R1–R7, bảng kết quả 7 mô hình, hướng dẫn tái lập 1 lệnh, checklist nghiệm thu Mục 9.2, giới hạn nghiên cứu Mục 11 và danh mục sửa đề cương Word Mục 12.
  4. Cập nhật và đóng gói toàn bộ nhật ký [`AI_USAGE_LOG.md`](AI_USAGE_LOG.md).
  5. Thực thi chạy lại toàn bộ pipeline `python run_pipeline.py` từ đầu để xác nhận tính tái lập tuyệt đối (Rule R7).
- **Trạng thái:** Hoàn thành 100% toàn bộ 7 giai đoạn của đồ án.

---

### Giai đoạn Hoàn Thiện Notebook: Chuẩn Hóa `notebook.ipynb` 21 Bước Theo Khuôn 4 Repo Tham Khảo
- **Thời gian:** 2026-11-02
- **Người giám sát / Phê duyệt:** Nhóm sinh viên CK10 (Yêu cầu viết notebook chuyên sâu theo prompt CK10 Olist Decision Tree)
- **Nội dung thực hiện bởi AI:**
  1. Tạo module alias [`src/np_tree.py`](src/np_tree.py) xuất các lớp `DecisionTree`, `BinMapper`, `Node` từ `src/model.py` để notebook import tự nhiên theo đúng yêu cầu đề bài.
  2. Xây dựng trình sinh tự động [`generate_complete_notebook.py`](generate_complete_notebook.py) tạo ra [`notebook.ipynb`](notebook.ipynb) đạt chuẩn 100% theo quy cách 4 repo tham khảo (LogR, KNN, HGB, K-Means):
     - **Tiêu đề lớn duy nhất** ở đầu notebook (`# DỰ ĐOÁN NGUY CƠ TRỄ HẠN BÀN GIAO ĐƠN HÀNG TRONG KHO E-LOGISTICS BẰNG THUẬT TOÁN DECISION TREE THUẦN NUMPY`).
     - **Đúng 21 mục tuần tự**, mỗi mục có tiêu đề dạng `## BƯỚC n: TÊN TIẾNG VIỆT (ENGLISH TERM)`.
     - **Cấu trúc 3 thành phần chuẩn mực cho mỗi bước:** (1) Markdown mở đầu (mục tiêu, lý do, công thức LaTeX $...$, 3–6 gạch đầu dòng), (2) Code cell thực thi (in số liệu có viền bảng `=` / `-`, vẽ hình đa panel chuẩn màu sắc và chú thích), (3) Markdown nhận xét thực nghiệm (phân tích số liệu cụ thể, không nói chung chung).
     - **Lưu sẵn 100% outputs và hình ảnh** nhúng trực tiếp dạng base64 PNG trong file `.ipynb`, mở file ra là xem được toàn bộ kết quả mà không cần chạy lại kernel.
     - **28 hình ảnh chất lượng cao** ($\ge 25$ hình theo kỳ vọng) được nhúng và đồng thời lưu vào thư mục `reports/figures/` (`buoc_01_*.png` đến `buoc_21_*.png`) sẵn sàng cho việc chèn vào báo cáo Word.
     - Mọi hình ảnh từ dữ liệu mô phỏng hàng đợi kho đều được gắn nhãn minh bạch `(DỮ LIỆU MÔ PHỎNG)`.
     - **Bảng tổng kết 21 bước** dạng DataFrame ở cuối notebook hiển thị đầy đủ Bước, Tên tiếng Anh, Việc đã làm, và Kết quả nghiệm thu chính.
     - **Mục Khả năng tái lập** ghi rõ môi trường, hạt giống `SEED = 42`, phiên bản thư viện, và lệnh tái lập 1 dòng.
     - **Cell demo suy luận cuối cùng:** Nạp trọng số từ `weights/tree.json` và `weights/thresholds.json`, kiểm tra tính nhất quán (sai khác $= 0.0$), và thực hiện suy luận dự đoán mức độ ưu tiên P1/P2/P3 cho 3 đơn hàng mới in dạng Banner trực quan.
  3. Kiểm định chất lượng:
     - `nbformat.validate`: PASS 100%.
     - Số lượng cells: 69 cells (24 code cells, 45 markdown cells). 0 cell rỗng kết quả.
     - Toàn bộ test suite `pytest tests/`: 30 passed, 1 skipped.
- **Trạng thái:** Hoàn tất nghiệm thu toàn diện.

---

### Cập Nhật Đồng Bộ Theo `tài liệu hướng dẫn đồ án` Đã Điều Chỉnh
- **Thời gian:** 2026-11-02
- **Người giám sát / Phê duyệt:** Nhóm sinh viên CK10
- **Nội dung thực hiện bởi AI:**
  1. Đồng bộ hóa nội dung từ `tài liệu hướng dẫn đồ án` sang `tài liệu hướng dẫn đồ án` tại gốc dự án.
  2. Rà soát chi tiết các điểm thay đổi:
     - Làm rõ yêu cầu loại bỏ Random Forest khỏi bảng mô hình chính (như đã chốt trong Mục 4.5 và sửa đề cương Word Mục 2.3).
     - Chuẩn hóa văn phong tự chủ của đồ án, lược bỏ các tham chiếu phụ thuộc bên ngoài.
  3. Cập nhật mã nguồn và tài liệu tương ứng:
     - [`src/evaluate.py`](src/evaluate.py): Gỡ bỏ việc huấn luyện và đánh giá Random Forest, đưa bảng so sánh No Free Lunch về đúng **7 mô hình chuẩn** (Dummy, FIFO, Slack, Seller History Rule, Tree Order Only, Logistic Regression, Decision Tree Core).
     - [`src/statistical_analysis.py`](src/statistical_analysis.py): Cập nhật vòng lặp `Iter 5` thành `Decision Tree + Triage Post-hoc` (khóa ngưỡng phân tầng $t_1/t_2$ và cờ Trưởng ca can thiệp).
     - [`src/export_web.py`](src/export_web.py): Xuất lại toàn bộ dữ liệu JSON cho web (`web/data/metrics.json` phản ánh chuẩn xác 7 mô hình và bảng lặp).
     - [`README.md`](README.md): Cập nhật bảng so sánh 7 mô hình và ghi rõ lý do không triển khai Random Forest (tập trung vào tính giải thích được XAI của Decision Tree).
  4. Thực thi kiểm định chất lượng:
     - Toàn bộ test suite `pytest tests/`: 30 passed, 1 skipped.
- **Trạng thái:** Đồng bộ hoàn tất 100%.

---

### Tái Thiết Kế & Hoàn Thiện `notebook.ipynb` Theo Mạch Kể Chuyện Báo Cáo Thực Nghiệm
- **Thời gian:** 2026-11-02
- **Người giám sát / Phê duyệt:** Nhóm sinh viên CK10 (Prompt chỉ định notebook phong cách báo cáo thực nghiệm đọc từ trên xuống)
- **Nội dung thực hiện bởi AI:**
  1. Xây dựng script biên tập chuyên biệt [`build_custom_notebook.py`](build_custom_notebook.py) tạo ra [`notebook.ipynb`](notebook.ipynb) chuyển từ 21 bước chia cứng sang **15 Mục báo cáo thực nghiệm logic, dẫn dắt tự nhiên từ trên xuống dưới**:
     - Tiêu đề đầu notebook + Tóm tắt điều hành (Executive Summary).
     - Mục I: Định nghĩa bài toán & Bản chất học máy trong E-Logistics (Bước 1, 2).
     - Mục II: Khám phá, Kiểm toán dữ liệu Olist & Lọc phễu Cohort (Bước 3, 4).
     - Mục III: Xử lý tiền kỳ: Co giãn thang đo & Biến phân loại (Bước 5, 6).
     - Mục IV: Kỹ thuật đặc trưng (20 features) & Chống rò rỉ tri thức Knowledge-Time Safe (Bước 8, 9).
     - Mục V: Chiến lược chia dữ liệu: Time-Split vs Random-Split (Bước 9, 10).
     - Mục VI: Thuật toán Cây quyết định NumPy thuần & Cơ chế phân nhánh (Bước 7).
     - Mục VII: Đánh giá mô hình cơ sở & Định lý No Free Lunch (Bước 11, 12).
     - Mục VIII: Đánh đổi Bias – Variance & Tinh chỉnh siêu tham số trên Validation (Bước 13, 14).
     - Mục IX: Kiểm định chéo chuỗi thời gian Expanding-Window Time CV (Bước 16).
     - Mục X: Thước đo hiệu năng & Khóa ngưỡng phân tầng Triage trên Validation (Bước 15).
     - Mục XI: Huấn luyện cuối cùng, Lưu trữ trọng số & Mở tập Test độc lập (Bước 17, 18).
     - Mục XII: Phân tích lỗi, Hiệu chuẩn xác suất & Điểm mù seller mới (Bước 19).
     - Mục XIII: Giải thích mô hình (Explainable AI - XAI) & 5 luật If-Then (Bước 20).
     - Mục XIV: Mô phỏng vận hành hàng đợi kho As-Is vs To-Be & Đánh giá rủi ro (Bước 21).
     - Mục XV: Bảng đối chiếu 21 bước pipeline, Khả năng tái lập & Demo suy luận từ weights.
  2. Tối ưu hóa hệ thống hình ảnh:
     - Tạo đúng **25 hình ảnh chọn lọc** (mỗi hình trả lời một câu hỏi khoa học cụ thể, không vẽ hình thừa thãi), lưu tại `reports/figures/`.
     - Toàn bộ hình ảnh từ dữ liệu mô phỏng hàng đợi kho được gắn nhãn minh bạch `(DỮ LIỆU MÔ PHỎNG)`.
     - Chú thích tiếng Việt có dấu, palette cố định chuẩn mực, hiển thị số trên đầu cột.
  3. Báo cáo trung thực các phát hiện khoa học:
     - Tín hiệu trễ xuất phát ~65% từ lịch sử người bán (`seller_han_rate`), các biến kích thước đơn hàng gần như vô dụng ($r < 0.03$, AUC = 0.518).
     - Lợi ích giảm trễ trong hàng đợi chủ yếu do luật hạn giao cam kết (EDD); AI phục vụ phân tầng cảnh báo sớm (Triage). Lạm dụng AI xếp theo level-first làm trễ tăng vọt lên 11.6%.
  4. Kiểm định chất lượng:
     - `nbformat.validate`: PASS 100%.
     - Số lượng cells: 48 cells (17 code cells $\le 60$ dòng, 31 markdown cells). 0 cell rỗng output.
     - 25 hình ảnh nhúng base64 PNG sẵn sàng xem offline ngay lập tức.
     - Kiểm thử toàn hệ thống `pytest tests/`: 30 passed, 1 skipped.
- **Trạng thái:** Hoàn thành xuất sắc toàn diện.




