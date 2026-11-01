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
  1. Cài đặt module mô phỏng hàng đợi kho [`src/simulate.py`](file:///c:/Users/DELL/Downloads/ck%20e%20logistic/src/simulate.py) theo đúng Phụ lục C và Mục 7:
     - 18,765 đơn hàng thật trong test, thời gian xử lý $6 + 3 \times (\text{n\_items} - 1)$ phút.
     - Ca làm việc 08:00 - 17:00, Thứ 2 đến Thứ 7.
     - So sánh 4 chính sách: FIFO (As-Is), EDD (baseline theo hạn), AI-EDD (To-Be) và AI Level-first (làm hết P1 rồi P2 rồi P3, bỏ qua hạn).
     - Quét toàn bộ lưới độ nhạy $K \in \{3,4,5,6\} \times \text{time\_factor} \in \{0.8, 0.9, 1.0\}$, lưu vào `outputs/05_simulation_grid.json`.
     - Kết quả tại $K=3$, 100% thời gian: FIFO trễ 7.2%, EDD trễ 0.8%, AI-EDD trễ 0.9%, AI Level-first trễ 11.6%.
  2. Viết module thống kê, phân tích lỗi và giải thích [`src/statistical_analysis.py`](file:///c:/Users/DELL/Downloads/ck%20e%20logistic/src/statistical_analysis.py):
     - **Bước 18:** Bootstrap 1,000 lần tính khoảng tin cậy 95% của ROC-AUC ([0.6395, 0.6714]) và PR-AUC ([0.1420, 0.1699]); Paired bootstrap hiệu Tree - Seller và Tree - FIFO; đo lường ROC-AUC theo từng tháng test (05, 06, 07/2018).
     - **Bước 19:** Phân tích 857 đơn trễ bị bỏ sót (FN) theo bang và lịch sử người bán; khảo sát đơn hàng sát ngưỡng $t_1 \pm 0.03$.
     - **Bước 20:** Xuất cây dạng text (`outputs/05_tree_structure.txt`), trích xuất quyết định chi tiết 5 đơn mẫu, tính Gini importance và Permutation importance trên validation.
     - **Bước 21:** Xây dựng bảng lặp 5 Iterations (`outputs/05_iteration_table.json`) và báo cáo tổng hợp `outputs/05_phase5_report.md`.
  3. Viết bộ kiểm thử [`tests/test_simulate.py`](file:///c:/Users/DELL/Downloads/ck%20e%20logistic/tests/test_simulate.py): Kiểm tra tính bảo toàn (`done >= t_arr`), ca làm việc hợp lệ và tính tuần tự khi $K=1$.
  4. Chạy toàn bộ test suite dự án qua `pytest`: 26/26 bài test vượt qua (100% PASS).
- **Trạng thái:** Hoàn thành nghiệm thu Giai đoạn 5 (Chưa làm web, tuân thủ đúng yêu cầu).

### Bổ sung Dữ liệu Giả định Kho & Minh bạch Giả định (Mục 4.7 & ASSUMPTIONS.md)
- **Thời gian:** 2026-11-01
- **Người giám sát / Phê duyệt:** Nhóm sinh viên CK10
- **Nội dung thực hiện bởi AI:**
  1. Soạn thảo tài liệu minh bạch hóa [`ASSUMPTIONS.md`](file:///c:/Users/DELL/Downloads/ck%20e%20logistic/ASSUMPTIONS.md):
     - Ghi nhận chi tiết các hạn chế của Olist (thiếu hạn dùng, lô hàng, tồn kho, chi phí).
     - Trình bày bảng giả định có kiểm soát: phân nhóm `sim_shelf_life_days` (180–365 ngày cho thực phẩm, mỹ phẩm), tồn an toàn $SS = z \cdot \sigma \cdot \sqrt{L}$, tồn kho hiện tại $SOH \sim \text{Poisson}(\mu \times 7) + SS$, ca làm việc 08:00–17:00, $K=3$, chi phí phạt trễ 10% giá trị đơn, chi phí lưu kho 0.05 R$/ngày.
     - Cam kết ranh giới: 100% cột giả lập mang tiền tố `sim_`, lưu tại `data/simulated/`, tuyệt đối không đưa vào mô hình học máy.
  2. Viết script tạo dữ liệu [`src/synth.py`](file:///c:/Users/DELL/Downloads/ck%20e%20logistic/src/synth.py) với `SEED = 42`:
     - Xuất `data/simulated/sku_inventory_simulated.csv` (32,951 SKU) gồm thống kê nhu cầu, tồn an toàn, tồn hiện tại và cờ nguy cơ đứt gãy.
     - Xuất `data/simulated/orders_simulated_features.csv` (98,666 đơn) gồm ngày nhận hàng, ngày hết hạn, cờ quy tắc FEFO (`sim_fefo_priority_flag` nếu hạn còn $\le 3$ ngày) và chi phí phạt trễ giả định.
  3. Viết bộ kiểm thử [`tests/test_synth.py`](file:///c:/Users/DELL/Downloads/ck%20e%20logistic/tests/test_synth.py):
     - Kiểm tra tính tất định của seed 42.
     - Kiểm tra mọi cột giả lập bắt buộc có tiền tố `sim_`.
     - Kiểm tra danh sách `FEATURES` trong `src/data_prep.py` không chứa bất kỳ cột `sim_` nào.
     - Kiểm tra 5 file CSV gốc trong `data/raw/` nguyên vẹn 100%.
  4. Cập nhật `.gitignore` để quản lý `data/simulated/*.csv`.
  5. Chạy toàn bộ test suite dự án qua `pytest`: 29/29 bài test vượt qua (100% PASS).
- **Trạng thái:** Hoàn tất bổ sung dữ liệu giả lập chuẩn tắc. Vẫn DỪNG lại, chưa làm web theo yêu cầu.







