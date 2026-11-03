# Thiết Kế Chuỗi Dịch Vụ Logistics Tự Động Có AI Ra Quyết Định
## Nghiệp Vụ Kho (Inventory) — Dự Đoán Nguy Cơ Trễ Bàn Giao Vận Tải

**Môn học:** E-Logistics & Supply Chain Analytics (CK10)  
**Tập dữ liệu:** Brazilian E-Commerce by Olist (Kaggle `olistbr/brazilian-ecommerce`, Version 2)  
**Thuật toán lõi:** Decision Tree phân tầng tự hiện thực bằng **NumPy thuần** (Không phụ thuộc `scikit-learn` trong mã nguồn sản phẩm)  
**Sản phẩm hoàn chỉnh:** Mã nguồn Python (`src/`) + Báo cáo Jupyter Notebook (`notebook.ipynb`) + Website phân tích & mô phỏng tương tác 8 trang (`web/`)

---

## 1. Tóm Tắt Đề Tài & Bài Toán Cốt Lõi

Trong chuỗi cung ứng thương mại điện tử nhiều người bán (marketplace logistics), khâu bàn giao hàng hóa từ kho người bán cho đơn vị vận chuyển (Carrier Handover) là điểm nghẽn nghiêm trọng nhất dẫn đến vỡ cam kết thời gian giao hàng (SLA) đối với khách hàng cuối.

### Phân biệt 2 nhãn dữ liệu:
1. **Trễ bàn giao vận tải (`late_handover` - Nhãn huấn luyện mô hình):**
   $$\text{late\_handover} = \mathbf{1}(\text{order\_delivered\_carrier\_date} > \text{shipping\_limit\_date})$$
   Đây là sự kiện xảy ra ngay tại khâu Kho/Người bán, thuộc phạm vi nghiệp vụ **Kho (Inventory)**.
2. **Trễ giao khách cuối (`late_delivery` - Hậu quả hạ nguồn):**
   $$\text{late\_delivery} = \mathbf{1}(\text{order\_delivered\_customer\_date} > \text{order\_estimated\_delivery\_date})$$
   Phụ thuộc vào năng lực vận chuyển chặng giữa và chặng cuối của đơn vị chuyển phát (Carrier).

### Mục tiêu hệ thống:
- **Dự đoán rủi ro:** Ước lượng xác suất trễ bàn giao ngay tại thời điểm đơn hàng được duyệt (`order_approved_at`).
- **Phân tầng quyết định (Triage):** Khóa 2 ngưỡng cắt trên tập Thẩm định ($t_1 = 0.1883, t_2 = 0.0651$) để chia đơn hàng thành 3 nhóm ưu tiên:
  - **P1 (Cực kỳ khẩn cấp - Top 15%):** Đơn rủi ro cao, tự động chuyển vào hàng đợi ưu tiên và cắm cờ can thiệp cho Trưởng ca.
  - **P2 (Cảnh báo - Top 55%):** Đơn có dấu hiệu nguy cơ, cần giám sát tiến độ.
  - **P3 (Tiêu chuẩn - Còn lại):** Xử lý theo quy trình thông thường.
- **Hỗ trợ con người:** Kết hợp quy tắc hạn dùng (FEFO) và rủi ro người bán mới để tạo cờ cảnh báo Trưởng ca (Supervisor Flag), giải phóng con người khỏi 60% quyết định thủ công.

---

## 2. Cấu Trúc Thư Mục Dự Án

```
ck10-olist-priority/
├── data/
│   ├── raw/                      # 5 CSV gốc Olist (nguyên vẹn, không sửa, không commit)
│   │   └── README.md             # Nguồn dữ liệu & hướng dẫn kiểm tra giấy phép
│   └── simulated/                # Dữ liệu mô phỏng nghiệp vụ kho (SEED=42, tiền tố sim_)
│       ├── sku_inventory_simulated.csv
│       └── orders_simulated_features.csv
├── src/                          # Mã nguồn chính - 100% NumPy thuần (CẤM sklearn)
│   ├── __init__.py
│   ├── data_prep.py              # Đọc CSV, lọc phễu, tạo đặc trưng chống rò rỉ, chia tập
│   ├── model.py                  # DecisionTree, LogisticRegression, BinMapper, Metrics
│   ├── evaluate.py               # So sánh 7 mô hình No Free Lunch, quét Bias-Variance
│   ├── train.py                  # GridSearch, Cross-Validation, khóa ngưỡng, lưu weights
│   ├── statistical_analysis.py   # Bootstrap CI 1,000 lần, phân tích lỗi FN, giải thích cây
│   ├── simulate.py               # Mô phỏng hàng đợi kho As-Is vs To-Be (18,765 đơn test)
│   ├── synth.py                  # Sinh dữ liệu mô phỏng nghiệp vụ kho độc lập (Mục 4.7)
│   ├── weights.py                # Xuất/nạp mô hình JSON/NPZ, kiểm tra khứ hồi (parity)
│   └── export_web.py             # Xuất 9 file dữ liệu JSON phục vụ giao diện Web
├── tests/                        # Kiểm thử tự động Pytest (chỉ dùng sklearn để đối chiếu)
│   ├── test_data_prep.py         # Kiểm tra phễu lọc 99,441 -> 95,180 -> 88,526 & SHA256
│   ├── test_leakage.py           # 7 bài test kiểm toán rò rỉ dữ liệu (Knowledge-time)
│   ├── test_metrics.py           # Kiểm tra độ chính xác metrics so với sklearn
│   ├── test_tree_vs_sklearn.py   # Kiểm tra cây NumPy khớp sklearn (chênh AUC < 0.01)
│   ├── test_split_cv.py          # Kiểm tra chia TimeSeriesSplit & StratifiedKFold
│   ├── test_weights.py           # Kiểm tra khứ hồi lưu/nạp trọng số
│   ├── test_simulate.py          # Kiểm tra bảo toàn hàng đợi và ca làm việc kho
│   ├── test_synth.py             # Kiểm tra tiền tố sim_ và tính độc lập của dữ liệu giả
│   └── test_web_parity.py        # Kiểm tra suy luận trình duyệt khớp 100% với Python
├── weights/                      # Trọng số và tham số đã huấn luyện
│   ├── tree.json                 # Cấu trúc cây quyết định (ngưỡng thật, xác suất p)
│   ├── thresholds.json           # Ngưỡng khóa t1, t2, t_f1 trên tập Validation
│   ├── best_params.json          # Siêu tham số tối ưu tìm được qua GridSearch
│   ├── feature_names.json        # 20 tên đặc trưng đầu vào
│   └── tree.npz                  # Định dạng nén nhị phân
├── outputs/                      # Báo cáo kết quả và hình ảnh sinh tự động
│   ├── 01_data_audit.json / .md  # Kết quả kiểm toán dữ liệu gốc
│   ├── 03_model_comparison.json  # So sánh 7 mô hình
│   ├── 04_final_metrics.json     # Đánh giá duy nhất trên tập Test
│   ├── 05_bootstrap_ci.json      # Khoảng tin cậy 95% bootstrap 1,000 lần
│   ├── 05_error_analysis.json    # Phân tích sâu 857 trường hợp False Negative
│   ├── 05_tree_structure.txt     # Bản text dump cấu trúc cây quyết định
│   ├── 05_simulation_grid.json   # Lưới mô phỏng K in {3,4,5,6} x factor in {0.8,0.9,1.0}
│   └── fig_*.png                 # Các đồ thị phân tích EDA và Bias-Variance
├── web/                          # Giao diện Web tĩnh (Triển khai GitHub Pages)
│   ├── index.html                # SPA gồm 8 tab hoàn chỉnh (Tiếng Việt, responsive)
│   ├── css/styles.css            # Giao diện hiện đại, chuẩn tiếp cận
│   ├── js/app.js                 # Điều khiển SPA, tải JSON, vẽ biểu đồ Chart.js
│   ├── js/tree.js                # Công cụ suy luận cây quyết định thuần JavaScript
│   ├── data/*.json               # 9 file dữ liệu JSON (không hard-code số)
│   └── vendor/chart.umd.js       # Thư viện Chart.js v4.4.1 ghim phiên bản offline
├── run_pipeline.py               # Script thực thi toàn bộ 21 bước tuần tự từ đầu
├── predict_new.py                # CLI suy luận đơn hàng mới từ file CSV
├── build_notebook.py             # Sinh notebook.ipynb bằng thư viện nbformat
├── notebook.ipynb                # Jupyter Notebook hoàn chỉnh 22 cells
├── ASSUMPTIONS.md                # Bảng giả định mô phỏng kho & tính toán tồn kho
├── AI_USAGE_LOG.md               # Nhật ký minh bạch hóa việc sử dụng AI (GĐ 0–7)
├── HUONG_DAN.md                  # Tài liệu hướng dẫn thiết kế chuẩn của đồ án
├── requirements.txt              # Danh sách thư viện và phiên bản cụ thể
└── pytest.ini                    # Cấu hình kiểm thử tự động
```

---

## 3. Các Nguyên Tắc Kỹ Thuật Bắt Buộc (Rules R1 – R7)

1. **R1 - Bất biến dữ liệu gốc:** Không bao giờ sửa, ghi đè hoặc commit 5 file CSV gốc trong `data/raw/`. Mã băm SHA-256 được kiểm toán tự động trước và sau mỗi lần chạy.
2. **R2 - NumPy thuần túy trong mã nguồn:** Toàn bộ thuật toán cây quyết định, rừng ngẫu nhiên, hồi quy logistic, chuẩn hóa và độ đo trong `src/` đều viết bằng NumPy. Thư viện `scikit-learn` tuyệt đối chỉ được dùng trong `tests/` để đối soát.
3. **R3 - Chống rò rỉ dữ liệu (Strict Knowledge-Time):** Chỉ sử dụng thông tin phát sinh trước `order_approved_at`. Tỷ lệ trễ lịch sử của người bán chỉ tính từ các đơn hàng đã giao hoàn tất trước thời điểm duyệt đơn.
4. **R4 - Phân chia theo chuỗi thời gian (Time-Split):** Chia cắt tại `2018-05-01`: Train Fit (55,808 đơn), Validation (13,953 đơn), Test (18,765 đơn). Không dùng chia ngẫu nhiên làm kết quả chính.
5. **R5 - Khóa ngưỡng trước khi mở Test:** Ngưỡng $t_1, t_2, t_{F1}$ và siêu tham số chỉ được chọn và khóa cứng trên Validation. Tập Test chỉ được đánh giá duy nhất một lần ở bước cuối cùng.
6. **R6 - Phân định rõ ràng dữ liệu giả định:** Các thuộc tính Olist thiếu (hạn dùng, tồn kho an toàn) được tạo bằng script riêng `src/synth.py` với `SEED=42` và tiền tố `sim_`. Tuyệt đối không đưa vào mô hình học máy.
7. **R7 - Tính tái lập tuyệt đối:** Chạy lại `python run_pipeline.py` từ đầu cho ra kết quả trùng khớp hoàn toàn.

---

## 4. Tóm Tắt Kết Quả Nghiệm Thu Thực Nghiệm

### 4.1. Bảng So Sánh 7 Mô Hình (No Free Lunch Principle)

| Mô hình | Val ROC-AUC | Test ROC-AUC | Test PR-AUC | Test Brier | Ghi chú vận hành |
|---|:---:|:---:|:---:|:---:|---|
| **Dummy Baseline** | 0.5000 | 0.5000 | 0.0754 | 0.0697 | Dự đoán xác suất tiên nghiệm |
| **FIFO Baseline** | 0.5057 | 0.4646 | 0.0693 | 0.2235 | Điểm tăng dần theo thời gian đến |
| **Slack Baseline** | 0.5401 | 0.4866 | 0.0751 | 0.2198 | Điểm nghịch đảo với thời gian đệm SLA |
| **Seller History Rule** | 0.6974 | 0.6712 | 0.1705 | 0.0698 | Luật đơn biến `seller_han_rate` |
| **Tree (Order Only)** | 0.5445 | 0.5183 | 0.0841 | 0.0735 | Không dùng đặc trưng người bán |
| **Decision Tree (Core)** | **0.6908** | **0.6553** | **0.1548** | **0.0711** | Mô hình chính (Cây nông depth=6) |
| **Logistic Regression** | 0.7061 | 0.6833 | 0.1764 | 0.0700 | Chuẩn hóa Z-score + GD + L2 |

> **Quyết định không làm Random Forest:** Đồ án **quyết định chỉ dùng Decision Tree** (tuân thủ hướng dẫn Mục 4.5 và sửa đề cương Word Mục 2.3). Lý do: Mô hình ensemble làm mất đi tính minh bạch diễn giải (XAI), không thể trích xuất cây luật If-Then trực quan dạng JSON để nhúng suy luận tức thì trên trình duyệt hoặc hệ thống điều phối kho; trong khi hiệu năng Decision Tree kết hợp luật nghiệp vụ đã giải quyết trọn vẹn bài toán phân tầng rủi ro (Triage).

> **Phát hiện khoa học:** Cây chỉ dùng đặc trưng đơn hàng (Tree Order Only) có AUC = 0.518 (tương đương đoán ngẫu nhiên). Nguồn tín hiệu dự báo đến 65% từ lịch sử người bán (`seller_han_rate`). Cây quyết định không hơn luật đơn biến về độ phân tách, nhưng mang lại giá trị vận hành vượt trội nhờ khả năng tích hợp đa điều kiện và phân tầng trực quan (Triage).

### 4.2. Hiệu Năng Phân Tầng (Triage) Trên Tập Test (18,765 đơn)

- **Nhóm P1 (Cực kỳ khẩn cấp):** Chiếm 15.6% tổng lượng đơn, tỷ lệ trễ thực tế lên tới 18.6% (gấp 2.5 lần tỷ lệ trễ trung bình của tập test là 7.5%), bắt được **38.9%** tổng số đơn trễ.
- **Nhóm P2 (Cảnh báo rủi ro):** Chiếm 50.4% lượng đơn, tỷ lệ trễ 5.9%.
- **Nhóm P3 (Tiêu chuẩn):** Chiếm 34.0% lượng đơn, tỷ lệ trễ thấp 4.7%.
- **Cờ Trưởng ca (Supervisor Flag):** Cảnh báo 41.1% tổng số đơn (gồm P1 hoặc người bán thiếu dữ liệu lịch sử hoặc hàng cận date FEFO), bắt được **57.5%** tổng số đơn trễ, giải phóng 58.9% đơn hàng để tự động hóa hoàn toàn.

### 4.3. Kết Quả Mô Phỏng Hàng Đợi Kho (As-Is vs To-Be)

Mô phỏng 18,765 đơn hàng trong 91 ngày tại trung tâm phân loại kho ($K=3$ nhân viên, ca làm việc 08:00–17:00 Thứ 2 đến Thứ 7, thời gian xử lý $T = 6 + 3 \times (n_{\text{items}} - 1)$ phút):

| Chính sách điều phối | Tỷ lệ trễ | Lead time TB | Lead time p95 | Đánh giá & Rủi ro |
|---|:---:|:---:|:---:|---|
| **FIFO (Hiện trạng As-Is)** | **7.2%** | 33.2 h | 108.0 h | Khớp với tỷ lệ trễ thực tế của Olist (7.5%) |
| **EDD (Luật hạn giao sớm nhất)** | **0.8%** | 33.2 h | 121.0 h | Quy tắc kinh điển, giảm 89% đơn trễ do nghẽn |
| **AI-EDD (Đề xuất To-Be)** | **0.9%** | 33.2 h | 121.0 h | Kết hợp hạn giao và giảm trừ 24h theo điểm rủi ro |
| **AI level-first (Ứng dụng sai)** | **11.6%** | 33.6 h | 189.0 h | **CẢNH BÁO:** Làm hết P1 rồi mới đến P2/P3 khiến trễ tăng vọt |

> **Báo cáo trung thực:** Mô phỏng chứng minh rằng sự cải thiện trễ từ 7.2% xuống 0.8% chủ yếu đến từ việc **sắp xếp theo thời hạn (EDD)**. Mô hình AI không giúp giảm trễ thêm trong hàng đợi vì AI dự báo trễ do người bán, trong khi hàng đợi giải quyết bài toán tắc nghẽn thông lượng. Lạm dụng AI bằng cách xếp đơn P1 làm trước bất chấp thời hạn cam kết (AI level-first) sẽ gây hại nghiêm trọng cho hệ thống logistics.

---

## 5. Hướng Dẫn Cài Đặt & Thực Thi

### 5.1. Cài Đặt Môi Trường
Yêu cầu Python $\ge 3.10$. Khuyến nghị sử dụng môi trường ảo:
```bash
python -m venv venv
# Trên Windows:
.\venv\Scripts\activate
# Trên Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 5.2. Chạy Toàn Bộ Pipeline Tự Động (1 Lệnh Duy Nhất)
Thực thi toàn bộ 21 bước từ dữ liệu thô, kiểm toán rò rỉ, huấn luyện, khóa ngưỡng, bootstrap, mô phỏng hàng đợi, xuất dữ liệu web và kiểm thử:
```bash
python run_pipeline.py
```

### 5.3. Chạy Bộ Kiểm Thử Độc Lập
Chạy bộ kiểm thử tự động gồm 31 test cases kiểm soát phễu, rò rỉ, tính tất định và tính khớp dữ liệu:
```bash
pytest
```

### 5.4. Dự Đoán Đơn Hàng Mới Bằng CLI
Dự đoán mức độ ưu tiên và cờ Trưởng ca cho một file CSV đơn hàng mới bằng mô hình đã lưu:
```bash
python predict_new.py path/to/new_orders.csv --output predictions.csv

# Hoặc chạy kiểm tra nhanh trên 10 đơn hàng mẫu:
python predict_new.py
```

### 5.5. Tái Tạo Báo Cáo Jupyter Notebook
Sinh lại tệp `notebook.ipynb` có đầy đủ 22 cells định dạng đẹp mắt bằng `nbformat`:
```bash
python build_notebook.py
```

### 5.6. Trải Nghiệm Giao Diện Web Dashboard Offline
Mở ứng dụng Web tĩnh (HTML/CSS/JS thuần, không cần kết nối internet):
```bash
cd web
python -m http.server 8000
```
Truy cập trình duyệt tại địa chỉ: `http://localhost:8000`

---

## 6. Danh Mục 8 Trang Trên Giao Diện Web (`web/index.html`)

1. **Tổng quan (Overview):** Bối cảnh bài toán, 3 KPI chính, tóm tắt các kết luận và cảnh báo khoa học quan trọng.
2. **Quy trình (Workflow):** Sơ đồ Swimlane trực quan thể hiện luồng đơn hàng từ Nhận đơn $\rightarrow$ Kho $\rightarrow$ Vận tải $\rightarrow$ Khách hàng; so sánh Hiện trạng (As-Is) vs Đề xuất (To-Be).
3. **Dữ liệu (Data):** Phễu lọc dữ liệu chuẩn tắc, kết quả kiểm toán 5 bảng CSV gốc, biểu đồ EDA biến động theo tháng và cú sốc SLA.
4. **Mô hình (Model):** Bảng so sánh 7 mô hình No Free Lunch, đồ thị Bias-Variance, bảng 5 vòng lặp phát triển thuật toán (Iterations 1-5).
5. **Thử nghiệm tương tác (Demo):** Chọn đơn hàng mẫu hoặc nhập tay các thông số để nhận điểm rủi ro, phân loại P1/P2/P3, cờ FEFO và hiển thị cây giải thích trực quan theo thời gian thực (100% JavaScript thuần).
6. **Mô phỏng hàng đợi (Simulation):** Bảng điều khiển tương tác thay đổi số nhân viên $K \in \{3,4,5,6\}$ và hệ số ca làm việc để so sánh 4 chính sách FIFO, EDD, AI-EDD, AI level-first.
7. **Rủi ro & Chi phí (Risk & Cost):** Ma trận nhầm lẫn (Confusion Matrix), thanh trượt chi phí $C_{FN}$ và $C_{FP}$ tính toán chi phí tối ưu theo ngưỡng, phân tích rủi ro khi AI đoán sai.
8. **Tài liệu & Đạo đức (Ethics & Docs):** Tuyên bố minh bạch việc sử dụng AI, bản quyền dữ liệu Olist, bảo vệ quyền riêng tư người bán và danh mục tài liệu tham khảo.

---

## 7. Các Giới Hạn Cần Lưu Ý (Mục 11 HUONG_DAN.md)

1. **Bản chất dữ liệu Olist:** Olist là sàn thương mại điện tử kết nối các người bán độc lập tại Brazil, không phải kho vận trung tâm của một doanh nghiệp duy nhất. Khái niệm "Kho (Inventory)" trong đề tài là sự ánh xạ logic nghiệp vụ vào thời gian xử lý và bàn giao của seller.
2. **Các thuộc tính thiếu hụt:** Olist không có hạn sử dụng, lô hàng, mã vị trí kệ, tồn kho an toàn hay giờ chốt xe thực tế. Toàn bộ các biến này là mô phỏng giả lập có kiểm soát (`sim_`) và không tham gia vào mô hình huấn luyện ML.
3. **Khả năng khái quát hóa:** Dữ liệu thuộc giai đoạn 2016–2018 tại một quốc gia (Brazil) và chỉ giữ các đơn hàng đã giao thành công bởi 1 người bán duy nhất (chấp nhận thiên lệch chọn mẫu do lọc đơn hủy).
4. **Năng lực phân tách của mô hình:** Mô hình đạt ROC-AUC ~0.655, phản ánh bài toán phân loại có độ nhiễu rất cao; điểm dự đoán đóng vai trò xếp hạng rủi ro tương đối chứ không phải xác suất tuyệt đối.
5. **Rủi ro vận hành khi dùng AI:** Việc tin tưởng tuyệt đối vào AI để bỏ qua quy tắc thời hạn cam kết (EDD) sẽ làm tỷ lệ trễ tăng gấp đôi (từ 7.2% lên 11.6%). Hệ thống logistics phải luôn duy trì con người trong vòng lặp quyết định (Human-in-the-loop).

---

## 8. Danh Mục Các Điểm Nhóm Cần Sửa Trong Đề Cương Word (Mục 12)

*Nhóm sinh viên tự đối chiếu và chỉnh sửa trong file Word nộp cho giảng viên, không để AI tự ý chỉnh sửa:*

| Vị trí trong Đề cương Word | Nội dung hiện tại trong Word | Nội dung nhóm cần sửa thành |
|---|---|---|
| **Phạm vi nghiên cứu (Mục 3)** | Dữ liệu mô phỏng / giả định toàn bộ | Dữ liệu thực tế Olist 2016–2018 (có lớp luật giả định bổ sung độc lập cho FEFO) |
| **Mục 2.2.1** | Phân loại đa lớp (Multiclass classification) | Phân loại nhị phân nguy cơ trễ bàn giao + Phân tầng 3 mức ưu tiên hậu xử lý (Triage) |
| **Mục 2.2.2** | Các biến $x_2$ FEFO, $x_3$ tồn an toàn, $x_5$ giờ chốt xe trong mô hình | Sử dụng 20 biến thực tế; FEFO và tồn an toàn là lớp luật kinh doanh giả định bổ sung |
| **Mục 2.3 (Bước 1)** | Làm sạch dữ liệu giả lập | Ghi nhận lỗi logic và cắt cửa sổ chuỗi thời gian trên dữ liệu thật |
| **Mục 2.4.1** | Sử dụng thư viện `scikit-learn` | Tự hiện thực bằng NumPy thuần túy (thư viện `scikit-learn` chỉ dùng kiểm toán đối soát) |
| **Mục 2.4.2** | 500–1000 đơn trong một ca cao điểm | 88,526 đơn thật trong cohort; mô phỏng hàng đợi 18,765 đơn trong 91 ngày tập test |
| **Mục 3.2** | Tính lead time, đúng hạn, giảm can thiệp thủ công | Báo cáo rõ: AI-EDD không hơn EDD; AI đóng vai trò công cụ lọc cảnh báo sớm cho Trưởng ca |

---

## 9. Bảng Kiểm Tra Nghiệm Thu (Acceptance Checklist)

- [x] 5 tệp CSV gốc nguyên vẹn tuyệt đối (Mã băm SHA-256 không thay đổi trước và sau chạy).
- [x] Thư mục `src/` không import `scikit-learn` (100% NumPy thuần túy).
- [x] Tập kiểm thử (Test) chỉ được sử dụng duy nhất một lần sau khi đã khóa cứng mô hình và ngưỡng.
- [x] Kết quả kiểm thử nằm hoàn toàn trong dải kỳ vọng khoa học (ROC-AUC: 0.6553, PR-AUC: 0.1548).
- [x] Đầy đủ bảng so sánh 7 mô hình cơ sở theo nguyên lý No Free Lunch.
- [x] Toàn bộ giả định mô phỏng được công bố minh bạch tại `ASSUMPTIONS.md` và lưới mô phỏng nhạy được xuất đầy đủ.
- [x] Báo cáo trung thực: Nêu rõ tín hiệu dự báo đến từ người bán và AI-EDD không vượt trội hơn EDD trong hàng đợi.
- [x] Dữ liệu giả lập FEFO tách biệt với tiền tố `sim_`, không đưa vào mô hình học máy.
- [x] Giao diện Web không hard-code số liệu, ẩn danh toàn bộ thông tin định danh và công khai sử dụng AI.
- [x] Kịch bản `run_pipeline.py` chạy lại từ đầu cho kết quả trùng khớp hoàn toàn (100% tái lập).
