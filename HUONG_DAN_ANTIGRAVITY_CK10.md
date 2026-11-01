# HƯỚNG DẪN CHO ANTIGRAVITY — ĐỒ ÁN CK10 (E-LOGISTICS)

**Đề tài:** Thiết kế chuỗi dịch vụ logistics tự động có AI ra quyết định — nghiệp vụ kho (Inventory)
**Sản phẩm:** Repo code (NumPy thuần) + Notebook + **Website** trình bày/demo
**Dữ liệu:** Olist Brazilian E-Commerce **bản gốc** (Kaggle `olistbr/brazilian-ecommerce`, Version 2)

---

## 0. CÁCH DÙNG FILE NÀY

1. Lưu file này ở gốc dự án (tên `HUONG_DAN.md`). Bắt agent **đọc hết** trước khi viết dòng code đầu tiên.
2. Làm theo **7 giai đoạn** ở Mục 6. Cuối mỗi giai đoạn: chạy test → ghi kết quả vào `outputs/` → tóm tắt cho nhóm → **DỪNG, chờ nhóm duyệt**.
3. Mục 4 và Phụ lục A–C là **kết quả đã chạy thử trên dữ liệu thật**. Nếu code của bạn ra số lệch xa Mục 4.9 thì có lỗi — tìm lỗi, **không chỉnh cho khớp**.
4. Phụ lục A–C là code tham chiếu đã kiểm chứng (số liệu khớp từng chữ số). Được phép viết lại cho gọn/đẹp, nhưng phải qua cùng bộ test.

**Prompt khởi động (dán vào Antigravity):**

```
Đọc toàn bộ file HUONG_DAN.md ở gốc dự án. Đây là đồ án nhóm môn E-logistics (CK10).
Làm đúng theo 7 giai đoạn trong Mục 6, MỖI LẦN CHỈ 1 GIAI ĐOẠN, xong thì dừng và báo cáo.
Bắt đầu Giai đoạn 0. Tuân thủ tuyệt đối Mục 2 (quy tắc bắt buộc) và Mục 10 (điều cấm).
Nếu thấy mâu thuẫn hoặc thiếu thông tin, hỏi nhóm — không tự đoán.
```

---

## 1. YÊU CẦU CỦA ĐỀ (đã đối chiếu file xlsx và đề cương Word)

| Nguồn | Yêu cầu | Cách đáp ứng trong dự án |
|---|---|---|
| xlsx, CK10 | Phân tích quy trình nhận đơn → kho → vận tải → giao hàng; chỉ ra điểm tự động hóa và điểm cần con người | Web trang "Quy trình" (sơ đồ swimlane As-Is / To-Be), Mục 8 |
| xlsx, CK10 | AI cho ≥ 1 quyết định: ưu tiên đơn / phân công nguồn lực / dự đoán trễ | **Dự đoán nguy cơ trễ bàn giao → xếp mức ưu tiên P1/P2/P3** (Mục 4.1) |
| xlsx, CK10 | Mô phỏng luồng trước/sau | Mô phỏng hàng đợi As-Is (FIFO) vs To-Be, Mục 7 |
| xlsx, CK10 | So sánh lead time, tỷ lệ đúng hạn, số quyết định thủ công; đánh giá rủi ro khi AI sai | Mục 7.3 (3 KPI) và web trang "Rủi ro" |
| xlsx, khung 5 bước | Vấn đề & KPI → Dữ liệu → AI → Đánh giá (có baseline) → Giải pháp & quản trị | Mục 6 + web; **có baseline** ở Mục 4.5 |
| xlsx | ≥ 3 tài liệu học thuật; bảo mật, đạo đức dữ liệu, trung thực khi dùng AI | Web trang "Tài liệu & đạo đức"; `AI_USAGE_LOG.md` |
| Word, 2.3 | Decision Tree / Random Forest; chia 80/20 hoặc K-Fold; tinh chỉnh max depth, min samples split, criterion | Mục 4.5, Giai đoạn 4 |
| Word, 2.3 bước 6 | Hậu xử lý: độ tin cậy thấp → đẩy cảnh báo Trưởng ca | Mục 4.8 |
| Word, 2.1.2 | Điểm bắt buộc có người: kiểm tra cảm quan hàng cận date, lệch tồn, hủy đơn phút chót | Web trang "Quy trình" + lớp luật FEFO (Mục 4.7) |

**Cách làm theo 4 repo mẫu (LogR, KNN, HGB, K-Means):** `model.py` (thuật toán NumPy thuần) → `run_pipeline.py` (chạy 21 bước, in ra CLI) → `weights.py` (lưu/nạp trọng số) → `predict_new.py` (suy luận mẫu mới) → `tests/` → `notebook.ipynb` → thư mục `weights/` và `outputs/`. Quy trình 21 bước và các điểm chống rò rỉ lấy từ HGB/KNN; `class_weight`, loss history lấy từ LogR; so sánh nhiều biến thể bằng cùng một bộ metric lấy từ K-Means.

---

## 2. QUY TẮC BẮT BUỘC (KHÔNG THƯƠNG LƯỢNG)

| # | Quy tắc |
|---|---|
| R1 | **Dữ liệu gốc.** Chỉ dùng 5 file CSV gốc trong `data/raw/`. Không sửa, không ghi đè, không dùng bản đã làm sạch của người khác. Mọi xử lý nằm trong code và được log. |
| R2 | **ML lõi viết thuần NumPy** (cây quyết định, Random Forest, Logistic Regression, split, K-Fold, GridSearch, metric), giống 4 repo mẫu. `scikit-learn` chỉ được import trong `tests/` để đối chiếu, **cấm trong `src/`**. |
| R3 | **Không rò rỉ.** Mỗi đặc trưng chỉ dùng thông tin biết được tại `order_approved_at`. Tập test chỉ mở **một lần** ở cuối (Mục 9.2). Không chọn tham số/ngưỡng bằng test. |
| R4 | **Không bịa.** Không bịa tài liệu tham khảo, số liệu, kết quả. Mọi con số trên web/báo cáo phải đọc từ file do code sinh ra (`outputs/*.json`). |
| R5 | **Trung thực khi dùng AI.** Tạo `AI_USAGE_LOG.md` ghi từng việc agent làm. Nhóm phải đọc hiểu code trước khi nộp, vì đề yêu cầu tính trung thực học thuật khi dùng AI. |
| R6 | **Báo cáo kết quả thật, kể cả khi không đẹp** (xem Mục 4.9 và 11). Cấm chỉnh tham số mô phỏng để AI "thắng". |
| R7 | **Tái lập.** Seed cố định (42), `requirements.txt` ghi version, chạy `python run_pipeline.py` từ đầu ra đúng kết quả. |

---

## 3. DỮ LIỆU — ĐÃ KIỂM TRA SẴN

5 file người dùng tải từ Kaggle khớp **từng byte** với bản đã đo (xác nhận là bản chưa làm sạch).

| File | Dòng × cột | Null / bất thường |
|---|---|---|
| `olist_orders_dataset.csv` | 99,441 × 8 | `order_approved_at` 160; `order_delivered_carrier_date` 1,783; `order_delivered_customer_date` 2,965 |
| `olist_order_items_dataset.csv` | 112,650 × 7 | không null |
| `olist_products_dataset.csv` | 32,951 × 9 | 610 dòng thiếu danh mục/tên/ảnh; 2 dòng thiếu cân nặng và kích thước. Tên cột sai chính tả (`product_name_lenght`) — **giữ nguyên** |
| `olist_sellers_dataset.csv` | 3,095 × 4 | không null |
| `olist_customers_dataset.csv` | 99,441 × 5 | không null; `customer_id` duy nhất theo đơn, `customer_unique_id` 96,096 |

**Lỗi logic trong dữ liệu (ghi log, KHÔNG xóa):** 166 đơn có ngày giao vận tải trước ngày đặt; 23 đơn có ngày khách nhận trước ngày giao vận tải; 117 đơn trong cohort bị đánh dấu `ts_anomaly`.

`order_status`: delivered 96,478; shipped 1,107; canceled 625; unavailable 609; invoiced 314; processing 301; created 5; approved 2.

**Phễu lọc (code Phụ lục A in ra đúng các số này):**

```
99,441 đơn → delivered, đủ 3 mốc thời gian, 1 seller: 95,180 đơn
→ cửa sổ 2017-01-01 đến 2018-07-31: 88,526 đơn (cohort)
   train_fit 55,808 (2017-01-05 → 2018-02-27) | val 13,953 (→ 2018-04-30) | test 18,765 (2018-05-01 → 2018-07-31)
```

**Vì sao cắt cửa sổ (phải ghi trong báo cáo):**
- **Tháng 8/2018 trở đi bị "kiểm duyệt phải" (right-censoring):** chỉ giữ đơn `delivered`, nên đơn giao muộn chưa về bị loại. Hậu quả: với nhãn `late_delivery`, quy tắc "SLA ngắn xử lý trước" cho AUC giả 0.816 riêng tháng 8/2018 (các tháng khác 0.43–0.65). Nếu để lẫn vào test, mọi kết quả sai.
- **Năm 2016 chỉ có vài trăm đơn** (thực chất tháng 10/2016), hành vi khác hẳn (SLA trung vị ~53 ngày, trễ bàn giao 49%).
- **Có cú sốc theo thời gian:** tỷ lệ trễ giao khách 14.3% (11/2017), 15.5% và 21.8% (2–3/2018), 1.4% (6/2018). Đây là lý do chia theo thời gian và không dùng `month` làm đặc trưng.

---

## 4. THIẾT KẾ ĐÃ CHỐT (KÈM BẰNG CHỨNG)

### 4.1 Bài toán và nhãn

- **Nhãn chính `late_handover`** = `order_delivered_carrier_date` > `shipping_limit_date` (nhỏ nhất trong đơn). Nghĩa: kho/người bán **không kịp bàn giao cho vận tải đúng hạn** — đúng phạm vi nghiệp vụ kho (Inventory) và "giờ chốt xe" trong đề. Tỷ lệ dương: 9.0% (cohort), 7.5% (test).
- **Nhãn phụ `late_delivery`** = `order_delivered_customer_date` > `order_estimated_delivery_date` — KPI đầu-cuối (SLA với khách). 8.0% (cohort), 4.8% (test). Đơn trễ bàn giao thì trễ giao khách 19.7% so với 3.6% ở đơn đúng hạn.
- **Cả hai nhãn là kết quả thật trong dữ liệu**, không tự tạo bằng luật từ đặc trưng đầu vào. Lý do: nếu gán nhãn ưu tiên bằng luật rồi cho cây học lại thì cây chỉ "học thuộc" luật, kết quả đẹp mà vô nghĩa.
- **Ba mức ưu tiên P1/P2/P3 là hậu xử lý** từ điểm rủi ro của mô hình nhị phân (Mục 4.6), không phải nhãn gốc. Ghi rõ điều này trong báo cáo (đề cương Word 2.2.1 nói "đa lớp", cần sửa thành "nhị phân + 3 mức ưu tiên").
- Chỉ lấy đơn **một người bán** (98.7% đơn có item), vì cột ngày giao vận tải là của cả đơn, đơn nhiều người bán thì không biết ngày nào của người bán nào.

### 4.2 Đặc trưng (20 cột, `data_prep.FEATURES`)

Tất cả biết được tại `order_approved_at`. Công thức đầy đủ trong Phụ lục A.

| Nhóm | Cột | Ý nghĩa |
|---|---|---|
| Thời hạn | `min_to_limit_h` | giờ còn lại từ lúc duyệt đến hạn bàn giao (≈ biến x1 trong đề cương) |
| | `sla_days` | số ngày từ lúc duyệt đến ngày giao ước tính |
| | `approve_lag_h` | giờ từ đặt đến duyệt |
| Đơn hàng | `n_items`, `n_products`, `price`, `freight`, `freight_ratio`, `weight`, `vol` | quy mô, cân nặng, thể tích (vol = dài × cao × rộng) |
| Địa lý | `same_state`, `cust_SP`, `sell_SP` | cùng bang, khách/người bán ở bang SP |
| Thời gian | `hour`, `dow` | giờ và thứ trong tuần lúc duyệt |
| Tải (≈ năng lực kho, x4) | `global_24h`, `seller_24h` | số đơn được duyệt trong 24h trước, toàn hệ thống và riêng người bán |
| Lịch sử (**knowledge-time**) | `seller_han_rate` | tỷ lệ trễ bàn giao lịch sử của người bán, làm mượt (k=20) |
| | `seller_late_rate`, `lane_late_rate` | tỷ lệ trễ giao khách lịch sử của người bán và của tuyến bang→bang |

**Quy tắc lịch sử (quan trọng nhất để không rò rỉ):** tại thời điểm duyệt `t`, chỉ tính các đơn cùng người bán đã được **giao xong trước `t`** (`order_delivered_customer_date ≤ t`). Hằng số tiên nghiệm (prior) chỉ tính trên các đơn trước `2018-05-01`. Dùng ngày giao cuối cùng là cách thận trọng (ngày giao vận tải thực ra biết sớm hơn).

**Đặc trưng KHÔNG được dùng:** mọi cột ngày của chính đơn đó sau thời điểm duyệt, `order_status`, review, payment, `customer_unique_id`, `month`.

**Thiếu giá trị:** `weight`, `vol` điền bằng trung vị **của tập train** (`fill_train_median`). Trong cohort hiện không còn null ở hai cột này nhưng vẫn giữ hàm để an toàn.

### 4.3 Bước 5 và 6 trong 21 bước (scaling và biến phân loại)

- Cây quyết định **bất biến với thang đo** → không scale. Chỉ `StandardScaler` (fit trên train) khi chạy baseline Logistic Regression.
- Bang (state) → cờ nhị phân `cust_SP`, `sell_SP`, `same_state` + `lane_late_rate`. Không one-hot 27 bang.
- Phải **giải thích trong báo cáo** vì sao không scale cho cây (đề cương Word có bước chuẩn hóa Min-Max/Z-score).

### 4.4 Chia dữ liệu

- **Chính: chia theo thời gian** (train_fit / val / test như Mục 3). Lý do: dữ liệu có cú sốc theo mùa; chia ngẫu nhiên làm AUC đẹp giả (0.721 so với 0.667).
- **So sánh: chia ngẫu nhiên phân tầng 80/20** (như 4 repo mẫu), ghi vào bảng và nhãn "lạc quan".
- Validation là 20% cuối của giai đoạn train (theo thời gian). Dùng để chọn ngưỡng và cắt P1/P2/P3.
- Cross-validation khi tuning: **expanding-window 4 fold** trên giai đoạn train (chính) và `StratifiedKFold(5)` (so sánh).
- Không refit trên train+val. Mô hình cuối = huấn luyện trên `train_fit`, ngưỡng khóa từ validation, test mở một lần.

### 4.5 Mô hình và baseline (bảng No Free Lunch, bước 12)

| Mô hình | Vai trò |
|---|---|
| Dummy (đoán đa số) | baseline sàn |
| Quy tắc FIFO (điểm = −thời gian duyệt) | As-Is trong đề cương |
| Quy tắc "slack" (điểm = −`min_to_limit_h`) | đơn gấp nhất xử lý trước |
| **Quy tắc lịch sử người bán** (điểm = `seller_han_rate`) | baseline mạnh một đặc trưng |
| Logistic Regression NumPy (gradient descent, L2, `class_weight`, như LogR repo) | baseline tuyến tính |
| **Decision Tree NumPy** (Gini/entropy, quantile bins) | **mô hình lõi theo đề** |
| Random Forest NumPy (bagging 100 cây, `max_features=√d`, depth 8, leaf 50) | so sánh |

- Siêu tham số cây (lưới): `max_depth ∈ {3,4,5,6,8}`, `min_samples_leaf ∈ {50,100,200,400}`, `min_samples_split ∈ {2·leaf}`, `criterion ∈ {gini, entropy}`, `n_bins ∈ {32,64}`. Chọn theo **PR-AUC** (CV expanding-window), không chọn theo accuracy.
- Dùng cây **không đặt `class_weight`** để điểm gần với xác suất; vẫn **không diễn giải điểm như xác suất tuyệt đối** (Brier 0.073 kém hằng số 0.070 do tỷ lệ trễ giảm từ 9.8% xuống 7.5% theo thời gian). Báo cáo Brier và bảng hiệu chuẩn.
- Cây tham chiếu (Phụ lục B) thiếu `criterion=entropy`, `min_samples_split`, `max_features` — **Giai đoạn 3 phải bổ sung**.

### 4.6 Ngưỡng và mức ưu tiên

- **Điểm rủi ro** `p` = tỷ lệ dương của lá. P1/P2/P3 cắt bằng phân vị của `p` trên **validation**: `t1` = phân vị 85%, `t2` = phân vị 45%. Khóa `t1`, `t2` vào `weights/thresholds.json` trước khi chạm test.
- Ngưỡng nhị phân (để lập confusion matrix): chọn theo F1 lớn nhất trên validation, hoặc theo chi phí `3·FN + 1·FP` (C_FN=3, C_FP=1 như HGB repo; ghi đây là giả định).
- **Trôi phân phối:** trên test, điểm thấp hơn trên validation nên P3 chiếm 37% thay vì 45%. Báo cáo tỷ lệ thật, không cắt lại bằng test.

### 4.7 FEFO (hạn dùng) và tồn kho an toàn — Olist KHÔNG có

Olist không có hạn dùng, lô, tồn kho. **Không bịa cột rồi cho mô hình học** (cột ngẫu nhiên chỉ là nhiễu). Cách làm trung thực:
- Thêm **lớp luật nghiệp vụ hậu xử lý** (đề cương Word, bước 6): nếu người dùng nhập hạn dùng còn lại ≤ N ngày (mặc định N=3) thì **nâng một mức ưu tiên** và gắn cờ "kiểm cảm quan trước khi nhả đơn" (điểm bắt buộc có người ở 2.1.2).
- Hạn dùng chỉ là **ô nhập trong demo web**, ghi rõ **"giả định, không học từ dữ liệu"**. Không có trong mô hình, không có trong mô phỏng.
- Báo cáo nêu rõ: x2 (FEFO) và x3 (tồn an toàn) của đề cương là **giới hạn dữ liệu**. Tùy chọn: nhóm tham khảo phân phối hạn dùng thật của dataset UCI "Stock Keeping Units" (id 531) làm cơ sở chọn ngưỡng N, ghi rõ nguồn.

### 4.8 Hậu xử lý và Trưởng ca (đề cương Word, bước 6)

Đơn cần **can thiệp thủ công** = `P1` **hoặc** `seller_hist_n < 20` (người bán chưa đủ lịch sử, độ tin cậy thấp). Trên test: P1 = 14.4%, người bán thiếu lịch sử = 27.2%, hợp = **38.9%** (bắt 52.2% đơn trễ thật). Tham số 20 phải được báo cáo độ nhạy (10/20/50).

### 4.9 KẾT QUẢ THAM CHIẾU (đo trên test 2018-05-01 → 2018-07-31, 18,765 đơn, tỷ lệ trễ bàn giao 7.5%)

| Hạng mục | Giá trị tham chiếu |
|---|---|
| Cây NumPy depth5/leaf200 — ROC-AUC / PR-AUC | **0.667 / 0.154** (depth 7: 0.675; depth 3: 0.663) |
| Quy tắc `seller_han_rate` một đặc trưng — AUC | **0.671 (bằng cây)** |
| Quy tắc FIFO / slack — AUC | 0.465 / 0.487 (≈ ngẫu nhiên) |
| Cây chỉ dùng đặc trưng đơn hàng (không lịch sử) — AUC | **0.509 (vô dụng)** |
| Logistic Regression / Random Forest — AUC | 0.684 / 0.700 (PR 0.156 / 0.172; sklearn đối chiếu, bản NumPy lệch ≤ 0.02) |
| Cây, chia ngẫu nhiên (lạc quan) — AUC | 0.721 |
| Ngưỡng F1 (val) = 0.228 — Precision / Recall / F1 | 0.173 / 0.385 / 0.239 (TN 14,781; FP 2,582; FN 862; TP 540) |
| P1 (14.4% đơn) | trễ 17.5% (gấp 2.3 lần nền), chứa 33.7% đơn trễ |
| P2 (48.3% đơn) | trễ 6.9%, chứa 44.4% |
| P3 (37.3% đơn) | trễ 4.4%, chứa 21.9% |
| Recall khi chỉ cảnh báo top 15% đơn | cây 0.35; luật người bán 0.39; slack 0.13; FIFO 0.12; ngẫu nhiên 0.15 |
| Nhãn phụ `late_delivery`, cây không lịch sử — AUC / PR-AUC | 0.613 / 0.077 (nền 4.8%) |
| Đặc trưng quan trọng nhất (tổng số mẫu tại các nút chia theo từng đặc trưng) | `seller_han_rate` 64%, `min_to_limit_h` 12%, `seller_24h` 8%, `vol` 8% |

**Cờ đỏ (dừng và kiểm tra):**
- AUC trên test > 0.80 → gần như chắc chắn rò rỉ.
- AUC < 0.55 khi đã có đặc trưng lịch sử → lỗi tính lịch sử (kiểm tra knowledge-time).
- Cây "thắng" luật người bán rõ rệt (> 0.03 AUC) → kiểm tra lại, số tham chiếu cho thấy hai bên ngang nhau.
- Mô phỏng cho AI vượt xa EDD → kiểm tra vòng lặp tự chứng minh (Mục 7.4).

**Thông điệp trung thực rút ra (đưa thẳng vào báo cáo):** tín hiệu dự đoán trễ trong Olist đến chủ yếu từ **hành vi lịch sử của người bán**, không phải từ đặc điểm đơn hàng. Cây quyết định xếp hạng rủi ro tốt hơn FIFO/slack rất nhiều (recall top-15%: 0.35 so với 0.12) nhưng **không hơn quy tắc một đặc trưng** — lợi thế của cây là minh bạch và gộp nhiều đặc trưng, không phải độ chính xác.

---

## 5. CẤU TRÚC REPO

```
ck10-olist-priority/
├── HUONG_DAN.md                  # file này
├── README.md                     # hướng dẫn chạy, kết quả, hạn chế
├── AI_USAGE_LOG.md               # nhật ký dùng AI (bắt buộc)
├── requirements.txt              # numpy, pandas, matplotlib, nbformat, jupyter, pytest ; scikit-learn CHỈ cho tests
├── .gitignore                    # data/raw/*.csv, __pycache__, .ipynb_checkpoints
├── data/
│   └── raw/                      # 5 CSV gốc (KHÔNG commit, có README chỉ link Kaggle)
├── src/
│   ├── __init__.py
│   ├── data_prep.py              # Phụ lục A (load, bảng đơn, đặc trưng, cohort, split)
│   ├── model.py                  # BinMapper, DecisionTree, RandomForest, LogisticRegression, StandardScaler, metrics, StratifiedKFold, TimeSeriesSplit, GridSearch
│   ├── evaluate.py               # metric, bootstrap CI, bảng hiệu chuẩn, triage recall@k, phân tích lỗi
│   ├── simulate.py               # Phụ lục C
│   ├── weights.py                # lưu/nạp cây (.json + .npz), kiểm tra khứ hồi
│   └── export_web.py             # xuất JSON cho web
├── run_pipeline.py               # chạy 21 bước, in CLI, ghi outputs/
├── predict_new.py                # nhận CSV đơn mới → điểm rủi ro, P1/P2/P3, cờ thủ công
├── build_notebook.py             # sinh và chạy notebook.ipynb bằng nbformat (như HGB repo)
├── notebook.ipynb
├── tests/                        # Mục 9.1
├── weights/
│   ├── tree.json                 # cấu trúc cây (nút, ngưỡng thật, p, n)
│   ├── thresholds.json           # t1, t2, ngưỡng F1, prior, median
│   ├── best_params.json
│   ├── feature_names.json
│   └── README.md
├── outputs/                      # *.json, *.csv, *.png do code sinh ra
└── web/                          # Mục 8 (tĩnh, triển khai GitHub Pages)
```

---

## 6. CÁC GIAI ĐOẠN

Ánh xạ 21 bước (theo HGB/KNN repo) → giai đoạn → file:

| Bước | Nội dung | GĐ | File chính |
|---|---|---|---|
| 1–3 | Bài toán, bản chất ML, khảo sát dữ liệu | 1 | `outputs/01_data_audit.md`, notebook |
| 4 | Khám phá và làm sạch (chỉ ghi nhận) | 1 | `data_prep.py` |
| 5–6 | Scaling, biến phân loại | 1 | notebook (Mục 4.3) |
| 7 | Chọn thuật toán, hàm mất mát | 2 | notebook |
| 8 | Kỹ thuật đặc trưng | 2 | `data_prep.add_features` |
| 9–10 | Chia dữ liệu, kiểm soát rò rỉ | 2 | `tests/test_leakage.py` |
| 11 | Baseline | 3 | `evaluate.py` |
| 12 | No Free Lunch | 3 | bảng so sánh 7 mô hình |
| 13 | Bias–Variance | 3 | train vs val vs test theo `max_depth` |
| 14 | Tuning | 4 | `model.GridSearch` |
| 15 | Metric | 4 | `evaluate.py` |
| 16 | Cross-validation | 4 | expanding-window + Stratified |
| 17 | Huấn luyện, lưu weights | 4 | `weights/` |
| 18 | Kiểm định thống kê | 5 | bootstrap CI |
| 19 | Phân tích lỗi | 5 | `outputs/19_error_analysis.*` |
| 20 | Giải thích mô hình | 5 | text dump cây, importance |
| 21 | Vòng lặp cải tiến, triển khai, mô phỏng | 5–6 | bảng Iter 1→4, `simulate.py`, web |

### Giai đoạn 0 — Khởi tạo
- Tạo cấu trúc Mục 5, `requirements.txt` (ghi version đang dùng), `.gitignore`, `AI_USAGE_LOG.md`, `README.md` khung.
- `data/raw/README.md`: link Kaggle, **ghi chú kiểm tra giấy phép** (nhóm tự kiểm tra trên trang Kaggle; không commit CSV).
- **Nghiệm thu:** `pytest` chạy được (0 test), `python -c "import numpy, pandas"` ổn.

### Giai đoạn 1 — Dữ liệu (bước 1–6)
- Viết `src/data_prep.py` (từ Phụ lục A). Hàm `audit_raw()` ghi `outputs/01_data_audit.json` và `.md`: shape, null từng cột, đếm `order_status`, các lỗi logic (166 / 23), phễu lọc, tỷ lệ nhãn theo tháng, khoảng thời gian.
- Biểu đồ EDA (matplotlib, lưu `outputs/fig_*.png`): đơn theo tháng; tỷ lệ trễ bàn giao và trễ giao khách theo tháng (thấy cú sốc 11/2017, 2–3/2018 và tháng 8/2018 bị kiểm duyệt); phân phối `min_to_limit_h`, `sla_days`; tỷ lệ trễ theo nhóm SLA.
- **Nghiệm thu:** phễu lọc ra đúng 99,441 → 95,180 → 88,526; null khớp bảng Mục 3; **không có file nào trong `data/raw/` bị sửa** (so sánh SHA256 trước/sau, lưu vào audit).

### Giai đoạn 2 — Đặc trưng và chia dữ liệu (bước 7–10)
- `add_features`, `select_cohort`, `time_split` (Phụ lục A). Thêm chia ngẫu nhiên phân tầng 80/20.
- Viết **kiểm toán rò rỉ** in ra CLI như HGB repo (mỗi dòng `[PASS]`/`[FAIL]`): không trùng chỉ mục train/val/test; mọi đơn test duyệt sau mọi đơn train; mọi đặc trưng lịch sử của đơn có `n` chỉ đến từ đơn giao xong trước `order_approved_at`; prior chỉ từ đơn trước 2018-05-01; trung vị điền thiếu chỉ từ train; ngưỡng chỉ từ validation; test chỉ dùng ở bước cuối.
- **Nghiệm thu:** `test_leakage.py` pass; số dòng train/val/test = 55,808 / 13,953 / 18,765.

### Giai đoạn 3 — Mô hình và baseline (bước 11–13)
- `src/model.py`: bắt đầu từ Phụ lục B (đã kiểm chứng AUC khớp sklearn trong 0.005). Bổ sung `criterion`, `min_samples_split`, `max_features`, xuất cấu trúc cây, `feature_importance` (tổng giảm Gini có trọng số). Viết `RandomForest` (bagging), `LogisticRegression` (GD, L2, `class_weight`, lưu `loss_history_`), `StandardScaler`, các metric (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, Brier).
- Bảng so sánh 7 mô hình ở Mục 4.5 trên cùng một split. Bias–variance: vẽ AUC train/val/test theo `max_depth` 1–12.
- **Nghiệm thu:** kết quả trong khoảng Mục 4.9; `test_tree_vs_sklearn.py` pass (xem 9.1).

### Giai đoạn 4 — Tuning, CV, huấn luyện cuối (bước 14–17)
- GridSearch (Mục 4.5) bằng expanding-window 4 fold, tiêu chí PR-AUC; in cả kết quả `StratifiedKFold(5)` để so sánh.
- Quét ngưỡng trên validation → khóa `t1`, `t2`, ngưỡng F1, ngưỡng chi phí. Lưu `weights/`.
- Chỉ **sau khi khóa** mới tính chỉ số trên test **một lần**; ghi `outputs/04_final_metrics.json`.
- **Nghiệm thu:** `weights.py` lưu rồi nạp lại cho dự đoán giống hệt (test khứ hồi); không tham số nào chọn theo test.

### Giai đoạn 5 — Thống kê, lỗi, giải thích, mô phỏng (bước 18–21)
- **Bước 18:** bootstrap 1,000 lần trên test: khoảng tin cậy 95% của AUC/PR-AUC; **bootstrap có cặp** hiệu AUC giữa cây và luật `seller_han_rate`, giữa cây và FIFO. Báo cáo đúng kết quả (khoảng tin cậy hiệu cây − luật người bán có thể chứa 0). AUC theo từng tháng trong test (5, 6, 7/2018).
- **Bước 19:** confusion matrix; đơn trễ bị bỏ sót (FN) theo nhóm: danh mục, bang, mức lịch sử người bán; ca sát ngưỡng `t1 ± 0.03`.
- **Bước 20:** in cây dạng text; đường đi của 5 đơn mẫu; importance (tổng giảm Gini) và permutation importance trên validation.
- **Bước 21:** bảng lặp cải tiến — Iter 1 Dummy; Iter 2 cây với đặc trưng đơn hàng (AUC 0.509); Iter 3 thêm lịch sử người bán (0.667); Iter 4 tuning và khóa ngưỡng; Iter 5 (tùy chọn) Random Forest.
- Mô phỏng: Mục 7.
- **Nghiệm thu:** các `outputs/*.json` đủ cho web ở Giai đoạn 6.

### Giai đoạn 6 — Xuất dữ liệu và xây web
- `export_web.py`, `predict_new.py`, web Mục 8.
- **Nghiệm thu:** test khớp Python–JavaScript (Mục 9.1); web chạy offline bằng `python -m http.server` trong `web/`.

### Giai đoạn 7 — Tài liệu và rà soát
- `README.md` đầy đủ; `build_notebook.py` sinh và chạy `notebook.ipynb`; hoàn thiện `AI_USAGE_LOG.md`.
- Chạy lại `python run_pipeline.py` từ đầu (xóa `outputs/`, `weights/`) → số liệu phải trùng lần chạy trước.
- Rà soát với Mục 9.2 (checklist) và Mục 10 (điều cấm). Lập danh sách cho nhóm các câu trong đề cương Word cần sửa (Mục 12).

---

## 7. MÔ PHỎNG AS-IS vs TO-BE

### 7.1 Thiết lập (code tham chiếu: Phụ lục C)

- Dùng **toàn bộ đơn của cửa sổ test** (18,765 đơn, 91 ngày, ~206 đơn/ngày làm việc). Thời điểm đơn đến (`order_approved_at`) và hạn bàn giao (`shipping_limit_date`) là **dữ liệu thật**.
- **Giả định phải ghi rõ trong báo cáo (không có trong dữ liệu):** một trung tâm xử lý duy nhất; K nhân viên; ca 08:00–17:00 thứ Hai đến thứ Bảy; thời gian xử lý một đơn = 6 + 3·(n_items − 1) phút; không ngắt giữa chừng, việc dở được chuyển sang ca sau.
- **Hiệu chuẩn K:** chọn K sao cho FIFO mô phỏng gần tỷ lệ trễ bàn giao thật (7.5%). Với K=3, FIFO = 7.2% (khớp). Đây là cách neo As-Is vào thực tế thay vì chọn số tùy ý.
- Chạy nhạy: K ∈ {3,4,5,6} × hệ số thời gian xử lý ∈ {0.8, 0.9, 1.0}. Xuất toàn bộ lưới ra `outputs/05_simulation_grid.json`.

### 7.2 Ba chính sách

| Chính sách | Khóa ưu tiên (nhỏ làm trước) | Ý nghĩa |
|---|---|---|
| **FIFO** (As-Is theo đề cương) | thứ tự đến | vận hành truyền thống |
| **EDD** (baseline mạnh, bổ sung) | hạn bàn giao | quy tắc kinh điển, không cần AI |
| **AI-EDD** (To-Be) | hạn − 24h × điểm rủi ro `p` | đơn rủi ro cao được xử lý sớm hơn theo tỷ lệ |

Thêm một chính sách minh họa **"AI level-first"** (làm hết P1 rồi P2 rồi P3, bỏ qua hạn) để cho thấy **rủi ro dùng AI sai cách** (Mục 3.3 của đề cương). Không dùng làm To-Be.

### 7.3 KPI (đúng 3 chỉ số đề yêu cầu)

1. **Lead time xử lý trong kho** = từ duyệt đến hoàn tất xử lý (trung bình và p95, giờ).
2. **Tỷ lệ đúng hạn** = % đơn hoàn tất trước `shipping_limit_date`.
3. **Số quyết định thủ công (%)**: As-Is = 100% đơn do người sắp xếp thứ tự (**giả định**); To-Be = % đơn cần can thiệp theo Mục 4.8 (38.9%). Công bố rõ đây là định nghĩa của nhóm.

### 7.4 Kết quả tham chiếu (code Phụ lục C, K=3, 100% thời gian xử lý, 18,765 đơn)

| Chính sách | Trễ hạn | Lead time TB | Lead time p95 |
|---|---|---|---|
| FIFO | **7.2%** (thật: 7.5%) | 33.2 h | 108 h |
| EDD | **0.8%** | 33.2 h | 121 h |
| AI-EDD (Δ=24h) | **0.9%** | 33.2 h | 121 h |
| AI level-first (minh họa sai) | **11.9%** | 33.6 h | 189 h |

Khi kho bớt quá tải (K ≥ 4, hoặc thời gian xử lý × 0.8) FIFO, EDD và AI-EDD đều chỉ trễ ~0.8–1.0%, chênh lệch gần như biến mất; riêng "AI level-first" vẫn tệ hơn (3.0% ở K=3 × 0.8). Lead time TB bằng nhau giữa các chính sách vì hệ thống không để người rảnh khi còn việc. Với K=3 và 100% thời gian xử lý, công suất sử dụng ≈ 95%.

**Phải báo cáo đúng như vậy:**
- Cải thiện từ 7.2% xuống ~0.8–0.9% đến từ **sắp xếp theo hạn (EDD)**, một quy tắc đơn giản. **AI-EDD không tốt hơn EDD** trong mô phỏng hàng đợi.
- Lý do: mô hình dự đoán trễ do **hành vi người bán**, còn mô phỏng chỉ tạo trễ do **nghẽn hàng đợi**. Hai cơ chế khác nhau nên điểm rủi ro không giúp ích thêm cho lịch xử lý.
- Giá trị thực của mô hình nằm ở **phân loại/cảnh báo (triage)**: chỉ xem top 15% đơn đã bắt được 35% đơn trễ (FIFO: 12%). Đề xuất nhóm trình bày AI là **công cụ cảnh báo để Trưởng ca can thiệp**, kết hợp EDD để sắp lịch.
- **Cấm** chỉnh thời gian xử lý, K, Δ hoặc cách tạo trễ để AI-EDD thắng EDD. Nếu muốn mô phỏng trễ do người bán, phải rút từ phân phối trễ thực trong dữ liệu và công bố, và coi là kịch bản bổ sung chứ không phải kết quả chính.
- Lead time **thật** (duyệt → giao vận tải) trung bình 54.6h trong test. Khác với lead time mô phỏng 33h vì bao gồm thời gian người bán, **không được so trực tiếp**.

---

## 8. WEBSITE

### 8.1 Yêu cầu kỹ thuật
- **Web tĩnh** (HTML + CSS + JavaScript thuần), không cần máy chủ; triển khai GitHub Pages từ thư mục `web/`. Mọi số liệu đọc từ `web/data/*.json` do `export_web.py` sinh. Không hard-code số vào HTML.
- Thư viện biểu đồ: Chart.js, **ghim đúng phiên bản** và đặt bản sao trong `web/vendor/` (chạy offline); không dùng `latest`.
- Tiếng Việt, responsive (đọc tốt trên điện thoại), tương phản màu đủ, có `alt` cho ảnh, điều hướng bằng bàn phím được. Không dùng ảnh có bản quyền. Không hiển thị bất kỳ ID khách hàng/người bán nào (rút gọn hoặc mã hóa).

### 8.2 Các trang

| Trang | Nội dung | Dữ liệu |
|---|---|---|
| **1. Tổng quan** | Vấn đề, câu hỏi nghiên cứu, 3 KPI, kết luận ngắn (gồm hạn chế) | `summary.json` |
| **2. Quy trình** | Sơ đồ swimlane Nhận đơn → Kho → Vận tải → Giao chặng cuối, **As-Is vs To-Be**, đánh dấu điểm tự động hóa và điểm cần con người (Word 2.1.2, 2.5), điểm nghẽn tại Inventory | tĩnh (SVG) |
| **3. Dữ liệu** | Nguồn, giấy phép, phễu lọc, biểu đồ EDA, giải thích cắt cửa sổ (kiểm duyệt phải) | `audit.json`, `eda_*.json` |
| **4. Mô hình** | 21 bước tóm tắt, bảng 7 mô hình, tuning, cây dạng sơ đồ tương tác, importance, bảng lặp cải tiến | `metrics.json`, `tree.json` |
| **5. Demo** | Chọn **đơn mẫu thật** từ test (50 đơn, ID rút gọn) hoặc nhập tay → điểm rủi ro, mức P1/P2/P3, **đường đi trong cây** (giải thích), cờ Trưởng ca, ô "hạn dùng còn lại" (lớp luật giả định, Mục 4.7), so với kết quả thực tế sau đó | `tree.json`, `sample_orders.json` |
| **6. Mô phỏng** | Chọn K và hệ số thời gian → bảng và biểu đồ 3 KPI cho FIFO/EDD/AI-EDD (đọc từ lưới tính sẵn) + lời giải thích trung thực Mục 7.4 | `simulation_grid.json` |
| **7. Rủi ro** | Confusion matrix, thanh trượt C_FN và C_FP tính lại chi phí theo ngưỡng (từ bảng quét tính sẵn), kịch bản AI sai (FN: đơn gấp bị xếp P3 → lỡ chuyến; FP: chiếm nguồn lực), triage recall@k | `threshold_sweep.json`, `triage.json` |
| **8. Tài liệu và đạo đức** | Giấy phép dữ liệu, ẩn danh, quyền riêng tư, minh bạch, trách nhiệm khi dùng AI, **công khai việc dùng AI**, danh mục tài liệu tham khảo | tĩnh |

**Tài liệu tham khảo:** để khung `[NHÓM ĐIỀN: tác giả, năm, tên, nguồn]` cho **tối thiểu 3 tài liệu học thuật**. **Cấm agent tự bịa trích dẫn.**

### 8.3 Suy luận trong trình duyệt
- `tree.json`: `{feature_names, nodes:[{f, thr, l, r, p, n}], t1, t2}`; ngưỡng `thr` là **giá trị thật** (không phải chỉ số bin; dùng `DecisionTree.threshold_value` ở Phụ lục B). Code xuất: Phụ lục D.
- **Quy ước rẽ: `x[f] < thr` đi trái, ngược lại đi phải (dấu `<`, KHÔNG phải `<=`).** Đã kiểm chứng: dùng `<=` làm sai 1,730 / 18,765 đơn; dùng `<` khớp Python tuyệt đối (sai số 0.0).
- JavaScript duyệt cây từ gốc, ghi lại đường đi để hiển thị (Phụ lục E). Kết quả phải **trùng Python** (test ở 9.1).
- Đặc trưng lịch sử (`seller_han_rate`…) khi nhập tay: cho chọn mức (thấp/trung bình/cao theo phân vị) kèm giải thích đây là lịch sử người bán.

---

## 9. KIỂM THỬ VÀ NGHIỆM THU

### 9.1 `tests/` (pytest, giống 4 repo mẫu)

| File | Kiểm tra |
|---|---|
| `test_data_prep.py` | phễu 99,441 → 95,180 → 88,526; nhãn là 0/1; không đặc trưng nào NaN sau `fill_train_median`; SHA256 file raw không đổi |
| `test_leakage.py` | split theo thời gian không giao nhau; **với 1,000 đơn ngẫu nhiên, tính lại lịch sử bằng vòng lặp chậm (brute force) chỉ từ đơn giao xong trước `order_approved_at`, so với bản vector hóa — phải bằng nhau**; prior chỉ từ trước 2018-05-01 |
| `test_tree_vs_sklearn.py` | cùng dữ liệu và siêu tham số: chênh AUC giữa bản NumPy và sklearn < 0.01; cây tất định (hai lần chạy giống nhau) |
| `test_metrics.py` | accuracy/precision/recall/F1/ROC-AUC/PR-AUC/Brier khớp sklearn trên 5 mảng nhỏ và 1 mảng ngẫu nhiên |
| `test_split_cv.py` | expanding-window không rò (mọi chỉ số val > mọi chỉ số train trong từng fold); Stratified giữ tỷ lệ nhãn |
| `test_weights.py` | lưu → nạp → dự đoán giống hệt |
| `test_simulate.py` | bảo toàn (mọi đơn được xử lý đúng 1 lần, `done ≥ t_arr`); không hoàn tất vào Chủ nhật hay ngoài 08–17h; FIFO phục vụ đúng thứ tự khi 1 nhân viên |
| `test_web_parity.py` | 200 đơn ngẫu nhiên: điểm từ Python và từ JavaScript (chạy bằng `node`) khác nhau < 1e-9 |

Lưu ý: bản mô phỏng đầu tiên của chúng tôi **từng có lỗi** (nhân viên rảnh sớm lấy nhầm đơn đến muộn nên hoàn tất trước lúc đơn đến). `test_simulate.py` được thiết kế để bắt lỗi này (`done ≥ t_arr`); Phụ lục C đã sửa.

### 9.2 Checklist nghiệm thu (agent tự tick, nhóm kiểm lại)

- [ ] Raw CSV nguyên vẹn (SHA256), dữ liệu không bị làm sạch trước khi vào pipeline
- [ ] `src/` không import `sklearn`
- [ ] Test chỉ được dùng 1 lần, sau khi khóa mô hình và ngưỡng
- [ ] Kết quả nằm trong khoảng Mục 4.9, không dính cờ đỏ
- [ ] Bảng 7 mô hình có đủ baseline (Dummy, FIFO, slack, luật người bán)
- [ ] Mô phỏng: giả định được liệt kê, lưới nhạy được công bố, AI-EDD không bị chỉnh để thắng
- [ ] Báo cáo nêu rõ: tín hiệu chủ yếu từ lịch sử người bán; cây không hơn luật một đặc trưng
- [ ] FEFO ghi là giả định, không nằm trong mô hình
- [ ] Web không hard-code số, không lộ ID, có công khai việc dùng AI
- [ ] Danh mục tài liệu tham khảo do nhóm điền, không bịa
- [ ] `run_pipeline.py` chạy lại từ đầu cho kết quả trùng

---

## 10. ĐIỀU CẤM

1. Dùng review, payment, `order_status` hoặc bất kỳ ngày nào xảy ra sau `order_approved_at` của chính đơn đó làm đặc trưng.
2. Dùng chia ngẫu nhiên làm kết quả chính (chỉ để so sánh, ghi "lạc quan").
3. Giữ đơn từ 2018-08 trở đi hoặc năm 2016 trong cohort.
4. Chọn tham số, ngưỡng, đặc trưng theo kết quả trên test.
5. Import `sklearn` trong `src/`.
6. Sửa/ghi đè file trong `data/raw/` hoặc commit CSV gốc lên Git (kích thước lớn và giấy phép của Olist có điều khoản phi thương mại cần nhóm tự kiểm tra).
7. Tạo cột hạn dùng/tồn kho giả rồi đưa vào mô hình huấn luyện.
8. Chỉnh tham số mô phỏng để AI thắng; viết "AI giảm X% trễ" khi mô phỏng không chứng minh được.
9. Bịa tài liệu tham khảo hoặc số liệu; viết số vào web mà không đọc từ `outputs/`.
10. Hiển thị ID khách hàng/người bán thật trên web.
11. Bỏ qua dừng-chờ-duyệt giữa các giai đoạn.

---

## 11. GIỚI HẠN CẦN NÊU TRONG BÁO CÁO VÀ WEB (Mục 3.3 đề cương)

- **Olist là sàn thương mại điện tử nhiều người bán**, không phải kho của một doanh nghiệp. "Trễ bàn giao" phản ánh hiệu suất của người bán/kho đối tác. Phạm vi "nhóm Inventory" là một sự **ánh xạ**, cần nói rõ.
- Không có hạn dùng, lô, tồn kho an toàn, năng lực kho thật, phương tiện → biến x2, x3, x4, x5 của đề cương chỉ được thay bằng biến thay thế hoặc luật giả định.
- Dữ liệu 2016–2018, một nước (Brazil), đã được nhà phát hành ẩn danh/lấy mẫu → khó khái quát.
- Chỉ đơn một người bán, chỉ đơn đã giao; đơn hủy/không giao bị loại → thiên lệch chọn mẫu.
- Dự đoán yếu (AUC ~0.67) và trôi theo thời gian; điểm rủi ro không phải xác suất hiệu chuẩn.
- Mô phỏng dựa trên giả định thời gian xử lý, không phải đo thực.
- Rủi ro khi AI sai: FN (đơn trễ bị xếp P3) làm lỡ chuyến; FP làm chiếm nguồn lực của đơn khác; dùng điểm rủi ro làm thứ tự làm việc tuyệt đối còn **tệ hơn FIFO** khi kho quá tải (11.9% so với 7.2% trong mô phỏng) → cần luôn kết hợp hạn bàn giao và có người duyệt.
- Quyền riêng tư và công bằng: mô hình dùng bang và hành vi người bán; cần kiểm tra thiên lệch theo bang/nhóm người bán (đề xuất bổ sung trong Giai đoạn 5).

---

## 12. CÁC ĐIỂM CẦN SỬA TRONG ĐỀ CƯƠNG WORD (nhóm tự sửa, không để agent sửa)

| Vị trí | Hiện tại | Nên sửa thành |
|---|---|---|
| Phạm vi (Phần mở đầu, mục 3) | dữ liệu mô phỏng/giả định | dữ liệu thật Olist 2016–2018 (có giả định bổ sung chỉ cho FEFO) |
| 2.2.1 | phân loại đa lớp | phân loại nhị phân nguy cơ trễ bàn giao + 3 mức ưu tiên hậu xử lý |
| 2.2.2 | x2 FEFO, x3 tồn an toàn, x5 giờ chốt xe | biến thực tế ở Mục 4.2; FEFO là lớp luật giả định |
| 2.3 bước 1 | làm sạch dữ liệu giả lập | ghi nhận lỗi và cắt cửa sổ trên dữ liệu thật |
| 2.4.1 | scikit-learn | NumPy thuần (sklearn chỉ để đối chiếu) |
| 2.4.2 | 500–1000 đơn một ca cao điểm | 88,526 đơn thật; mô phỏng 91 ngày (tùy chọn thêm "ca cao điểm") |
| 3.2 | tính lead time, đúng hạn, thủ công | nêu rõ đây là kết quả mô phỏng có giả định; AI-EDD không hơn EDD |

---

# PHỤ LỤC A — `src/data_prep.py` (đã chạy: phễu 95,180 → 88,526; split 55,808 / 13,953 / 18,765)

```python
"""data_prep.py - Dung bang order-level tu 5 file Olist GOC. Khong sua file goc."""
import numpy as np
import pandas as pd

TS_COLS = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date',
           'order_delivered_customer_date', 'order_estimated_delivery_date']
COHORT_START, COHORT_END = '2017-01-01', '2018-08-01'     # [start, end)
FEATURES = ['min_to_limit_h', 'sla_days', 'approve_lag_h', 'n_items', 'n_products', 'price', 'freight',
            'freight_ratio', 'weight', 'vol', 'same_state', 'cust_SP', 'sell_SP', 'hour', 'dow',
            'global_24h', 'seller_24h', 'seller_han_rate', 'seller_late_rate', 'lane_late_rate']


def load_raw(raw_dir):
    r = lambda n, **kw: pd.read_csv(f'{raw_dir}/olist_{n}_dataset.csv', **kw)
    return dict(orders=r('orders', parse_dates=TS_COLS),
                items=r('order_items', parse_dates=['shipping_limit_date']),
                products=r('products'), sellers=r('sellers'), customers=r('customers'))


def build_order_table(raw):
    """1 dong / don. Chi giu: delivered + 1 seller + du 3 moc thoi gian. Gan 2 nhan that."""
    o, it, p = raw['orders'], raw['items'].copy(), raw['products']
    it = it.merge(p[['product_id', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']],
                  on='product_id', how='left')
    it['vol'] = it.product_length_cm * it.product_height_cm * it.product_width_cm
    g = it.groupby('order_id').agg(n_items=('order_item_id', 'count'), n_sellers=('seller_id', 'nunique'),
                                   n_products=('product_id', 'nunique'), price=('price', 'sum'),
                                   freight=('freight_value', 'sum'), weight=('product_weight_g', 'sum'),
                                   vol=('vol', 'sum'), limit=('shipping_limit_date', 'min'),
                                   seller=('seller_id', 'first')).reset_index()
    d = (o.merge(g, on='order_id')
          .merge(raw['sellers'][['seller_id', 'seller_state']], left_on='seller', right_on='seller_id')
          .merge(raw['customers'][['customer_id', 'customer_state']], on='customer_id'))
    d = d[d.order_status == 'delivered'].dropna(subset=['order_approved_at', 'order_delivered_carrier_date',
                                                        'order_delivered_customer_date'])
    d = d[d.n_sellers == 1].copy()
    d['late_delivery'] = (d.order_delivered_customer_date > d.order_estimated_delivery_date).astype(int)
    d['late_handover'] = (d.order_delivered_carrier_date > d['limit']).astype(int)
    d['ts_anomaly'] = ((d.order_delivered_carrier_date < d.order_purchase_timestamp) |
                       (d.order_delivered_customer_date < d.order_delivered_carrier_date)).astype(int)  # chi danh dau, KHONG xoa
    return d.sort_values('order_approved_at').reset_index(drop=True)


def _hist_rate(d, key, label, prior, t_app, t_del, k=20):
    """Ty le tre LICH SU co lam muot, chi tinh tren cac don cung `key` da GIAO XONG truoc thoi diem duyet cua don hien tai."""
    out, cnt = np.full(len(d), prior), np.zeros(len(d))
    for _, idx in d.groupby(key).indices.items():
        ev = idx[np.argsort(t_del[idx])]
        cl = np.cumsum(label[ev])
        pos = np.searchsorted(t_del[ev], t_app[idx], side='right')      # so don da giao xong <= luc duyet
        n, l = pos, np.where(pos > 0, cl[np.maximum(pos - 1, 0)], 0)
        out[idx], cnt[idx] = (l + k * prior) / (n + k), n
    return out, cnt


def add_features(d, train_end='2018-05-01'):
    """Moi dac trung chi dung thong tin biet DUOC tai thoi diem order_approved_at.
    `train_end`: hang so tien nghiem (prior) cua ty le tre chi tinh tren cac don duyet TRUOC moc nay (khong nhin giai doan test)."""
    d = d.copy()
    sec = lambda s: s.values.astype('datetime64[s]').astype(np.int64)
    t_app, t_del = sec(d.order_approved_at), sec(d.order_delivered_customer_date)
    d['min_to_limit_h'] = (d['limit'] - d.order_approved_at).dt.total_seconds() / 3600
    d['sla_days'] = (d.order_estimated_delivery_date - d.order_approved_at).dt.total_seconds() / 86400
    d['approve_lag_h'] = (d.order_approved_at - d.order_purchase_timestamp).dt.total_seconds() / 3600
    d['freight_ratio'] = d.freight / d.price.clip(lower=1)
    d['same_state'] = (d.seller_state == d.customer_state).astype(int)
    d['cust_SP'] = (d.customer_state == 'SP').astype(int)
    d['sell_SP'] = (d.seller_state == 'SP').astype(int)
    d['hour'], d['dow'] = d.order_approved_at.dt.hour, d.order_approved_at.dt.dayofweek
    # tai trong 24h truoc (don da duoc duyet truoc do): toan he thong va rieng seller
    d['global_24h'] = np.arange(len(d)) - np.searchsorted(t_app, t_app - 86400, side='left')
    s24 = np.zeros(len(d))
    for _, idx in d.groupby('seller').indices.items():
        tt = t_app[idx]
        s24[idx] = np.arange(len(idx)) - np.searchsorted(tt, tt - 86400, side='left')
    d['seller_24h'] = s24
    # lich su seller / tuyen (knowledge-time safe)
    y_del, y_han = d.late_delivery.values, d.late_handover.values
    pre = (d.order_approved_at < train_end).values
    prior_del, prior_han = y_del[pre].mean(), y_han[pre].mean()
    d['lane'] = d.seller_state + '>' + d.customer_state
    d['seller_late_rate'], d['seller_hist_n'] = _hist_rate(d, 'seller', y_del, prior_del, t_app, t_del)
    d['seller_han_rate'], _ = _hist_rate(d, 'seller', y_han, prior_han, t_app, t_del)
    d['lane_late_rate'], _ = _hist_rate(d, 'lane', y_del, prior_del, t_app, t_del)
    return d


def select_cohort(d, start=COHORT_START, end=COHORT_END):
    """Cat cua so SAU KHI tinh lich su (de don dau cua so van co lich su). Bo 2016 (rat it don) va tu 2018-08 (right-censoring)."""
    return d[(d.order_approved_at >= start) & (d.order_approved_at < end)].reset_index(drop=True)


def fill_train_median(d, train_idx, cols=('weight', 'vol')):
    """Dien thieu bang trung vi cua TAP TRAIN (khong dung toan bo du lieu)."""
    d = d.copy()
    for c in cols:
        d[c] = d[c].fillna(d.loc[train_idx, c].median())
    return d


def time_split(d, test_start='2018-05-01', val_frac=0.2):
    tr = np.where(d.order_approved_at < test_start)[0]
    te = np.where(d.order_approved_at >= test_start)[0]
    m = int(len(tr) * (1 - val_frac))
    return tr[:m], tr[m:], te          # train_fit, validation, test (theo thu tu thoi gian)
```

# PHỤ LỤC B — Cây quyết định NumPy tham chiếu (AUC khớp sklearn trong 0.005; huấn luyện < 1 giây)

Thiếu `criterion=entropy`, `min_samples_split`, `max_features` — cần bổ sung ở Giai đoạn 3.

```python
"""Reference: Decision Tree (Gini) in pure NumPy with quantile-binned thresholds. Verified against sklearn on Olist."""
import numpy as np

class BinMapper:
    def __init__(self, n_bins=64): self.n_bins = n_bins
    def fit(self, X):
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges_ = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
        return self
    def transform(self, X):
        return np.stack([np.searchsorted(self.edges_[j], X[:, j], side='right') for j in range(X.shape[1])], 1).astype(np.int16)

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_leaf=200, class_weight=None, n_bins=64):
        self.max_depth, self.min_leaf, self.cw, self.n_bins = max_depth, min_samples_leaf, class_weight, n_bins
    def fit(self, X, y):
        self.bm_ = BinMapper(self.n_bins).fit(X); B = self.bm_.transform(X)
        w = np.ones(len(y))
        if self.cw == 'balanced':
            n1 = y.sum(); w = np.where(y == 1, len(y) / (2 * n1), len(y) / (2 * (len(y) - n1)))
        self.feat, self.thr, self.left, self.right, self.val, self.cnt = [], [], [], [], [], []
        self._grow(B, y.astype(float), w, np.arange(len(y)), 0)
        self.feat, self.thr = np.array(self.feat), np.array(self.thr)
        self.left, self.right, self.val = np.array(self.left), np.array(self.right), np.array(self.val)
        return self
    def _new(self, v, n):
        self.feat.append(-1); self.thr.append(-1); self.left.append(-1); self.right.append(-1); self.val.append(v); self.cnt.append(n)
        return len(self.feat) - 1
    def _grow(self, B, y, w, idx, depth):
        wy, ww = (w[idx] * y[idx]), w[idx]
        node = self._new(wy.sum() / ww.sum(), len(idx))
        if depth >= self.max_depth or len(idx) < 2 * self.min_leaf or wy.sum() == 0 or wy.sum() == ww.sum(): return node
        best = (0.0, -1, -1); tot_w, tot_p = ww.sum(), wy.sum()
        gini_parent = 2 * (tot_p / tot_w) * (1 - tot_p / tot_w)
        for j in range(B.shape[1]):
            bj = B[idx, j]; nb = bj.max() + 1
            hw = np.bincount(bj, ww, nb); hp = np.bincount(bj, wy, nb); hn = np.bincount(bj, minlength=nb)
            cw_, cp_, cn_ = np.cumsum(hw)[:-1], np.cumsum(hp)[:-1], np.cumsum(hn)[:-1]
            ok = (cn_ >= self.min_leaf) & (len(idx) - cn_ >= self.min_leaf) & (cw_ > 0) & (tot_w - cw_ > 0)
            if not ok.any(): continue
            pl = cp_ / np.where(cw_ > 0, cw_, 1); pr = (tot_p - cp_) / np.where(tot_w - cw_ > 0, tot_w - cw_, 1)
            gain = gini_parent - (cw_ * 2 * pl * (1 - pl) + (tot_w - cw_) * 2 * pr * (1 - pr)) / tot_w
            gain = np.where(ok, gain, -1); k = int(np.argmax(gain))
            if gain[k] > best[0]: best = (gain[k], j, k)
        if best[1] < 0 or best[0] <= 1e-7: return node
        _, j, k = best; mask = B[idx, j] <= k
        self.feat[node], self.thr[node] = j, k
        l = self._grow(B, y, w, idx[mask], depth + 1); r = self._grow(B, y, w, idx[~mask], depth + 1)
        self.left[node], self.right[node] = l, r
        return node
    def predict_proba(self, X):
        B = self.bm_.transform(X); node = np.zeros(len(B), dtype=int)
        while True:
            f = self.feat[node]; leaf = f < 0
            if leaf.all(): break
            go = np.where(leaf, 0, B[np.arange(len(B)), np.where(leaf, 0, f)] <= self.thr[node])
            node = np.where(leaf, node, np.where(go, self.left[node], self.right[node]))
        return self.val[node]
    def threshold_value(self, node):   # real-valued threshold for export (web / JSON)
        j, k = self.feat[node], self.thr[node]; e = self.bm_.edges_[j]
        return float(e[k]) if k < len(e) else float('inf')
```

# PHỤ LỤC C — `src/simulate.py` (đã sửa lỗi bắt đầu trước giờ đơn đến; kết quả Mục 7.4)

```python
"""simulate.py - Mo phong hang doi 1 trung tam xu ly don: K nhan vien, ca 08:00-17:00 Thu 2 - Thu 7.
GIA DINH (phai neu trong bao cao): thoi gian xu ly 1 don = SVC_BASE + SVC_PER_ITEM*(n_items-1) phut.
Thoi diem don den (order_approved_at) va han ban giao (shipping_limit_date) la du lieu THAT."""
import heapq
import numpy as np

SVC_BASE, SVC_PER_ITEM = 6.0, 3.0
DAY_START, DAY_END = 8 * 60, 17 * 60


def to_min(x):                       # datetime64 -> so phut ke tu epoch
    return x.astype('datetime64[m]').astype(np.int64)


def next_work(t):                    # day t toi phut lam viec ke tiep (bo Chu nhat; epoch 1970-01-01 la Thu 5)
    while True:
        dow, m = ((t // 1440) + 3) % 7, t % 1440         # Mon=0 ... Sun=6
        if dow == 6 or m >= DAY_END:
            t = (t // 1440 + 1) * 1440 + DAY_START; continue
        if m < DAY_START:
            return (t // 1440) * 1440 + DAY_START
        return t


def simulate(t_arr, t_due, svc, prio, K):
    """Tra ve thoi diem hoan tat (phut). prio nho hon = lam truoc. Khong preempt; don lam do duoc chuyen sang ca sau."""
    n, order = len(t_arr), np.argsort(t_arr, kind='stable')
    free = [int(t_arr.min())] * K; heapq.heapify(free)
    done, pend, i = np.full(n, -1, dtype=np.int64), [], 0
    while i < n or pend:
        tf = next_work(heapq.heappop(free))
        if not pend and i < n:
            tf = max(tf, next_work(int(t_arr[order[i]])))
        while i < n and t_arr[order[i]] <= tf:
            heapq.heappush(pend, (prio[order[i]], order[i])); i += 1
        _, j = heapq.heappop(pend)
        t, rem = max(tf, int(t_arr[j])), float(svc[j])      # khong bao gio bat dau truoc luc don den
        while rem > 0:
            t = next_work(t); use = min(rem, DAY_END - (t % 1440)); t += use; rem -= use
        done[j] = int(np.ceil(t)); heapq.heappush(free, done[j])
    return done


def policies(t_due, p_risk, delta_h=24.0):
    """FIFO = As-Is theo de cuong. EDD = baseline manh (han gan nhat truoc). AI_EDD = To-Be (han da tru di rui ro)."""
    n = len(t_due)
    return {'FIFO': np.arange(n, dtype=float),
            'EDD': t_due.astype(float),
            'AI_EDD': t_due - delta_h * 60.0 * p_risk}


def kpis(done, t_arr, t_due):
    lead_h = (done - t_arr) / 60.0
    return dict(late_rate=float((done > t_due).mean()), lead_mean_h=float(lead_h.mean()), lead_p95_h=float(np.percentile(lead_h, 95)))
```

# PHỤ LỤC D — Xuất cây sang JSON (đã kiểm chứng: dự đoán từ JSON khớp Python, sai số 0.0 trên 18,765 đơn)

```python
import json

def export_tree(tree, feature_names, t1, t2):
    """Xuat cay sang JSON cho web. Quy uoc: x[f] < thr  ->  di trai (thr la gia tri THAT, khong phai chi so bin)."""
    nodes = []
    for i in range(len(tree.feat)):
        leaf = tree.feat[i] < 0
        nodes.append({'f': int(tree.feat[i]), 'thr': None if leaf else tree.threshold_value(i),
                      'l': int(tree.left[i]), 'r': int(tree.right[i]), 'p': float(tree.val[i]), 'n': int(tree.cnt[i])})
    return {'feature_names': list(feature_names), 'nodes': nodes, 't1': float(t1), 't2': float(t2), 'rule': 'x[f] < thr -> left'}

def predict_json(model, x):
    """Tham chieu cho ban JavaScript: duyet cay tu goc."""
    nodes, i = model['nodes'], 0
    while nodes[i]['f'] >= 0:
        i = nodes[i]['l'] if x[nodes[i]['f']] < nodes[i]['thr'] else nodes[i]['r']
    return nodes[i]['p']
```

# PHỤ LỤC E — Suy luận trong trình duyệt (`web/js/tree.js`)

```javascript
// Duyet cay tu goc; tra ve diem rui ro va duong di (de hien thi giai thich). Quy uoc: x[f] < thr -> trai.
function predictTree(model, x) {
  const nodes = model.nodes, path = [];
  let i = 0;
  while (nodes[i].f >= 0) {
    const n = nodes[i], goLeft = x[n.f] < n.thr;
    path.push({ feature: model.feature_names[n.f], thr: n.thr, value: x[n.f], dir: goLeft ? 'left' : 'right', n: n.n });
    i = goLeft ? n.l : n.r;
  }
  return { p: nodes[i].p, support: nodes[i].n, path };
}
function priorityLevel(p, model) { return p >= model.t1 ? 'P1' : (p >= model.t2 ? 'P2' : 'P3'); }
if (typeof module !== 'undefined') module.exports = { predictTree, priorityLevel };
```

Cách gọi mô phỏng (tham chiếu, thuộc Phụ lục C):

```python
import numpy as np, data_prep as dp, simulate as sm
from model import DecisionTree      # từ Phụ lục B

raw = dp.load_raw('data/raw')
c = dp.select_cohort(dp.add_features(dp.build_order_table(raw)))
tr_fit, va, te = dp.time_split(c)
c = dp.fill_train_median(c, tr_fit)
X, y = c[dp.FEATURES].values, c.late_handover.values
tree = DecisionTree(max_depth=5, min_samples_leaf=200).fit(X[tr_fit], y[tr_fit])
p = tree.predict_proba(X[te])

w = c.iloc[te].reset_index(drop=True)
t_arr, t_due = sm.to_min(w.order_approved_at.values), sm.to_min(w['limit'].values)
svc = sm.SVC_BASE + sm.SVC_PER_ITEM * (w.n_items.values - 1)
for name, prio in sm.policies(t_due, p).items():
    done = sm.simulate(t_arr, t_due, svc, prio, K=3)
    print(name, sm.kpis(done, t_arr, t_due))
```
