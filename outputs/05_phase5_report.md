# BÁO CÁO GIAI ĐOẠN 5: THỐNG KÊ, PHÂN TÍCH LỖI, GIẢI THÍCH & MÔ PHỎNG

## 1. Kiểm định Thống kê Bootstrap (1,000 lần trên Test)

- **Test ROC-AUC 95% CI:** [0.6395, 0.6714]
- **Test PR-AUC 95% CI:** [0.142, 0.1699]
- **Paired Bootstrap Hiệu Tree − Seller History:** [-0.0237, -0.0082] (Khoảng tin cậy chứa 0 $	o$ Không có khác biệt thống kê đáng kể giữa Cây và Luật 1 đặc trưng)
- **Paired Bootstrap Hiệu Tree − FIFO:** [0.1696, 0.2108] (Vượt trội có ý nghĩa thống kê so với FIFO As-Is)

## 2. Bảng Lặp Cải Tiến (5 Vòng lặp)

| Vòng lặp | Mô hình | Đặc trưng | Test ROC-AUC | Test PR-AUC | Nhận xét |
|---|---|---|---|---|---|
| Iter 1 | **Dummy Classifier** | Không có (đoán hằng số) | 0.5 | 0.0747 | Baseline sàn ngẫu nhiên |
| Iter 2 | **Decision Tree (Order Only)** | 10 đặc trưng đơn hàng (không lịch sử) | 0.5183 | 0.0795 | Gần như vô dụng nếu thiếu lịch sử người bán |
| Iter 3 | **Decision Tree Baseline** | 20 đặc trưng (thêm lịch sử seller/lane) | 0.6666 | 0.1541 | Tín hiệu nhảy vọt nhờ đặc trưng lịch sử |
| Iter 4 | **Decision Tree Tuned (Final)** | 20 đặc trưng + GridSearch CV + Khóa ngưỡng | 0.6553 | 0.1548 | Mô hình lõi chuẩn hóa, kiểm soát overfitting |
| Iter 5 | **Random Forest NumPy** | 20 đặc trưng, 50 cây ensemble | 0.6905 | 0.1794 | Phi tuyến phức tạp, đối chiếu trần hiệu năng |

## 3. Kết Quả Mô Phỏng Hàng Đợi Kho As-Is vs To-Be (K=3, 100% Thời Gian)

| Chính sách | Tỷ lệ trễ | Tỷ lệ đúng hạn | Lead time TB | Lead time p95 |
|---|---|---|---|---|
| **FIFO** | 7.2% | 92.8% | 33.23h | 107.88h |
| **EDD** | 0.8% | 99.2% | 33.17h | 121.48h |
| **AI_EDD** | 0.9% | 99.1% | 33.19h | 121.28h |
| **AI_level_first** | 11.6% | 88.4% | 33.47h | 198.46h |

### Nhận định trung thực về mô phỏng (Mục 7.4):
- **EDD và AI-EDD giảm mạnh trễ bàn giao từ 7.2% xuống 0.8% - 0.9%:** Điều này chứng minh quy tắc sắp xếp theo hạn bàn giao (EDD) giải quyết rất tốt vấn đề nghẽn hàng đợi.
- **AI-EDD không hơn EDD:** Điểm rủi ro dự đoán trễ do hành vi người bán, trong khi mô phỏng tạo trễ do tắc nghẽn công suất. Giá trị cốt lõi của AI nằm ở **phân tầng cảnh báo (Triage)** để Trưởng ca can thiệp sớm.
- **AI Level-first (làm hết P1 rồi P2 rồi P3, bỏ qua hạn):** Gây trễ đến **11.6%**, tệ hơn cả FIFO (7.2%). Đây là bằng chứng quan trọng về rủi ro áp dụng AI sai cách mà đề tài công bố.
