# BẢNG SO SÁNH CÁC MÔ HÌNH (NO FREE LUNCH - BƯỚC 12)

Đo lường trên cùng tập dữ liệu phân chia theo chuỗi thời gian (Train: 55,808 | Val: 13,953 | Test: 18,765).

| STT | Mô hình | Vai trò | Val ROC-AUC | Val PR-AUC | Test ROC-AUC | Test PR-AUC | Test Brier |
|---|---|---|---|---|---|---|---|
| 1 | **Dummy** | Baseline sàn (đoán hằng số tỷ lệ train) | 0.5000 | 0.0750 | **0.5000** | **0.0747** | 0.0697 |
| 2 | **Rule_FIFO** | As-Is đề cương (đơn đến trước làm trước) | 0.5534 | 0.0880 | **0.4646** | **0.0671** | 0.3656 |
| 3 | **Rule_Slack** | Baseline kinh điển (slack ngắn nhất làm trước) | 0.4813 | 0.0714 | **0.4866** | **0.0722** | 23816.3746 |
| 4 | **Rule_Seller_History** | Baseline mạnh 1 đặc trưng (tỷ lệ trễ lịch sử seller) | 0.7728 | 0.1999 | **0.6712** | **0.1514** | 0.0680 |
| 5 | **Tree_Order_Only** | Cây không lịch sử (chỉ đặc trưng đơn hàng) | 0.5348 | 0.0823 | **0.5183** | **0.0795** | 0.0743 |
| 6 | **Logistic_Regression** | Baseline tuyến tính (chuẩn hóa Z-score, class_weight) | 0.7463 | 0.1787 | **0.6833** | **0.1580** | 0.2530 |
| 7 | **Decision_Tree_Core** | Mô hình lõi theo đề (depth 5, leaf 200, thuần NumPy) | 0.7538 | 0.1685 | **0.6666** | **0.1541** | 0.0727 |
| 8 | **Random_Forest** | So sánh phi tuyến phức tạp (50 cây, depth 8, leaf 50) | 0.7385 | 0.1715 | **0.6905** | **0.1794** | 0.0688 |


## Nhận định thực tế và bài học trung thực (Mục 4.9 & 11)

1. **Quy tắc lịch sử người bán (`seller_han_rate`) là tín hiệu mạnh nhất:** Đạt Test ROC-AUC ~0.671, ngang bằng với Decision Tree lõi (~0.667).
2. **Cây không có đặc trưng lịch sử (`Tree_Order_Only`):** Test ROC-AUC giảm xuống ~0.509 (gần như dự đoán ngẫu nhiên), chứng minh thông tin trễ bàn giao trong Olist nằm ở năng lực/hành vi của người bán chứ không phải kích thước/giá trị đơn hàng.
3. **As-Is FIFO & Slack:** ROC-AUC chỉ đạt ~0.465 và ~0.487 (dưới ngẫu nhiên).
4. **Lợi thế thực sự của Decision Tree:** Không phải độ chính xác vượt trội mà là tính **minh bạch**, khả năng gộp nhiều đặc trưng và phân nhánh điều kiện hỗ trợ Trưởng ca ra quyết định.
