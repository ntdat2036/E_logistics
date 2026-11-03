# BẢNG GIẢ ĐỊNH VÀ DỮ LIỆU MÔ PHỎNG (ASSUMPTIONS & SIMULATED DATA)

Tài liệu này công bố minh bạch toàn bộ các giả định kỹ thuật và dữ liệu mô phỏng được bổ sung cho đề tài CK10 (E-logistics), tuân thủ tuyệt đối quy định về liêm chính học thuật và tính tái lập thực nghiệm.

---

## 1. Giới hạn của Tập dữ liệu Olist Gốc

Dataset Olist Brazilian E-Commerce là dữ liệu sàn thương mại điện tử công khai, không thu thập trực tiếp từ hệ thống WMS (Warehouse Management System) nội bộ của một doanh nghiệp đơn lẻ. Do đó, tập dữ liệu gốc **thiếu các trường thông tin kho vận chuyên sâu**:
- Không có hạn sử dụng, số lô sản xuất (Batch/Lot).
- Không có số lượng tồn kho tức thời (Stock on hand) và tồn kho an toàn (Safety stock).
- Không có chi phí lưu kho, chi phí phạt trễ theo hợp đồng SLA.
- Không có số lượng công nhân thực tế tại từng trạm đóng gói.

---

## 2. Bảng Giả Định Bổ Sung (Dùng cho Lớp luật FEFO, Cảnh báo Kho & Mô phỏng)

| Thành phần thiếu | Cách bổ sung (Giả định có kiểm soát) | Công thức / Tham số | Ứng dụng |
|---|---|---|---|
| **Hạn sử dụng / Lô hàng (FEFO)** | Gán `sim_shelf_life_days` theo nhóm `product_category_name`. Thực phẩm (`alimentos`, `bebidas`), Dược mỹ phẩm (`perfumaria`, `beleza_saude`): 180–365 ngày; Thiết bị điện tử/gia dụng: không áp hạn. Ngày nhập `sim_received_date` sinh ngẫu nhiên trước ngày đặt hàng với seed cố định. | $\text{sim\_expiry\_date} = \text{sim\_received\_date} + \text{sim\_shelf\_life_days}$ | Lớp luật FEFO hậu xử lý, cảnh báo đơn cận date |
| **Tồn kho hiện tại** | Sinh `sim_stock_on_hand` cho từng SKU theo phân phối Poisson quanh nhu cầu trung bình 7 ngày + tồn kho an toàn. | $\text{sim\_stock\_on\_hand} \sim \text{Poisson}(\mu_{\text{daily}} \times 7) + SS$ | Bảng cảnh báo tồn kho trên Web |
| **Tồn kho an toàn ($SS$)** | Tính theo công thức quản trị tồn kho chuẩn tắc với mức phục vụ 95% ($z = 1.65$), thời gian giao hàng bổ sung $L = 3$ ngày. | $SS = z \cdot \sigma_{\text{nhu cầu ngày}} \cdot \sqrt{L}$ | Cảnh báo nguy cơ đứt gãy tồn kho (`sim_stockout_risk`) |
| **Nhân sự & Thời gian xử lý** | 1 trung tâm xử lý duy nhất; $K = 3$ nhân viên. Thời gian xử lý: 6 phút cơ bản + 3 phút cho mỗi món bổ sung. Không ngắt quãng, việc dở chuyển sang ca sau. | $\text{svc} = 6.0 + 3.0 \times (n_{\text{items}} - 1)$ (phút). $K=3$ được neo để FIFO ra tỷ lệ trễ ~7.2% khớp thực tế 7.5%. | Mô phỏng hàng đợi kho As-Is vs To-Be |
| **Giờ làm việc kho** | Ca hành chính 08:00 – 17:00 từ Thứ Hai đến Thứ Bảy; nghỉ Chủ nhật (540 phút làm việc / ngày). | Khung giờ $m \in [480, 1020]$, $\text{DOW} \neq 6$ | Mô phỏng hàng đợi kho |
| **Chi phí vận hành** | Chi phí phạt trễ hạn bàn giao giả định = 10% giá trị đơn hàng; Chi phí lưu kho = 0.05 R$/ngày/món. | $\text{Cost}_{\text{delay}} = 0.1 \times \text{Price}$; $\text{Cost}_{\text{holding}} = 0.05 \times \text{Days}$ | So sánh kinh tế giữa các chính sách trên Web |

---

## 3. Quy Tắc Kỹ Thuật Bắt Buộc Để Tránh Gây Nhầm Lẫn

1. **Định danh bằng tiền tố:** Toàn bộ các cột dữ liệu giả định bắt buộc mang tiền tố `sim_` (ví dụ: `sim_expiry_date`, `sim_stock_on_hand`, `sim_safety_stock`).
2. **Tách biệt vị trí lưu trữ:**
   - Dữ liệu gốc Olist nguyên vẹn 100% nằm tại `data/raw/`.
   - Toàn bộ dữ liệu giả lập được sinh bởi script riêng [`src/synth.py`](src/synth.py) với `SEED = 42` và lưu tại `data/simulated/`.
3. **Ranh giới bất khả xâm phạm với Mô hình Machine Learning:**
   - **Tuyệt đối KHÔNG đưa bất kỳ cột `sim_` nào vào mô hình học máy trong `src/data_prep.py` hay `src/model.py`.**
   - Mọi đặc trưng huấn luyện ($X$) và nhãn mục tiêu ($y$) của mô hình dự đoán trễ bàn giao đều là dữ liệu thật 100% của Olist tại thời điểm `order_approved_at`.
   - Dữ liệu `sim_` chỉ đóng vai trò đánh giá chính sách và hiển thị giao diện minh họa quy trình quản trị kho toàn diện.
4. **Chú thích trên giao diện Web & Báo cáo:** Mọi thông tin hiển thị có liên quan đến hạn sử dụng, tồn kho, chi phí đều đi kèm nhãn chú thích rõ ràng: *“Dữ liệu mô phỏng giả định — Dùng để kiểm thử chính sách”*.

