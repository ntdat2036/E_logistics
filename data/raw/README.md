# Dữ liệu gốc Olist Brazilian E-Commerce

## Nguồn dữ liệu
- **Kaggle:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Version 2).
- Bao gồm 5 tập tin CSV gốc:
  - `olist_orders_dataset.csv`
  - `olist_order_items_dataset.csv`
  - `olist_products_dataset.csv`
  - `olist_sellers_dataset.csv`
  - `olist_customers_dataset.csv`

## Lưu ý về Giấy phép & Bảo mật (Mục 2 - R1 & Mục 10)
- **Kiểm tra giấy phép:** Nhóm tự kiểm tra điều khoản và giấy phép sử dụng trên trang Kaggle trước khi phát hành/sử dụng thương mại (dữ liệu có điều khoản phi thương mại / nghiên cứu học thuật).
- **Tuyệt đối không commit dữ liệu gốc lên Git:** Các file CSV có dung lượng lớn và phải được quản lý cục bộ (đã cấu hình trong `.gitignore`).
- **Nguyên vẹn dữ liệu:** Không chỉnh sửa, không ghi đè, không xóa dòng trực tiếp trên các file CSV này. Mọi biến đổi dữ liệu phải thực hiện thông qua mã nguồn trong `src/data_prep.py` và được ghi log đầy đủ.

