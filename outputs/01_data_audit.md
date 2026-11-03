# BÁO CÁO KIỂM TOÁN DỮ LIỆU GỐC (DATA AUDIT - BƯỚC 1-6)

## 1. Tính toàn vẹn của 5 tập tin CSV gốc (Mã băm SHA256)

| Tên tập tin | SHA256 |
|---|---|
| `olist_customers_dataset.csv` | `983a422239e1712ded753b3bf9ecf47dc73f144d306029dcfa99e70a226883d2` |
| `olist_order_items_dataset.csv` | `0bc4d068c4fe38cbb01bd90e8746e3c613fe7b4baef75fab7b0e329701c3e279` |
| `olist_orders_dataset.csv` | `8df58ef3d2d7e9944010f7beecd9b75367f5588ec6e3c91cec19ae3345ef9ecf` |
| `olist_products_dataset.csv` | `3e6569628a17fbc75fd206ee357b59e20364b9afa90f5b6cd5b4d624c58aa9cc` |
| `olist_sellers_dataset.csv` | `1f643d2b950373b85735e7794b20986f528d7a000432e7c6f9bcbb44d0846a0e` |

## 2. Kích thước và Dữ liệu Khuyết thiếu (Missing Values)

| Bảng | Dòng × Cột | Cột khuyết thiếu (Null) |
|---|---|---|
| `orders` | 99,441 × 8 | `order_approved_at`: 160, `order_delivered_carrier_date`: 1783, `order_delivered_customer_date`: 2965 |
| `items` | 112,650 × 7 | Không có null |
| `products` | 32,951 × 9 | `product_category_name`: 610, `product_name_lenght`: 610, `product_description_lenght`: 610, `product_photos_qty`: 610, `product_weight_g`: 2, `product_length_cm`: 2, `product_height_cm`: 2, `product_width_cm`: 2 |
| `sellers` | 3,095 × 4 | Không có null |
| `customers` | 99,441 × 5 | Không có null |

## 3. Phân phối Trạng thái Đơn hàng (`order_status`)

| Trạng thái | Số lượng đơn |
|---|---|
| `delivered` | 96,478 |
| `shipped` | 1,107 |
| `canceled` | 625 |
| `unavailable` | 609 |
| `invoiced` | 314 |
| `processing` | 301 |
| `created` | 5 |
| `approved` | 2 |

## 4. Lỗi Logic Thời gian (Chỉ đánh dấu cờ `ts_anomaly`, KHÔNG xóa)

- Ngày giao vận tải trước ngày đặt (`carrier < purchase`): **166** đơn.
- Ngày khách nhận trước ngày giao vận tải (`customer < carrier`): **23** đơn.
- Số đơn bị gắn cờ `ts_anomaly` trong cohort: **117** đơn.

## 5. Phễu Lọc Dữ liệu (Cohort Filter)

- **Tổng số đơn ban đầu:** 99,441 đơn
- **Sau khi lọc (delivered, đủ 3 mốc thời gian, 1 seller):** 95,180 đơn
- **Cohort (2017-01-01 đến 2018-07-31):** 88,526 đơn
  - **Train fit:** 55,808 đơn (2017-01-05 to 2018-02-27)
  - **Validation:** 13,953 đơn (2018-02-27 to 2018-04-30)
  - **Test (khóa đến cuối):** 18,765 đơn (2018-05-01 to 2018-07-31)

## 6. Tỷ lệ Trễ theo Tháng & Bằng chứng Kiểm duyệt Phải (Right-Censoring)

| Tháng | Số đơn | Trễ bàn giao (late_handover) | Trễ giao khách (late_delivery) |
|---|---|---|---|
| 2016-09 | 1 | 100.0% (1) | 100.0% (1) |
| 2016-10 | 262 | 48.9% (128) | 1.1% (3) |
| 2016-12 | 1 | 0.0% (0) | 0.0% (0) |
| 2017-01 | 711 | 22.4% (159) | 3.1% (22) |
| 2017-02 | 1,626 | 17.4% (283) | 3.1% (50) |
| 2017-03 | 2,549 | 6.8% (172) | 5.6% (143) |
| 2017-04 | 2,260 | 7.1% (160) | 7.9% (178) |
| 2017-05 | 3,516 | 7.2% (255) | 3.7% (130) |
| 2017-06 | 3,117 | 7.8% (244) | 3.9% (121) |
| 2017-07 | 3,766 | 7.6% (287) | 3.5% (132) |
| 2017-08 | 4,157 | 7.0% (289) | 3.2% (135) |
| 2017-09 | 4,125 | 7.2% (296) | 5.2% (216) |
| 2017-10 | 4,378 | 8.0% (349) | 5.3% (233) |
| 2017-11 | 7,055 | 12.9% (913) | 14.3% (1,006) |
| 2017-12 | 5,616 | 11.1% (621) | 8.9% (501) |
| 2018-01 | 6,890 | 10.3% (707) | 6.7% (458) |
| 2018-02 | 6,485 | 12.3% (797) | 15.5% (1,007) |
| 2018-03 | 6,996 | 8.7% (606) | 21.8% (1,525) |
| 2018-04 | 6,514 | 6.1% (396) | 5.6% (362) |
| 2018-05 | 6,827 | 6.6% (450) | 8.3% (566) |
| 2018-06 | 5,989 | 6.9% (413) | 1.4% (83) |
| 2018-07 | 5,949 | 9.1% (539) | 4.2% (248) |
| 2018-08 | 6,390 | 9.2% (588) | 10.8% (687) |
