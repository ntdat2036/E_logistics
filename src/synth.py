"""synth.py - Sinh dữ liệu mô phỏng giả định cho các trường thông tin Olist còn thiếu (Mục 4.7 & ASSUMPTIONS.md).
TUÂN THỦ TUYỆT ĐỐI MỤC 2 (R4) VÀ MỤC 10 (ĐIỀU CẤM 7):
- Sử dụng SEED = 42 cố định để đảm bảo tính tái lập (R7).
- Mọi cột giả lập bắt buộc mang tiền tố 'sim_'.
- Lưu tách biệt tại data/simulated/, tuyệt đối không ghi đè vào data/raw/.
- Dữ liệu này chỉ phục vụ lớp luật FEFO, bảng cảnh báo tồn kho và so sánh chi phí trên Web.
- TUYỆT ĐỐI KHÔNG đưa vào mô hình học máy trong src/data_prep.py hay src/model.py.
"""
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd

SEED = 42

# Phân loại danh mục hàng cận date (perishable)
PERISHABLE_CATEGORIES = {
    'alimentos': (180, 270),
    'alimentos_bebidas': (180, 300),
    'bebidas': (180, 365),
    'perfumaria': (270, 365),
    'beleza_saude': (270, 365),
    'bebes': (300, 365),
    'pet_shop': (300, 365),
}


def generate_simulated_data(raw_dir='data/raw', out_dir='data/simulated', seed=SEED):
    """Sinh toàn bộ các trường giả lập cho SKU và Đơn hàng với SEED cố định."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Đọc dữ liệu gốc (chỉ đọc)
    products = pd.read_csv(os.path.join(raw_dir, 'olist_products_dataset.csv'))
    items = pd.read_csv(os.path.join(raw_dir, 'olist_order_items_dataset.csv'))
    orders = pd.read_csv(
        os.path.join(raw_dir, 'olist_orders_dataset.csv'),
        parse_dates=['order_purchase_timestamp', 'order_approved_at'],
    )

    # --------------------------------------------------------------------------
    # 1. SINH DỮ LIỆU TỒN KHO & HẠN DÙNG CẤP SKU (sku_inventory_simulated.csv)
    # --------------------------------------------------------------------------
    print("1. Đang tính toán và sinh dữ liệu giả định cấp SKU...")

    # Nhu cầu bán hàng thật theo ngày của từng sản phẩm
    items_merged = items.merge(orders[['order_id', 'order_approved_at']], on='order_id', how='inner')
    items_merged['date'] = items_merged['order_approved_at'].dt.date

    # Tính độ lệch chuẩn và trung bình nhu cầu theo ngày
    daily_sales = items_merged.groupby(['product_id', 'date']).size().reset_index(name='qty')
    demand_stats = daily_sales.groupby('product_id').agg(
        sim_daily_demand_mean=('qty', 'mean'),
        sim_daily_demand_std=('qty', 'std'),
    ).reset_index()

    # Điền giá trị mặc định cho sản phẩm bán ít
    demand_stats['sim_daily_demand_std'] = demand_stats['sim_daily_demand_std'].fillna(0.5)

    sku_df = products[['product_id', 'product_category_name']].copy()
    sku_df = sku_df.merge(demand_stats, on='product_id', how='left')
    sku_df['sim_daily_demand_mean'] = sku_df['sim_daily_demand_mean'].fillna(0.2)
    sku_df['sim_daily_demand_std'] = sku_df['sim_daily_demand_std'].fillna(0.5)

    # Gán hạn sử dụng (shelf_life_days) theo danh mục
    shelf_lives = []
    is_perish = []
    for cat in sku_df['product_category_name']:
        if cat in PERISHABLE_CATEGORIES:
            low, high = PERISHABLE_CATEGORIES[cat]
            shelf_lives.append(int(rng.integers(low, high + 1)))
            is_perish.append(1)
        else:
            shelf_lives.append(0)  # Không áp hạn
            is_perish.append(0)

    sku_df['sim_is_perishable'] = is_perish
    sku_df['sim_shelf_life_days'] = shelf_lives

    # Tồn kho an toàn SS = z * sigma * sqrt(L), với z=1.65 (95% service), L=3 ngày
    z = 1.65
    lead_time = 3.0
    ss = np.ceil(z * sku_df['sim_daily_demand_std'].values * np.sqrt(lead_time)).astype(int)
    sku_df['sim_safety_stock'] = np.maximum(1, ss)

    # Tồn kho hiện tại SOH ~ Poisson(mu * 7 ngày) + SS
    expected_soh = sku_df['sim_daily_demand_mean'].values * 7.0
    soh = rng.poisson(lam=expected_soh) + sku_df['sim_safety_stock'].values
    sku_df['sim_stock_on_hand'] = soh

    # Cờ nguy cơ đứt gãy tồn kho
    sku_df['sim_stockout_risk_flag'] = (
        sku_df['sim_stock_on_hand'] < sku_df['sim_safety_stock']
    ).astype(int)

    # Chi phí lưu kho giả định (R$/ngày/món)
    sku_df['sim_holding_cost_daily_brl'] = 0.05

    sku_path = os.path.join(out_dir, 'sku_inventory_simulated.csv')
    sku_df.to_csv(sku_path, index=False, encoding='utf-8')
    print(f"  -> Đã lưu {len(sku_df):,} dòng vào {sku_path}")

    # --------------------------------------------------------------------------
    # 2. SINH DỮ LIỆU ĐẶC TRƯNG MÔ PHỎNG CẤP ĐƠN HÀNG (orders_simulated_features.csv)
    # --------------------------------------------------------------------------
    print("2. Đang sinh dữ liệu giả định cấp Đơn hàng (FEFO & Chi phí)...")

    # Map thông tin SKU vào từng item
    order_sku_info = items[['order_id', 'product_id', 'price']].merge(
        sku_df[['product_id', 'sim_is_perishable', 'sim_shelf_life_days']],
        on='product_id',
        how='left',
    )
    order_sku_info = order_sku_info.merge(
        orders[['order_id', 'order_approved_at']],
        on='order_id',
        how='inner',
    )

    # Giả lập ngày nhập kho sim_received_date
    # Trong kho thực tế: ~95% hàng mới nhập (10-60 ngày), ~5% lô hàng cũ cận date (còn <= 3 ngày hạn)
    n_items = len(order_sku_info)
    is_near_expiry = (rng.random(n_items) < 0.05) & (order_sku_info['sim_is_perishable'] == 1)

    shelf_arr = order_sku_info['sim_shelf_life_days'].values
    lag_days = np.where(
        is_near_expiry,
        np.maximum(1, shelf_arr - rng.integers(0, 5, size=n_items)),
        rng.integers(10, 61, size=n_items),
    )
    received_dates = order_sku_info['order_approved_at'] - pd.to_timedelta(lag_days, unit='D')
    order_sku_info['sim_received_date'] = received_dates

    # Tính ngày hết hạn sim_expiry_date
    expiry_dates = []
    days_to_exp = []
    for _, row in order_sku_info.iterrows():
        if row['sim_is_perishable'] == 1 and row['sim_shelf_life_days'] > 0:
            exp = row['sim_received_date'] + pd.to_timedelta(row['sim_shelf_life_days'], unit='D')
            diff = (exp - row['order_approved_at']).total_seconds() / 86400.0
            expiry_dates.append(exp)
            days_to_exp.append(max(0.0, diff))
        else:
            expiry_dates.append(pd.NaT)
            days_to_exp.append(9999.0)

    order_sku_info['sim_expiry_date'] = expiry_dates
    order_sku_info['sim_days_to_expiry'] = days_to_exp

    # Tổng hợp lên cấp đơn hàng
    order_sim_summary = order_sku_info.groupby('order_id').agg(
        sim_has_perishable_item=('sim_is_perishable', 'max'),
        sim_min_days_to_expiry=('sim_days_to_expiry', 'min'),
        sim_order_total_price=('price', 'sum'),
    ).reset_index()

    # Cờ quy tắc FEFO (nếu còn hạn <= 3 ngày: gắn cờ để nâng ưu tiên và kiểm cảm quan theo Mục 4.7)
    order_sim_summary['sim_fefo_priority_flag'] = (
        (order_sim_summary['sim_has_perishable_item'] == 1)
        & (order_sim_summary['sim_min_days_to_expiry'] <= 3.0)
    ).astype(int)

    # Chi phí phạt trễ giả định (10% giá trị đơn hàng)
    order_sim_summary['sim_delay_penalty_cost_brl'] = (
        0.10 * order_sim_summary['sim_order_total_price']
    ).round(2)

    # Chi phí lưu kho ước tính giả định
    order_sim_summary['sim_estimated_holding_cost_brl'] = 0.50

    orders_path = os.path.join(out_dir, 'orders_simulated_features.csv')
    order_sim_summary.to_csv(orders_path, index=False, encoding='utf-8')
    print(f"  -> Đã lưu {len(order_sim_summary):,} dòng vào {orders_path}")

    # Tóm tắt
    fefo_flagged = int(order_sim_summary['sim_fefo_priority_flag'].sum())
    perish_orders = int(order_sim_summary['sim_has_perishable_item'].sum())
    print(f"Tổng số đơn có hàng cận date: {perish_orders:,} đơn.")
    print(f"Tổng số đơn kích hoạt cờ FEFO (hạn còn <= 3 ngày): {fefo_flagged:,} đơn.")
    print("=== Hoàn tất tạo dữ liệu giả định với SEED = 42 ===")

    return sku_df, order_sim_summary


if __name__ == '__main__':
    generate_simulated_data()

