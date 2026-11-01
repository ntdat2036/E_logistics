"""data_prep.py - Dựng bảng order-level từ 5 file Olist GỐC. Không sửa file gốc.
Tuân thủ tuyệt đối Mục 2 (R1, R2, R3) và Phụ lục A của HUONG_DAN.md.
"""
import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import hashlib
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TS_COLS = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date',
]
COHORT_START, COHORT_END = '2017-01-01', '2018-08-01'  # [start, end)
FEATURES = [
    'min_to_limit_h',
    'sla_days',
    'approve_lag_h',
    'n_items',
    'n_products',
    'price',
    'freight',
    'freight_ratio',
    'weight',
    'vol',
    'same_state',
    'cust_SP',
    'sell_SP',
    'hour',
    'dow',
    'global_24h',
    'seller_24h',
    'seller_han_rate',
    'seller_late_rate',
    'lane_late_rate',
]

RAW_FILES = [
    'olist_customers_dataset.csv',
    'olist_orders_dataset.csv',
    'olist_order_items_dataset.csv',
    'olist_products_dataset.csv',
    'olist_sellers_dataset.csv',
]


def compute_raw_sha256(raw_dir='data/raw'):
    """Tính mã băm SHA256 của 5 file gốc để kiểm tra tính toàn vẹn (R1)."""
    hashes = {}
    for fname in sorted(RAW_FILES):
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            continue
        h = hashlib.sha256()
        with open(fpath, 'rb') as fp:
            while chunk := fp.read(8192 * 1024):
                h.update(chunk)
        hashes[fname] = h.hexdigest()
    return hashes


def load_raw(raw_dir='data/raw'):
    """Đọc 5 file dữ liệu Olist gốc từ data/raw/."""
    r = lambda n, **kw: pd.read_csv(os.path.join(raw_dir, f'olist_{n}_dataset.csv'), **kw)
    return dict(
        orders=r('orders', parse_dates=TS_COLS),
        items=r('order_items', parse_dates=['shipping_limit_date']),
        products=r('products'),
        sellers=r('sellers'),
        customers=r('customers'),
    )


def build_order_table(raw):
    """1 dòng / đơn. Chỉ giữ: delivered + 1 seller + đủ 3 mốc thời gian. Gán 2 nhãn thật."""
    o, it, p = raw['orders'], raw['items'].copy(), raw['products']
    it = it.merge(
        p[['product_id', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']],
        on='product_id',
        how='left',
    )
    it['vol'] = it.product_length_cm * it.product_height_cm * it.product_width_cm
    g = it.groupby('order_id').agg(
        n_items=('order_item_id', 'count'),
        n_sellers=('seller_id', 'nunique'),
        n_products=('product_id', 'nunique'),
        price=('price', 'sum'),
        freight=('freight_value', 'sum'),
        weight=('product_weight_g', 'sum'),
        vol=('vol', 'sum'),
        limit=('shipping_limit_date', 'min'),
        seller=('seller_id', 'first'),
    ).reset_index()

    d = (
        o.merge(g, on='order_id')
        .merge(raw['sellers'][['seller_id', 'seller_state']], left_on='seller', right_on='seller_id')
        .merge(raw['customers'][['customer_id', 'customer_state']], on='customer_id')
    )
    d = d[d.order_status == 'delivered'].dropna(
        subset=['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date']
    )
    d = d[d.n_sellers == 1].copy()

    # Nhãn thật
    d['late_delivery'] = (d.order_delivered_customer_date > d.order_estimated_delivery_date).astype(int)
    d['late_handover'] = (d.order_delivered_carrier_date > d['limit']).astype(int)

    # Lỗi logic thời gian: chỉ đánh dấu, KHÔNG xóa (Mục 3)
    d['ts_anomaly'] = (
        (d.order_delivered_carrier_date < d.order_purchase_timestamp)
        | (d.order_delivered_customer_date < d.order_delivered_carrier_date)
    ).astype(int)

    return d.sort_values('order_approved_at').reset_index(drop=True)


def _hist_rate(d, key, label, prior, t_app, t_del, k=20):
    """Tỷ lệ trễ LỊCH SỬ có làm mượt, chỉ tính trên các đơn cùng `key` đã GIAO XONG trước thời điểm duyệt của đơn hiện tại."""
    out, cnt = np.full(len(d), prior, dtype=float), np.zeros(len(d), dtype=int)
    for _, idx in d.groupby(key).indices.items():
        ev = idx[np.argsort(t_del[idx])]
        cl = np.cumsum(label[ev])
        pos = np.searchsorted(t_del[ev], t_app[idx], side='right')  # số đơn đã giao xong <= lúc duyệt
        n = pos
        l = np.where(pos > 0, cl[np.maximum(pos - 1, 0)], 0)
        out[idx] = (l + k * prior) / (n + k)
        cnt[idx] = n
    return out, cnt


def add_features(d, train_end='2018-05-01'):
    """Mỗi đặc trưng chỉ dùng thông tin biết ĐƯỢC tại thời điểm order_approved_at.
    `train_end`: hằng số tiên nghiệm (prior) của tỷ lệ trễ chỉ tính trên các đơn duyệt TRƯỚC mốc này (không nhìn giai đoạn test).
    """
    d = d.copy()
    sec = lambda s: s.values.astype('datetime64[s]').astype(np.int64)
    t_app, t_del = sec(d.order_approved_at), sec(d.order_delivered_customer_date)

    d['min_to_limit_h'] = (d['limit'] - d.order_approved_at).dt.total_seconds() / 3600.0
    d['sla_days'] = (d.order_estimated_delivery_date - d.order_approved_at).dt.total_seconds() / 86400.0
    d['approve_lag_h'] = (d.order_approved_at - d.order_purchase_timestamp).dt.total_seconds() / 3600.0
    d['freight_ratio'] = d.freight / d.price.clip(lower=1.0)
    d['same_state'] = (d.seller_state == d.customer_state).astype(int)
    d['cust_SP'] = (d.customer_state == 'SP').astype(int)
    d['sell_SP'] = (d.seller_state == 'SP').astype(int)
    d['hour'] = d.order_approved_at.dt.hour
    d['dow'] = d.order_approved_at.dt.dayofweek

    # Tải trong 24h trước (đơn đã được duyệt trước đó): toàn hệ thống và riêng seller
    d['global_24h'] = np.arange(len(d)) - np.searchsorted(t_app, t_app - 86400, side='left')
    s24 = np.zeros(len(d), dtype=int)
    for _, idx in d.groupby('seller').indices.items():
        tt = t_app[idx]
        s24[idx] = np.arange(len(idx)) - np.searchsorted(tt, tt - 86400, side='left')
    d['seller_24h'] = s24

    # Lịch sử seller / tuyến (knowledge-time safe)
    y_del, y_han = d.late_delivery.values, d.late_handover.values
    pre = (d.order_approved_at < train_end).values
    prior_del, prior_han = float(y_del[pre].mean()), float(y_han[pre].mean())

    d['lane'] = d.seller_state + '>' + d.customer_state
    d['seller_late_rate'], d['seller_hist_n'] = _hist_rate(d, 'seller', y_del, prior_del, t_app, t_del)
    d['seller_han_rate'], _ = _hist_rate(d, 'seller', y_han, prior_han, t_app, t_del)
    d['lane_late_rate'], _ = _hist_rate(d, 'lane', y_del, prior_del, t_app, t_del)

    return d


def select_cohort(d, start=COHORT_START, end=COHORT_END):
    """Cắt cửa sổ SAU KHI tính lịch sử (để đơn đầu cửa sổ vẫn có lịch sử).
    Bỏ 2016 (rất ít đơn) và từ 2018-08 (right-censoring).
    """
    return d[(d.order_approved_at >= start) & (d.order_approved_at < end)].reset_index(drop=True)


def fill_train_median(d, train_idx, cols=('weight', 'vol')):
    """Điền thiếu bằng trung vị của TẬP TRAIN (không dùng toàn bộ dữ liệu)."""
    d = d.copy()
    for c in cols:
        median_val = d.loc[train_idx, c].median()
        d[c] = d[c].fillna(median_val)
    return d


def time_split(d, test_start='2018-05-01', val_frac=0.2):
    """Chia dữ liệu theo chuỗi thời gian: train_fit / val / test."""
    tr = np.where(d.order_approved_at < test_start)[0]
    te = np.where(d.order_approved_at >= test_start)[0]
    m = int(len(tr) * (1 - val_frac))
    return tr[:m], tr[m:], te  # train_fit, validation, test


def audit_raw(raw_dir='data/raw', out_dir='outputs'):
    """Kiểm toán dữ liệu gốc (Bước 1-6) và lưu báo cáo vào outputs/."""
    os.makedirs(out_dir, exist_ok=True)
    raw = load_raw(raw_dir)
    sha256_map = compute_raw_sha256(raw_dir)

    # 1. Shapes và Nulls
    tables_info = {}
    for name, df in raw.items():
        null_counts = {col: int(cnt) for col, cnt in df.isnull().sum().items() if cnt > 0}
        tables_info[name] = {
            'shape': list(df.shape),
            'rows': int(df.shape[0]),
            'cols': int(df.shape[1]),
            'null_cols': null_counts,
        }

    # 2. Phân phối order_status
    orders = raw['orders']
    status_counts = {str(k): int(v) for k, v in orders.order_status.value_counts().items()}

    # 3. Lỗi logic thời gian trên toàn bộ orders
    anom_carrier_before_purchase = int(
        (orders.order_delivered_carrier_date < orders.order_purchase_timestamp).sum()
    )
    anom_customer_before_carrier = int(
        (orders.order_delivered_customer_date < orders.order_delivered_carrier_date).sum()
    )

    # 4. Phễu lọc
    d_order = build_order_table(raw)
    d_feat = add_features(d_order)
    d_cohort = select_cohort(d_feat)
    tr_fit, val_idx, test_idx = time_split(d_cohort)

    cohort_anom_count = int(d_cohort['ts_anomaly'].sum())

    funnel = {
        '0_raw_orders': int(len(orders)),
        '1_delivered_with_timestamps_1seller': int(len(d_order)),
        '2_cohort_20170101_to_20180801': int(len(d_cohort)),
        'splits': {
            'train_fit': int(len(tr_fit)),
            'validation': int(len(val_idx)),
            'test': int(len(test_idx)),
        },
    }

    # 5. Tỷ lệ nhãn theo tháng trên toàn bộ dữ liệu đơn đã giao
    d_order['month_str'] = d_order.order_approved_at.dt.to_period('M').astype(str)
    monthly_stats = []
    for month, grp in d_order.groupby('month_str'):
        monthly_stats.append({
            'month': month,
            'orders_count': int(len(grp)),
            'late_handover_count': int(grp.late_handover.sum()),
            'late_handover_rate': round(float(grp.late_handover.mean()), 4),
            'late_delivery_count': int(grp.late_delivery.sum()),
            'late_delivery_rate': round(float(grp.late_delivery.mean()), 4),
        })

    # 6. Khoảng thời gian
    time_ranges = {
        'all_orders_min_approved': str(orders.order_approved_at.min()),
        'all_orders_max_approved': str(orders.order_approved_at.max()),
        'cohort_min_approved': str(d_cohort.order_approved_at.min()),
        'cohort_max_approved': str(d_cohort.order_approved_at.max()),
        'train_fit_range': f"{d_cohort.iloc[tr_fit[0]].order_approved_at.strftime('%Y-%m-%d')} to {d_cohort.iloc[tr_fit[-1]].order_approved_at.strftime('%Y-%m-%d')}",
        'validation_range': f"{d_cohort.iloc[val_idx[0]].order_approved_at.strftime('%Y-%m-%d')} to {d_cohort.iloc[val_idx[-1]].order_approved_at.strftime('%Y-%m-%d')}",
        'test_range': f"{d_cohort.iloc[test_idx[0]].order_approved_at.strftime('%Y-%m-%d')} to {d_cohort.iloc[test_idx[-1]].order_approved_at.strftime('%Y-%m-%d')}",
    }

    audit_data = {
        'sha256': sha256_map,
        'tables_info': tables_info,
        'order_status': status_counts,
        'logic_anomalies': {
            'carrier_before_purchase': anom_carrier_before_purchase,
            'customer_before_carrier': anom_customer_before_carrier,
            'cohort_ts_anomaly': cohort_anom_count,
        },
        'funnel': funnel,
        'monthly_stats': monthly_stats,
        'time_ranges': time_ranges,
    }

    # Lưu JSON
    json_path = os.path.join(out_dir, '01_data_audit.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)

    # Lưu Markdown
    md_path = os.path.join(out_dir, '01_data_audit.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# BÁO CÁO KIỂM TOÁN DỮ LIỆU GỐC (DATA AUDIT - BƯỚC 1-6)\n\n")
        f.write("## 1. Tính toàn vẹn của 5 tập tin CSV gốc (Mã băm SHA256)\n\n")
        f.write("| Tên tập tin | SHA256 |\n|---|---|\n")
        for fname, hval in sha256_map.items():
            f.write(f"| `{fname}` | `{hval}` |\n")

        f.write("\n## 2. Kích thước và Dữ liệu Khuyết thiếu (Missing Values)\n\n")
        f.write("| Bảng | Dòng × Cột | Cột khuyết thiếu (Null) |\n|---|---|---|\n")
        for name, info in tables_info.items():
            null_str = ", ".join([f"`{c}`: {n}" for c, n in info['null_cols'].items()]) if info['null_cols'] else "Không có null"
            f.write(f"| `{name}` | {info['rows']:,} × {info['cols']} | {null_str} |\n")

        f.write("\n## 3. Phân phối Trạng thái Đơn hàng (`order_status`)\n\n")
        f.write("| Trạng thái | Số lượng đơn |\n|---|---|\n")
        for st, c in status_counts.items():
            f.write(f"| `{st}` | {c:,} |\n")

        f.write("\n## 4. Lỗi Logic Thời gian (Chỉ đánh dấu cờ `ts_anomaly`, KHÔNG xóa)\n\n")
        f.write(f"- Ngày giao vận tải trước ngày đặt (`carrier < purchase`): **{anom_carrier_before_purchase}** đơn.\n")
        f.write(f"- Ngày khách nhận trước ngày giao vận tải (`customer < carrier`): **{anom_customer_before_carrier}** đơn.\n")
        f.write(f"- Số đơn bị gắn cờ `ts_anomaly` trong cohort: **{cohort_anom_count}** đơn.\n")

        f.write("\n## 5. Phễu Lọc Dữ liệu (Khớp hoàn toàn với Mục 3)\n\n")
        f.write(f"- **Tổng số đơn ban đầu:** {funnel['0_raw_orders']:,} đơn\n")
        f.write(f"- **Sau khi lọc (delivered, đủ 3 mốc thời gian, 1 seller):** {funnel['1_delivered_with_timestamps_1seller']:,} đơn\n")
        f.write(f"- **Cohort (2017-01-01 đến 2018-07-31):** {funnel['2_cohort_20170101_to_20180801']:,} đơn\n")
        f.write(f"  - **Train fit:** {funnel['splits']['train_fit']:,} đơn ({time_ranges['train_fit_range']})\n")
        f.write(f"  - **Validation:** {funnel['splits']['validation']:,} đơn ({time_ranges['validation_range']})\n")
        f.write(f"  - **Test (khóa đến cuối):** {funnel['splits']['test']:,} đơn ({time_ranges['test_range']})\n")

        f.write("\n## 6. Tỷ lệ Trễ theo Tháng & Bằng chứng Kiểm duyệt Phải (Right-Censoring)\n\n")
        f.write("| Tháng | Số đơn | Trễ bàn giao (late_handover) | Trễ giao khách (late_delivery) |\n|---|---|---|---|\n")
        for m in monthly_stats:
            f.write(f"| {m['month']} | {m['orders_count']:,} | {m['late_handover_rate']:.1%} ({m['late_handover_count']:,}) | {m['late_delivery_rate']:.1%} ({m['late_delivery_count']:,}) |\n")

    return audit_data, d_order, d_cohort


def generate_eda_plots(d_order, d_cohort, out_dir='outputs'):
    """Vẽ 4 biểu đồ EDA bắt buộc theo Giai đoạn 1."""
    os.makedirs(out_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. outputs/fig_orders_monthly.png: Số lượng đơn theo tháng
    d_order['month_str'] = d_order.order_approved_at.dt.to_period('M').astype(str)
    monthly_counts = d_order.groupby('month_str').size()

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(monthly_counts.index, monthly_counts.values, color='#3498db', edgecolor='#2980b9')
    # Tô màu cohort vs ngoài cohort
    for i, month in enumerate(monthly_counts.index):
        if month < '2017-01' or month >= '2018-08':
            bars[i].set_color('#bdc3c7')
    ax.set_title('Số lượng đơn hàng được duyệt theo tháng (Xám: Ngoài Cohort, Xanh: Trong Cohort)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Tháng phê duyệt (order_approved_at)', fontsize=11)
    ax.set_ylabel('Số lượng đơn hàng', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_orders_monthly.png'), dpi=150)
    plt.close(fig)

    # 2. outputs/fig_delay_shock.png: Tỷ lệ trễ bàn giao và trễ giao khách theo tháng
    monthly_rates = d_order.groupby('month_str').agg(
        han_rate=('late_handover', 'mean'),
        del_rate=('late_delivery', 'mean'),
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly_rates.index, monthly_rates['han_rate'], marker='o', linewidth=2, color='#e67e22', label='Trễ bàn giao (late_handover)')
    ax.plot(monthly_rates.index, monthly_rates['del_rate'], marker='s', linewidth=2, color='#e74c3c', label='Trễ giao khách (late_delivery)')
    ax.axvspan('2018-08', monthly_rates.index[-1], color='gray', alpha=0.2, label='Kiểm duyệt phải (Right-censoring)')
    ax.axvspan('2017-11', '2017-11', color='yellow', alpha=0.3, label='Cú sốc Black Friday (11/2017)')
    ax.axvspan('2018-02', '2018-03', color='orange', alpha=0.2, label='Cú sốc đình công/vận tải (02-03/2018)')
    ax.set_title('Tỷ lệ trễ theo tháng: Thể hiện các cú sốc thời gian và kiểm duyệt phải (08/2018)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Tháng', fontsize=11)
    ax.set_ylabel('Tỷ lệ trễ', fontsize=11)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    plt.xticks(rotation=45, ha='right')
    ax.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_delay_shock.png'), dpi=150)
    plt.close(fig)

    # 3. outputs/fig_sla_dist.png: Phân phối min_to_limit_h và sla_days
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # min_to_limit_h
    axes[0].hist(d_cohort['min_to_limit_h'].clip(-48, 240), bins=60, color='#2ecc71', edgecolor='black', alpha=0.7)
    axes[0].set_title('Phân phối thời gian còn lại đến hạn bàn giao (min_to_limit_h)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Giờ (clip [-48, 240])', fontsize=10)
    axes[0].set_ylabel('Số đơn hàng', fontsize=10)
    axes[0].axvline(0, color='red', linestyle='--', label='Hết hạn lúc duyệt')
    axes[0].legend()

    # sla_days
    axes[1].hist(d_cohort['sla_days'].clip(0, 60), bins=60, color='#9b59b6', edgecolor='black', alpha=0.7)
    axes[1].set_title('Phân phối thời gian cam kết giao hàng (sla_days)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Số ngày (clip [0, 60])', fontsize=10)
    axes[1].set_ylabel('Số đơn hàng', fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_sla_dist.png'), dpi=150)
    plt.close(fig)

    # 4. outputs/fig_delay_by_sla.png: Tỷ lệ trễ theo nhóm SLA
    sla_bins = [-100, 7, 14, 21, 28, 35, 1000]
    sla_labels = ['<=7 ngày', '8-14 ngày', '15-21 ngày', '22-28 ngày', '29-35 ngày', '>35 ngày']
    d_cohort_copy = d_cohort.copy()
    d_cohort_copy['sla_bucket'] = pd.cut(d_cohort_copy['sla_days'], bins=sla_bins, labels=sla_labels)

    sla_rates = d_cohort_copy.groupby('sla_bucket', observed=True).agg(
        han_rate=('late_handover', 'mean'),
        del_rate=('late_delivery', 'mean'),
        count=('order_id', 'count')
    )

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(sla_labels))
    w = 0.35
    ax1.bar(x - w / 2, sla_rates['han_rate'], width=w, color='#e67e22', label='Tỷ lệ trễ bàn giao')
    ax1.bar(x + w / 2, sla_rates['del_rate'], width=w, color='#e74c3c', label='Tỷ lệ trễ giao khách')
    ax1.set_ylabel('Tỷ lệ trễ', fontsize=11)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax1.set_xticks(x)
    ax1.set_xticklabels(sla_labels)
    ax1.set_title('Tỷ lệ trễ phân theo nhóm thời hạn cam kết (SLA)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig_delay_by_sla.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    print("=== Đang thực hiện Kiểm toán Dữ liệu (Data Audit) ===")
    audit_data, d_order, d_cohort = audit_raw('data/raw', 'outputs')
    print("-> Đã xuất outputs/01_data_audit.json và outputs/01_data_audit.md")
    print("=== Đang tạo các biểu đồ EDA ===")
    generate_eda_plots(d_order, d_cohort, 'outputs')
    print("-> Đã xuất các biểu đồ fig_*.png vào outputs/")
