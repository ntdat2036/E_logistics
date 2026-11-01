"""simulate.py - Mô phỏng hàng đợi 1 trung tâm xử lý đơn hàng: K nhân viên, ca 08:00-17:00 Thứ 2 - Thứ 7.
TUÂN THỦ TUYỆT ĐỐI MỤC 2 (R2), MỤC 7 VÀ PHỤ LỤC C CỦA HUONG_DAN.md.
GIẢ ĐỊNH (công bố trong báo cáo): Thời gian xử lý 1 đơn = SVC_BASE + SVC_PER_ITEM * (n_items - 1) phút.
Thời điểm đơn đến (order_approved_at) và hạn bàn giao (shipping_limit_date) là dữ liệu THẬT.
"""
import heapq
import json
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd

SVC_BASE, SVC_PER_ITEM = 6.0, 3.0
DAY_START, DAY_END = 8 * 60, 17 * 60  # 08:00 đến 17:00 (540 phút làm việc / ngày)


def to_min(x):
    """Chuyển đổi mảng thời gian datetime64 sang số phút kể từ epoch."""
    return x.astype('datetime64[m]').astype(np.int64)


def next_work(t):
    """Đẩy mốc thời gian t tới phút làm việc hợp lệ kế tiếp (bỏ qua Chủ nhật; ca 08:00-17:00).
    Lưu ý: epoch 1970-01-01 là Thứ 5, do đó công thức ((t // 1440) + 3) % 7 cho Mon=0 ... Sun=6.
    """
    while True:
        dow = ((t // 1440) + 3) % 7
        m = t % 1440
        if dow == 6 or m >= DAY_END:
            t = (t // 1440 + 1) * 1440 + DAY_START
            continue
        if m < DAY_START:
            return (t // 1440) * 1440 + DAY_START
        return t


def simulate(t_arr, t_due, svc, prio, K):
    """Mô phỏng hàng đợi K nhân viên.
    Trả về mảng thời điểm hoàn tất (phút kể từ epoch).
    prio nhỏ hơn = ưu tiên làm trước.
    Không preempt (không ngắt ngang đơn đang làm); đơn làm dở cuối ca được tiếp tục vào đầu ca sau.
    Đã kiểm chứng: Không bao giờ bắt đầu đơn trước thời điểm đơn đến (t >= t_arr[j]).
    """
    n = len(t_arr)
    order = np.argsort(t_arr, kind='stable')
    free = [int(t_arr.min())] * K
    heapq.heapify(free)

    done = np.full(n, -1, dtype=np.int64)
    pend = []
    i = 0

    while i < n or pend:
        tf = next_work(heapq.heappop(free))
        if not pend and i < n:
            tf = max(tf, next_work(int(t_arr[order[i]])))
        while i < n and t_arr[order[i]] <= tf:
            heapq.heappush(pend, (prio[order[i]], order[i]))
            i += 1

        _, j = heapq.heappop(pend)
        t = max(tf, int(t_arr[j]))  # Đảm bảo tuyệt đối không bắt đầu trước lúc đơn đến
        rem = float(svc[j])

        while rem > 0:
            t = next_work(t)
            use = min(rem, DAY_END - (t % 1440))
            t += use
            rem -= use

        done[j] = int(np.ceil(t))
        heapq.heappush(free, done[j])

    return done


def policies(t_due, p_risk, delta_h=24.0, t1=None, t2=None):
    """Tạo các chính sách ưu tiên theo Mục 7.2:
    - FIFO: Thứ tự đến (As-Is truyền thống theo đề cương).
    - EDD: Sắp theo hạn bàn giao (baseline kinh điển, không cần AI).
    - AI_EDD: Hạn bàn giao - 24h * điểm rủi ro (To-Be cải tiến).
    - AI_level_first: Minh họa sai lầm (làm hết P1 rồi P2 rồi P3, bỏ qua hạn).
    """
    n = len(t_due)
    pols = {
        'FIFO': np.arange(n, dtype=float),
        'EDD': t_due.astype(float),
        'AI_EDD': t_due - delta_h * 60.0 * p_risk,
    }
    if t1 is not None and t2 is not None:
        level = np.where(p_risk >= t1, 1.0, np.where(p_risk >= t2, 2.0, 3.0))
        # P1 = 1e12, P2 = 2e12, P3 = 3e12, bên trong cùng level sắp theo thứ tự đến
        pols['AI_level_first'] = level * 1e12 + np.arange(n, dtype=float)
    return pols


def kpis(done, t_arr, t_due):
    """Tính 3 KPI cốt lõi theo Mục 7.3: Lead time TB/p95, Tỷ lệ đúng hạn, Tỷ lệ trễ hạn."""
    lead_h = (done - t_arr) / 60.0
    return dict(
        late_rate=round(float((done > t_due).mean()), 4),
        on_time_rate=round(float((done <= t_due).mean()), 4),
        lead_mean_h=round(float(lead_h.mean()), 2),
        lead_p95_h=round(float(np.percentile(lead_h, 95)), 2),
    )


def run_simulation_grid(t_arr, t_due, n_items, p_risk, t1, t2, out_dir='outputs'):
    """Chạy phân tích độ nhạy trên lưới: K ∈ {3,4,5,6} × svc_mult ∈ {0.8, 0.9, 1.0} (Mục 7.1)."""
    os.makedirs(out_dir, exist_ok=True)
    k_list = [3, 4, 5, 6]
    mult_list = [0.8, 0.9, 1.0]

    grid_results = {}
    pols = policies(t_due, p_risk, delta_h=24.0, t1=t1, t2=t2)

    for k in k_list:
        grid_results[f"K={k}"] = {}
        for m in mult_list:
            m_str = f"time_factor={m:.1f}"
            grid_results[f"K={k}"][m_str] = {}
            svc = (SVC_BASE + SVC_PER_ITEM * (n_items - 1)) * m

            for pol_name, prio in pols.items():
                done = simulate(t_arr, t_due, svc, prio, K=k)
                res = kpis(done, t_arr, t_due)
                grid_results[f"K={k}"][m_str][pol_name] = res

    # Lưu kết quả lưới vào JSON
    grid_path = os.path.join(out_dir, '05_simulation_grid.json')
    with open(grid_path, 'w', encoding='utf-8') as f:
        json.dump(grid_results, f, indent=2)

    return grid_results

