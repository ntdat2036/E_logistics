"""test_simulate.py - Kiểm thử mô phỏng hàng đợi kho (Mục 9.1).
1. Tính bảo toàn: Mọi đơn được xử lý đúng 1 lần, thời điểm hoàn tất done >= t_arr.
2. Không đơn nào hoàn tất vào Chủ nhật hay ngoài khung giờ 08:00 - 17:00.
3. Khi K=1, chính sách FIFO phục vụ tuần tự đúng thứ tự đơn đến.
"""
import numpy as np
import pytest
from src.simulate import simulate, next_work, policies, kpis, DAY_START, DAY_END


def test_next_work_rules():
    """Kiểm tra hàm next_work không bao giờ trả về Chủ nhật hoặc ngoài khung 08:00-17:00."""
    # Thử trên 500 mốc thời gian ngẫu nhiên (phút kể từ epoch)
    rng = np.random.default_rng(42)
    sample_t = rng.integers(1_000_000, 2_000_000, size=500)

    for t in sample_t:
        t_work = next_work(t)
        assert t_work >= t, "next_work phải >= t"
        dow = ((t_work // 1440) + 3) % 7
        m = t_work % 1440
        assert dow != 6, f"next_work trả về Chủ nhật! (t_work={t_work}, dow={dow})"
        assert DAY_START <= m <= DAY_END, f"next_work ngoài khung 08:00-17:00! (m={m})"


def test_simulation_conservation_and_working_hours():
    """Kiểm tra mọi đơn được hoàn tất, done >= t_arr, không vào Chủ nhật hay ngoài giờ."""
    n = 200
    # Giả lập 200 đơn đến rải rác trong 5 ngày làm việc
    rng = np.random.default_rng(2026)
    # Bắt đầu vào Thứ Hai lúc 08:30 (epoch phút)
    start_time = 1440 * 1000 + 8 * 60 + 30
    t_arr = np.sort(start_time + np.cumsum(rng.exponential(scale=15.0, size=n)).astype(np.int64))
    t_due = t_arr + rng.integers(24 * 60, 72 * 60, size=n)
    svc = rng.choice([6.0, 9.0, 12.0], size=n)
    prio = np.arange(n, dtype=float)

    done = simulate(t_arr, t_due, svc, prio, K=3)

    assert len(done) == n, "Tất cả đơn phải được xử lý"
    assert (done >= t_arr).all(), "LỖI BẮT ĐẦU TRƯỚC GIỜ ĐẾN: Tồn tại đơn có done < t_arr!"

    # Kiểm tra ca làm việc
    for d in done:
        dow = ((d // 1440) + 3) % 7
        m = d % 1440
        assert dow != 6, f"Đơn hoàn tất vào Chủ nhật! (done={d})"
        assert DAY_START <= m <= DAY_END, f"Đơn hoàn tất ngoài giờ làm việc! (done={d}, m={m})"


def test_fifo_single_worker_order():
    """Khi K=1, chính sách FIFO phải phục vụ tuần tự theo thứ tự đơn đến."""
    n = 50
    rng = np.random.default_rng(123)
    start_time = 1440 * 1000 + 8 * 60 + 30
    t_arr = np.sort(start_time + np.cumsum(rng.integers(5, 20, size=n)))
    t_due = t_arr + 24 * 60
    svc = np.full(n, 10.0)
    prio = np.arange(n, dtype=float)  # FIFO

    done = simulate(t_arr, t_due, svc, prio, K=1)

    # Đơn thứ tự trước phải hoàn tất trước đơn thứ tự sau khi 1 nhân viên
    assert (np.diff(done) >= 0).all(), "Khi K=1, FIFO phải hoàn tất đơn theo đúng thứ tự tăng dần!"

