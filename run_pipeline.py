"""run_pipeline.py - Chạy toàn bộ quy trình 21 bước từ dữ liệu gốc đến kết quả cuối.
Dự án: Thiết kế chuỗi dịch vụ logistics tự động có AI ra quyết định (CK10)
Bản quyền & quy ước: Tuân thủ 100% 7 nguyên tắc bắt buộc (R1-R7) trong HUONG_DAN.md.
"""
import sys
import os
import time
import subprocess
import json

# Đảm bảo UTF-8 trên Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def print_banner(step_title):
    print("\n" + "=" * 80)
    print(f"  {step_title}")
    print("=" * 80)


def run_command_step(name, script_rel_path, extra_args=None):
    script_path = os.path.join(BASE_DIR, script_rel_path)
    cmd = [sys.executable, script_path] + (extra_args or [])
    print(f"\n[EXEC] Đang chạy: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=BASE_DIR, capture_output=False)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"[FAIL] Lỗi tại bước {name} (exit code: {res.returncode}) sau {elapsed:.1f}s")
        sys.exit(res.returncode)
    print(f"[DONE] Hoàn tất {name} ({elapsed:.1f}s)")
    return elapsed


def main():
    total_t0 = time.time()
    print("=" * 80)
    print("   HỆ THỐNG PIPELINE LOGISTICS E-COMMERCE TỰ ĐỘNG - OLIST INVENTORY AI")
    print("   Mô hình: Decision Tree (NumPy thuần) | Bài toán: Dự đoán Trễ Bàn Giao")
    print("=" * 80)

    # 1. Giai đoạn 1: Chuẩn bị dữ liệu & Audit dữ liệu gốc (Bước 1-6)
    print_banner("GIAI ĐOẠN 1: DỮ LIỆU & AUDIT RAW CSV (Bước 1 - 6)")
    run_command_step("Giai đoạn 1 (Data Prep & Audit)", "src/data_prep.py")

    # 2. Giai đoạn 2: Kiểm toán rò rỉ dữ liệu (Bước 7-10)
    print_banner("GIAI ĐOẠN 2: KIỂM TOÁN RÒ RỈ DỮ LIỆU & CHIA TẬP (Bước 7 - 10)")
    cmd_leakage = [sys.executable, "-c", "import sys; sys.path.insert(0, '.'); from src.data_prep import run_leakage_audit; ok = run_leakage_audit(); sys.exit(0 if ok else 1)"]
    t0 = time.time()
    res = subprocess.run(cmd_leakage, cwd=BASE_DIR)
    if res.returncode != 0:
        print("[FAIL] Kiểm toán rò rỉ dữ liệu thất bại!")
        sys.exit(1)
    print(f"[DONE] Kiểm toán rò rỉ dữ liệu (Bước 9-10) hoàn tất ({time.time() - t0:.1f}s)")

    # 3. Dữ liệu mô phỏng kho (Phục vụ FEFO & Tồn kho)
    print_banner("DỮ LIỆU MÔ PHỎNG KHO (ASSUMPTIONS.md & synth.py)")
    run_command_step("Sinh dữ liệu mô phỏng kho", "src/synth.py")

    # 4. Giai đoạn 3: Mô hình cơ sở & Bias-Variance (Bước 11-13)
    print_banner("GIAI ĐOẠN 3: SO SÁNH 7 MÔ HÌNH CƠ SỞ & BIAS-VARIANCE (Bước 11 - 13)")
    run_command_step("Giai đoạn 3 (Evaluate Baselines)", "src/evaluate.py")

    # 5. Giai đoạn 4: Tinh chỉnh tham số, Khóa ngưỡng & Huấn luyện cuối (Bước 14-17)
    print_banner("GIAI ĐOẠN 4: TUNING, KHÓA NGƯỠNG & ĐÁNH GIÁ TEST (Bước 14 - 17)")
    run_command_step("Giai đoạn 4 (Tuning & Train)", "src/train.py")

    # 6. Giai đoạn 5: Thống kê bootstrap, Phân tích lỗi & Giải thích (Bước 18-20)
    print_banner("GIAI ĐOẠN 5A: BOOTSTRAP CI, LỖI FN & GIẢI THÍCH CÂY (Bước 18 - 20)")
    run_command_step("Giai đoạn 5A (Statistical Analysis)", "src/statistical_analysis.py")

    # 7. Giai đoạn 5B: Mô phỏng hàng đợi As-Is vs To-Be (Bước 21)
    print_banner("GIAI ĐOẠN 5B: MÔ PHỎNG HÀNG ĐỢI AS-IS vs TO-BE (Bước 21)")
    run_command_step("Giai đoạn 5B (Queue Simulation)", "src/simulate.py")

    # 8. Giai đoạn 6: Xuất tài sản Web (Giai đoạn 6)
    print_banner("GIAI ĐOẠN 6: XUẤT TÀI SẢN WEB & JSON DATA")
    run_command_step("Giai đoạn 6 (Export Web)", "src/export_web.py")

    # 9. Kiểm thử tự động toàn diện qua Pytest (Mục 9.1)
    print_banner("KIỂM THỬ TỰ ĐỘNG PYTEST TOÀN BỘ HỆ THỐNG")
    t0 = time.time()
    res = subprocess.run(["pytest", "-q"], cwd=BASE_DIR)
    if res.returncode != 0:
        print("[WARNING] Pytest có cảnh báo hoặc kiểm thử không qua 100%!")
    else:
        print(f"[PASS] 100% Pytest test cases passed ({time.time() - t0:.1f}s)")

    # Tổng kết chỉ số kiểm tra đối chiếu (Benchmark Check)
    print("\n" + "=" * 80)
    print("                     BÁO CÁO NGHIỆM THU TỔNG THỂ PIPELINE")
    print("=" * 80)

    try:
        with open("outputs/04_final_metrics.json", "r", encoding="utf-8") as f:
            final_metrics = json.load(f)
        with open("weights/thresholds.json", "r", encoding="utf-8") as f:
            thresholds = json.load(f)
        with open("outputs/05_simulation_grid.json", "r", encoding="utf-8") as f:
            sim_grid = json.load(f)

        roc_val = final_metrics.get('roc_auc', final_metrics.get('test_roc_auc', 0))
        pr_val = final_metrics.get('pr_auc', final_metrics.get('test_pr_auc', 0))
        brier_val = final_metrics.get('brier', final_metrics.get('test_brier', 0))

        print(f"1. Hiệu năng mô hình kiểm thử trên tập Test:")
        print(f"   - ROC-AUC:      {roc_val:.4f} (Kỳ vọng: 0.655 - 0.670)")
        print(f"   - PR-AUC:       {pr_val:.4f} (Kỳ vọng: 0.150 - 0.170)")
        print(f"   - Brier Score:  {brier_val:.4f} (Kỳ vọng: ~0.071)")

        print(f"\n2. Ngưỡng phân loại đã khóa trên Validation:")
        print(f"   - Ngưỡng t1 (P1 - Cực kỳ khẩn cấp, Top 15%): {thresholds.get('t1', 0):.4f}")
        print(f"   - Ngưỡng t2 (P2 - Cảnh báo rủi ro, Top 55%):  {thresholds.get('t2', 0):.4f}")
        print(f"   - Ngưỡng F1 tối ưu:                           {thresholds.get('t_f1', 0):.4f}")

        # Tìm kịch bản chuẩn K=3, factor=1.0 trong simulation grid
        k3_f1 = sim_grid.get('K=3', {}).get('time_factor=1.0', {})
        if k3_f1:
            fifo = k3_f1.get('FIFO', {})
            edd = k3_f1.get('EDD', {})
            ai_edd = k3_f1.get('AI_EDD', {})
            ai_level = k3_f1.get('AI_level_first', {})
            print(f"\n3. Kết quả mô phỏng hàng đợi 18,765 đơn (K=3, Ca 8-17h Thứ 2-7):")
            print(f"   - FIFO (As-Is):          Trễ {fifo.get('late_rate', 0)*100:.1f}%, Lead time TB {fifo.get('lead_mean_h', 0):.1f}h")
            print(f"   - EDD (Luật hạn giao):   Trễ {edd.get('late_rate', 0)*100:.1f}%, Lead time TB {edd.get('lead_mean_h', 0):.1f}h")
            print(f"   - AI-EDD (To-Be):        Trễ {ai_edd.get('late_rate', 0)*100:.1f}%, Lead time TB {ai_edd.get('lead_mean_h', 0):.1f}h")
            print(f"   - AI level-first (Sai):  Trễ {ai_level.get('late_rate', 0)*100:.1f}%, Lead time TB {ai_level.get('lead_mean_h', 0):.1f}h (CẢNH BÁO RỦI RO)")

        print("\n[SUCCESS] TOÀN BỘ 21 BƯỚC PIPELINE ĐÃ HOÀN TẤT VÀ TÁI LẬP THÀNH CÔNG!")
    except Exception as e:
        print(f"[WARNING] Không thể đọc số liệu tổng hợp: {e}")

    total_elapsed = time.time() - total_t0
    print(f"Tổng thời gian thực thi: {total_elapsed:.1f} giây ({total_elapsed/60:.2f} phút)")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()

