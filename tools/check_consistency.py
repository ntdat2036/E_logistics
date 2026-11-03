"""tools/check_consistency.py - Kiểm toán tính nhất quán số liệu và tài liệu toàn repo.
Kiểm tra các chỉ số giữa outputs/, web/data/, README.md, và notebook.ipynb.
Báo cáo: [PASS] hoặc [FAIL] cho từng hạng mục. Mục tiêu: 0 lỗi không nhất quán.
"""
import os
import sys
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def check_consistency():
    failures = []
    passes = []

    def record_pass(item_name, detail=""):
        passes.append((item_name, detail))
        print(f"  [PASS] {item_name}: {detail}")

    def record_fail(item_name, detail):
        failures.append((item_name, detail))
        print(f"  [FAIL] {item_name}: {detail}")

    print("=" * 80)
    print("KIỂM TOÁN TÍNH NHẤT QUÁN TOÀN REPO E_LOGISTICS (CK10)")
    print("=" * 80)

    # 1. Nạp các nguồn dữ liệu chính
    p_03 = BASE_DIR / 'outputs' / '03_model_comparison.json'
    p_04 = BASE_DIR / 'outputs' / '04_final_metrics.json'
    p_05_iter = BASE_DIR / 'outputs' / '05_iteration_table.json'
    p_05_sim = BASE_DIR / 'outputs' / '05_simulation_grid.json'
    p_web_metrics = BASE_DIR / 'web' / 'data' / 'metrics.json'
    p_readme = BASE_DIR / 'README.md'
    p_notebook = BASE_DIR / 'notebook.ipynb'

    d_03 = json.load(open(p_03, encoding='utf-8'))
    d_04 = json.load(open(p_04, encoding='utf-8'))
    d_iter = json.load(open(p_05_iter, encoding='utf-8'))
    d_sim = json.load(open(p_05_sim, encoding='utf-8'))
    d_web = json.load(open(p_web_metrics, encoding='utf-8'))
    readme_text = open(p_readme, encoding='utf-8').read()
    nb_text = open(p_notebook, encoding='utf-8').read()

    # --- HẠNG MỤC 1: Metric Mô hình Lõi Decision Tree Core ---
    print("\n1. Kiểm tra mô hình chính Decision Tree Core (outputs vs web vs docs):")
    core_03 = d_03['7_Decision_Tree_Core']['test']
    roc_04 = d_04['roc_auc']
    pr_04 = d_04['pr_auc']
    brier_04 = d_04['brier']

    if (core_03['roc_auc'] == roc_04 == 0.6553):
        record_pass("Test ROC-AUC Decision Tree", f"0.6553 khớp giữa 03_model_comparison ({core_03['roc_auc']}) và 04_final_metrics ({roc_04})")
    else:
        record_fail("Test ROC-AUC Decision Tree", f"Lệch: 03={core_03['roc_auc']} vs 04={roc_04}")

    if (core_03['pr_auc'] == pr_04 == 0.1548):
        record_pass("Test PR-AUC Decision Tree", f"0.1548 khớp giữa 03_model_comparison ({core_03['pr_auc']}) và 04_final_metrics ({pr_04})")
    else:
        record_fail("Test PR-AUC Decision Tree", f"Lệch: 03={core_03['pr_auc']} vs 04={pr_04}")

    if (core_03['brier'] == brier_04 == 0.0711):
        record_pass("Test Brier Score Decision Tree", f"0.0711 khớp giữa 03_model_comparison ({core_03['brier']}) và 04_final_metrics ({brier_04})")
    else:
        record_fail("Test Brier Score Decision Tree", f"Lệch: 03={core_03['brier']} vs 04={brier_04}")

    # Web metrics
    web_core = d_web['no_free_lunch_comparison']['7_Decision_Tree_Core']['test']
    if web_core['roc_auc'] == 0.6553 and web_core['pr_auc'] == 0.1548 and web_core['brier'] == 0.0711:
        record_pass("web/data/metrics.json Decision Tree", "Khớp 100% (0.6553 / 0.1548 / 0.0711)")
    else:
        record_fail("web/data/metrics.json Decision Tree", f"Lệch trong web metrics: {web_core}")

    # README Table 4.1
    if "0.6553" in readme_text and "0.1548" in readme_text and "0.0711" in readme_text:
        record_pass("README.md Bảng 4.1 Decision Tree", "Có đầy đủ 0.6553, 0.1548, 0.0711")
    else:
        record_fail("README.md Bảng 4.1 Decision Tree", "Không tìm thấy đủ các chỉ số của Decision Tree trong README.md")

    # --- HẠNG MỤC 2: Brier score các luật heuristic FIFO và Slack ---
    print("\n2. Kiểm tra Brier Score phi xác suất của FIFO và Slack:")
    fifo_brier = d_03['2_Rule_FIFO']['test']['brier']
    slack_brier = d_03['3_Rule_Slack']['test']['brier']
    expected_non_prob = "n/a (không phải xác suất)"

    if fifo_brier == expected_non_prob and slack_brier == expected_non_prob:
        record_pass("Brier FIFO/Slack trong 03_model_comparison", f"Hiển thị '{expected_non_prob}'")
    else:
        record_fail("Brier FIFO/Slack trong 03_model_comparison", f"Lệch: FIFO='{fifo_brier}', Slack='{slack_brier}'")

    if expected_non_prob in readme_text:
        record_pass("README.md Bảng 4.1 Brier FIFO/Slack", f"Hiển thị '{expected_non_prob}'")
    else:
        record_fail("README.md Bảng 4.1 Brier FIFO/Slack", "README chưa cập nhật nhãn n/a cho FIFO/Slack")

    # --- HẠNG MỤC 3: Triage P1/P2/P3 & Cờ Trưởng ca ---
    print("\n3. Kiểm tra phân tầng rủi ro Triage P1/P2/P3 & Cờ Trưởng ca:")
    p1_captured = d_04['priority_levels']['P1']['late_captured_share']
    sup_captured = d_04['shift_supervisor_manual_flag']['late_orders_captured_share']

    if round(p1_captured, 3) == 0.389:
        record_pass("P1 late orders captured share", f"38.9% ({p1_captured:.4f})")
    else:
        record_fail("P1 late orders captured share", f"Giá trị không phải 38.9%: {p1_captured}")

    if round(sup_captured, 3) == 0.575:
        record_pass("Supervisor flag late captured share", f"57.5% ({sup_captured:.4f})")
    else:
        record_fail("Supervisor flag late captured share", f"Giá trị không phải 57.5%: {sup_captured}")

    iter5_note = d_iter[-1]['note']
    if "38.9%" in iter5_note and "57.5%" in iter5_note:
        record_pass("05_iteration_table.json Iter 5 note", f"Khớp chính xác: '{iter5_note}'")
    else:
        record_fail("05_iteration_table.json Iter 5 note", f"Lệch số trong note Iter 5: '{iter5_note}'")

    # --- HẠNG MỤC 4: Kết quả mô phỏng hàng đợi kho K=3, factor=1.0 ---
    print("\n4. Kiểm tra kết quả mô phỏng hàng đợi kho:")
    sim_k3 = d_sim['K=3']['time_factor=1.0']
    fifo_late = sim_k3['FIFO']['late_rate']
    edd_late = sim_k3['EDD']['late_rate']
    aiedd_late = sim_k3['AI_EDD']['late_rate']
    ailvl_late = sim_k3['AI_level_first']['late_rate']

    if round(fifo_late, 3) == 0.072 and round(edd_late, 3) == 0.008 and round(aiedd_late, 3) == 0.009 and round(ailvl_late, 3) == 0.116:
        record_pass("Tỷ lệ trễ mô phỏng 05_simulation_grid", f"FIFO: {fifo_late:.1%}, EDD: {edd_late:.1%}, AI-EDD: {aiedd_late:.1%}, AI-level: {ailvl_late:.1%}")
    else:
        record_fail("Tỷ lệ trễ mô phỏng 05_simulation_grid", f"Số liệu mô phỏng bất thường: FIFO={fifo_late}, EDD={edd_late}, AI-EDD={aiedd_late}, AI-level={ailvl_late}")

    # --- HẠNG MỤC 5: Kiểm tra dọn dẹp file thừa & không rò rỉ thông tin nội bộ ---
    print("\n5. Kiểm tra tính toàn vẹn kho lưu trữ (không file thừa, không đường dẫn tuyệt đối):")
    p_huong_dan = BASE_DIR / 'HUONG_DAN.md'
    if not p_huong_dan.exists():
        record_pass("File HUONG_DAN.md đã xóa vĩnh viễn", "Đã xóa khỏi filesystem")
    else:
        record_fail("File HUONG_DAN.md", "Vẫn còn tồn tại trong repo!")

    redundant_scripts = [BASE_DIR / 'build_custom_notebook.py', BASE_DIR / 'generate_complete_notebook.py']
    found_redundant = [f.name for f in redundant_scripts if f.exists()]
    if not found_redundant:
        record_pass("Script sinh notebook thừa", "Chỉ giữ đúng build_notebook.py")
    else:
        record_fail("Script sinh notebook thừa", f"Vẫn còn file thừa: {found_redundant}")

    # Kiểm tra đường dẫn cá nhân
    personal_hits = []
    exclude_dirs = {'.git', '.pytest_cache', '__pycache__', '.ipynb_checkpoints', 'data'}
    bad_patterns = ['Users/' + 'DELL', 'Users\\' + 'DELL', 'file:///' + 'c:', 'file:///' + 'C:']
    for root, dirs, files in os.walk(BASE_DIR):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in files:
            if f.endswith(('.py', '.md', '.html', '.js', '.css', '.txt')):
                p = os.path.join(root, f)
                if Path(p).resolve() == Path(__file__).resolve():
                    continue
                try:
                    c = open(p, encoding='utf-8', errors='ignore').read()
                    if any(pat in c for pat in bad_patterns):
                        personal_hits.append(p)
                except Exception:
                    pass

    if not personal_hits:
        record_pass("Đường dẫn cá nhân tuyệt đối", "0 đường dẫn tuyệt đối trong toàn bộ mã nguồn và tài liệu")
    else:
        record_fail("Đường dẫn cá nhân tuyệt đối", f"Phát hiện tại: {personal_hits}")

    # --- TỔNG KẾT ---
    print("\n" + "=" * 80)
    print(f"KẾT QUẢ KIỂM TOÁN: {len(passes)} PASSED, {len(failures)} FAILED")
    print("=" * 80)

    if failures:
        print("\nCÁC MỤC THẤT BẠI:")
        for name, det in failures:
            print(f"  ❌ {name}: {det}")
        sys.exit(1)
    else:
        print("\n🎉 TOÀN BỘ CÁC HẠNG MỤC ĐÃ ĐẠT CHUẨN NHẤT QUÁN 100%! 0 LỖI KHÔNG NHẤT QUÁN.")
        sys.exit(0)


if __name__ == '__main__':
    check_consistency()
