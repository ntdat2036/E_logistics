"""test_reproducibility.py - Kiểm thử tính tái lập 100% (byte-identical) của dữ liệu xuất bản web.
Đảm bảo hai lần xuất liên tiếp cho kết quả trùng khớp hoàn toàn từng byte.
"""
import os
import sys
import tempfile
import filecmp
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.export_web import export_all_web_data


def test_triage_and_web_data_reproducibility():
    """Kiểm tra export_all_web_data chạy hai lần vào hai thư mục tạm độc lập cho ra các file json giống hệt nhau 100%."""
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        out1 = os.path.join(tmpdir1, 'web_data')
        out2 = os.path.join(tmpdir2, 'web_data')

        export_all_web_data(out_dir=out1)
        export_all_web_data(out_dir=out2)

        files1 = sorted(os.listdir(out1))
        files2 = sorted(os.listdir(out2))

        assert files1 == files2, f"Danh sách file không khớp: {files1} vs {files2}"
        assert 'triage.json' in files1, "Thiếu file triage.json"

        for fname in files1:
            p1 = os.path.join(out1, fname)
            p2 = os.path.join(out2, fname)
            assert filecmp.cmp(p1, p2, shallow=False), f"File {fname} không trùng khớp byte giữa hai lần xuất!"
