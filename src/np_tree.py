"""np_tree.py - Re-export module cho DecisionTree, BinMapper, Node từ src/model.py.
Giữ lại module này làm alias độc lập để phục vụ quy ước gọi tên và import trực tiếp của bài toán CK10 E-logistics.
Tuân thủ 100% nguyên tắc R2 (Không import scikit-learn).
"""
import sys
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.model import BinMapper, Node, DecisionTree

__all__ = ['BinMapper', 'Node', 'DecisionTree']

