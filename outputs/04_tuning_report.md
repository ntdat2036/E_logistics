# BÁO CÁO TINH CHỈNH VÀ CHỈ SỐ CUỐI CÙNG (GIAI ĐOẠN 4)

## 1. Siêu tham số tối ưu (GridSearch CV Expanding-Window)

```json
{
  "max_depth": 6,
  "min_samples_leaf": 100,
  "criterion": "gini",
  "n_bins": 32,
  "min_samples_split": 200
}
```

## 2. Ngưỡng phân vị và ngưỡng phân loại đã khóa (từ Validation)

```json
{
  "t1": 0.1883,
  "t2": 0.0651,
  "t_f1": 0.1799,
  "val_best_f1": 0.2828,
  "t_cost": 0.506,
  "val_min_cost": 3101.0,
  "c_fn": 3.0,
  "c_fp": 1.0,
  "prior_del": 0.0889,
  "prior_han": 0.0952,
  "train_median_weight": 800.0,
  "train_median_vol": 7803.0
}
```

## 3. Chỉ số kiểm định trên tập Test

- **Test ROC-AUC:** 0.6553
- **Test PR-AUC:** 0.1548
- **Test Brier score:** 0.0711
- **Precision / Recall / F1 tại ngưỡng 0.1799:** 0.1859 / 0.3887 / 0.2515
- **Confusion Matrix:** TN=14976, FP=2387, FN=857, TP=545

## 4. Phân tầng ưu tiên trên Test

| Mức ưu tiên | Tỷ lệ đơn | Tỷ lệ trễ | Số đơn trễ bắt được |
|---|---|---|---|
| **P1** | 15.6% (2,932) | 18.6% | 38.9% (545) |
| **P2** | 50.4% (9,459) | 5.9% | 39.9% (559) |
| **P3** | 34.0% (6,374) | 4.7% | 21.3% (298) |

- **Recall top 15% rủi ro nhất:** 38.0%
- **Tổng số đơn cần Trưởng ca xử lý thủ công:** 41.1% (bắt được 57.5% đơn trễ)
