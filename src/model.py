"""model.py - Thuật toán Machine Learning và bộ đo lường thuần NumPy.
Quy tắc R2: KHÔNG import scikit-learn trong src/.
Bao gồm:
- BinMapper: Ánh xạ quantile bins (32 hoặc 64)
- DecisionTree: Cây quyết định thuần NumPy (hỗ trợ Gini/Entropy, min_samples_split, max_features, class_weight)
- LogisticRegression: Hồi quy Logistic tối ưu Gradient Descent, L2, class_weight, loss_history_
- StandardScaler: Chuẩn hóa Z-score thuần NumPy
- Bộ metrics thuần NumPy: accuracy, precision, recall, f1, roc_auc, pr_auc (average precision), confusion_matrix, brier_score
- Cross-validation splitters: TimeSeriesSplit (expanding-window), StratifiedKFold
- GridSearch: Tìm kiếm lưới siêu tham số theo CV thuần NumPy
"""
import numpy as np


# ==============================================================================
# 1. BỘ CHUẨN HÓA & ÁNH XẠ BIN
# ==============================================================================

class BinMapper:
    """Ánh xạ các đặc trưng liên tục thành các bin phân vị (quantile bins)."""
    def __init__(self, n_bins=64):
        self.n_bins = n_bins

    def fit(self, X):
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges_ = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        return np.stack(
            [np.searchsorted(self.edges_[j], X[:, j], side='right') for j in range(X.shape[1])],
            axis=1,
        ).astype(np.int16)


class StandardScaler:
    """Chuẩn hóa Z-score thuần NumPy: (X - mean) / std."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ==============================================================================
# 2. CÂY QUYẾT ĐỊNH THUẦN NUMPY
# ==============================================================================

class DecisionTree:
    """Cây quyết định phân loại nhị phân thuần NumPy.
    Hỗ trợ criterion ('gini' hoặc 'entropy'), min_samples_split, max_features, class_weight.
    """
    def __init__(
        self,
        max_depth=5,
        min_samples_leaf=200,
        min_samples_split=None,
        criterion='gini',
        max_features=None,
        class_weight=None,
        n_bins=64,
        random_state=42,
    ):
        self.max_depth = max_depth
        self.min_leaf = min_samples_leaf
        self.min_samples_split = (
            min_samples_split if min_samples_split is not None else 2 * min_samples_leaf
        )
        self.criterion = criterion.lower()
        self.max_features = max_features
        self.cw = class_weight
        self.n_bins = n_bins
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.rng_ = np.random.default_rng(self.random_state)
        self.n_features_in_ = X.shape[1]
        self.n_samples_ = len(y)

        self.bm_ = BinMapper(self.n_bins).fit(X)
        B = self.bm_.transform(X)

        w = np.ones(len(y), dtype=float)
        if self.cw == 'balanced':
            n1 = float(y.sum())
            n0 = float(len(y) - n1)
            w = np.where(y == 1, len(y) / (2.0 * max(n1, 1.0)), len(y) / (2.0 * max(n0, 1.0)))

        self.feat, self.thr, self.left, self.right, self.val, self.cnt = [], [], [], [], [], []
        self._importances_raw = np.zeros(self.n_features_in_, dtype=float)

        self._grow(B, y, w, np.arange(len(y)), 0)

        self.feat = np.array(self.feat, dtype=int)
        self.thr = np.array(self.thr, dtype=int)
        self.left = np.array(self.left, dtype=int)
        self.right = np.array(self.right, dtype=int)
        self.val = np.array(self.val, dtype=float)
        self.cnt = np.array(self.cnt, dtype=int)

        tot_imp = self._importances_raw.sum()
        if tot_imp > 0:
            self.feature_importances_ = self._importances_raw / tot_imp
        else:
            self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)

        return self

    def _new(self, v, n):
        self.feat.append(-1)
        self.thr.append(-1)
        self.left.append(-1)
        self.right.append(-1)
        self.val.append(v)
        self.cnt.append(n)
        return len(self.feat) - 1

    def _calc_impurity(self, p):
        p = np.clip(p, 1e-15, 1.0 - 1e-15)
        if self.criterion == 'entropy':
            return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
        # Default gini
        return 2.0 * p * (1.0 - p)

    def _grow(self, B, y, w, idx, depth):
        wy, ww = (w[idx] * y[idx]), w[idx]
        tot_w = float(ww.sum())
        tot_p = float(wy.sum())
        p_parent = tot_p / max(tot_w, 1e-12)
        node = self._new(p_parent, len(idx))

        if (
            depth >= self.max_depth
            or len(idx) < self.min_samples_split
            or len(idx) < 2 * self.min_leaf
            or tot_p == 0
            or tot_p == tot_w
        ):
            return node

        parent_impurity = self._calc_impurity(p_parent)

        # Chọn tập đặc trưng phân nhánh (max_features)
        if self.max_features == 'sqrt':
            k_feats = max(1, int(np.sqrt(self.n_features_in_)))
            feat_subset = self.rng_.choice(self.n_features_in_, size=k_feats, replace=False)
        elif isinstance(self.max_features, int):
            k_feats = min(self.max_features, self.n_features_in_)
            feat_subset = self.rng_.choice(self.n_features_in_, size=k_feats, replace=False)
        elif isinstance(self.max_features, float):
            k_feats = max(1, int(self.max_features * self.n_features_in_))
            feat_subset = self.rng_.choice(self.n_features_in_, size=k_feats, replace=False)
        else:
            feat_subset = np.arange(self.n_features_in_)

        best = (0.0, -1, -1)  # (best_gain, best_feat, best_thr)

        for j in feat_subset:
            bj = B[idx, j]
            nb = int(bj.max()) + 1
            hw = np.bincount(bj, ww, minlength=nb)
            hp = np.bincount(bj, wy, minlength=nb)
            hn = np.bincount(bj, minlength=nb)

            cw_ = np.cumsum(hw)[:-1]
            cp_ = np.cumsum(hp)[:-1]
            cn_ = np.cumsum(hn)[:-1]

            ok = (
                (cn_ >= self.min_leaf)
                & ((len(idx) - cn_) >= self.min_leaf)
                & (cw_ > 0)
                & ((tot_w - cw_) > 0)
            )
            if not ok.any():
                continue

            pl = cp_ / np.where(cw_ > 0, cw_, 1.0)
            pr = (tot_p - cp_) / np.where((tot_w - cw_) > 0, tot_w - cw_, 1.0)

            if self.criterion == 'entropy':
                imp_l = self._calc_impurity(pl)
                imp_r = self._calc_impurity(pr)
                gain = parent_impurity - (cw_ * imp_l + (tot_w - cw_) * imp_r) / tot_w
            else:
                gain = parent_impurity - (
                    cw_ * 2.0 * pl * (1.0 - pl) + (tot_w - cw_) * 2.0 * pr * (1.0 - pr)
                ) / tot_w

            gain = np.where(ok, gain, -1.0)
            k = int(np.argmax(gain))
            if gain[k] > best[0]:
                best = (float(gain[k]), int(j), int(k))

        if best[1] < 0 or best[0] <= 1e-7:
            return node

        best_gain, j, k = best
        mask = B[idx, j] <= k
        self.feat[node], self.thr[node] = j, k

        # Lưu importance giảm impurity
        self._importances_raw[j] += (len(idx) / self.n_samples_) * best_gain

        l = self._grow(B, y, w, idx[mask], depth + 1)
        r = self._grow(B, y, w, idx[~mask], depth + 1)
        self.left[node], self.right[node] = l, r
        return node

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        B = self.bm_.transform(X)
        node = np.zeros(len(B), dtype=int)
        while True:
            f = self.feat[node]
            leaf = f < 0
            if leaf.all():
                break
            go = np.where(leaf, 0, B[np.arange(len(B)), np.where(leaf, 0, f)] <= self.thr[node])
            node = np.where(leaf, node, np.where(go, self.left[node], self.right[node]))
        return self.val[node]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def threshold_value(self, node):
        """Trả về giá trị ngưỡng thực tế (float) phục vụ web và json."""
        j, k = self.feat[node], self.thr[node]
        if j < 0 or k < 0:
            return None
        e = self.bm_.edges_[j]
        return float(e[k]) if k < len(e) else float('inf')





# ==============================================================================
# 3. HỒI QUY LOGISTIC THUẦN NUMPY (GRADIENT DESCENT, L2, CLASS_WEIGHT)
# ==============================================================================

class LogisticRegression:
    """Hồi quy Logistic tối ưu hóa bằng Gradient Descent thuần NumPy."""
    def __init__(
        self,
        lr=0.05,
        epochs=400,
        l2_reg=1.0,
        class_weight=None,
        random_state=42,
    ):
        self.lr = lr
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.class_weight = class_weight
        self.random_state = random_state
        self.weights_ = None
        self.loss_history_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape

        # Thêm cột bias
        X_b = np.c_[np.ones(n), X]
        self.weights_ = rng.normal(0, 0.01, size=d + 1)
        self.loss_history_ = []

        # Trọng số mẫu
        w = np.ones(n, dtype=float)
        if self.class_weight == 'balanced':
            n1 = float(y.sum())
            n0 = float(n - n1)
            w = np.where(y == 1, n / (2.0 * max(n1, 1.0)), n / (2.0 * max(n0, 1.0)))

        for _ in range(self.epochs):
            z = np.clip(X_b @ self.weights_, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-z))

            # Cross entropy loss có trọng số + L2
            p_safe = np.clip(p, 1e-15, 1.0 - 1e-15)
            loss = -np.mean(w * (y * np.log(p_safe) + (1.0 - y) * np.log(1.0 - p_safe)))
            loss += (self.l2_reg / (2.0 * n)) * np.sum(self.weights_[1:] ** 2)
            self.loss_history_.append(float(loss))

            # Gradient
            grad = (X_b.T @ (w * (p - y))) / n
            grad[1:] += (self.l2_reg / n) * self.weights_[1:]

            self.weights_ -= self.lr * grad

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        X_b = np.c_[np.ones(len(X)), X]
        z = np.clip(X_b @ self.weights_, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ==============================================================================
# 4. HỆ THỐNG ĐỘ ĐO LƯỜNG THUẦN NUMPY (METRICS)
# ==============================================================================

def accuracy_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return np.array([[tn, fp], [fn, tp]])


def precision_score(y_true, y_pred, zero_division=0.0):
    cm = confusion_matrix(y_true, y_pred)
    tp, fp = cm[1, 1], cm[0, 1]
    denom = tp + fp
    return float(tp / denom) if denom > 0 else float(zero_division)


def recall_score(y_true, y_pred, zero_division=0.0):
    cm = confusion_matrix(y_true, y_pred)
    tp, fn = cm[1, 1], cm[1, 0]
    denom = tp + fn
    return float(tp / denom) if denom > 0 else float(zero_division)


def f1_score(y_true, y_pred, zero_division=0.0):
    p = precision_score(y_true, y_pred, zero_division=0.0)
    r = recall_score(y_true, y_pred, zero_division=0.0)
    denom = p + r
    return float((2.0 * p * r) / denom) if denom > 0 else float(zero_division)


def brier_score_loss(y_true, y_prob):
    y_true, y_prob = np.asarray(y_true, dtype=float), np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def roc_auc_score(y_true, y_score):
    """Tính ROC-AUC thuần NumPy qua xếp hạng Mann-Whitney U với xử lý đồng hạng (ties)."""
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score, dtype=float)
    n1 = y_true.sum()
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return 0.5

    order = np.argsort(y_score)
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)

    sorted_score = y_score[order]
    dup_mask = np.r_[False, sorted_score[1:] == sorted_score[:-1]]
    if dup_mask.any():
        _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
        cum_counts = np.cumsum(counts)
        start_ranks = np.r_[0, cum_counts[:-1]] + 1
        end_ranks = cum_counts
        avg_ranks = (start_ranks + end_ranks) / 2.0
        ranks = avg_ranks[inv]

    sum_r1 = ranks[y_true].sum()
    u = sum_r1 - (n1 * (n1 + 1)) / 2.0
    return float(u / (n1 * n0))


def average_precision_score(y_true, y_score):
    """Tính PR-AUC (Average Precision) thuần NumPy theo công thức phân đoạn chuẩn của sklearn."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    desc_order = np.argsort(y_score, kind='mergesort')[::-1]
    y_true_sorted = y_true[desc_order]
    y_score_sorted = y_score[desc_order]

    distinct_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, len(y_score_sorted) - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = (threshold_idxs + 1) - tps

    total_pos = tps[-1]
    if total_pos == 0:
        return 0.0

    precisions = tps / (tps + fps)
    recalls = tps / total_pos

    delta_recall = np.diff(np.r_[0.0, recalls])
    ap = np.sum(delta_recall * precisions)
    return float(ap)


# ==============================================================================
# 5. CROSS-VALIDATION SPLITTERS THUẦN NUMPY
# ==============================================================================

class TimeSeriesSplit:
    """Expanding-window time series cross-validator thuần NumPy."""
    def __init__(self, n_splits=4):
        self.n_splits = n_splits

    def split(self, X, y=None):
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        for i in range(1, self.n_splits + 1):
            train_end = fold_size * i
            val_end = min(train_end + fold_size, n)
            if i == self.n_splits:
                val_end = n
            yield np.arange(0, train_end), np.arange(train_end, val_end)


class StratifiedKFold:
    """Stratified K-Fold cross-validator thuần NumPy."""
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        idx0 = np.where(y == 0)[0].copy()
        idx1 = np.where(y == 1)[0].copy()

        if self.shuffle:
            rng.shuffle(idx0)
            rng.shuffle(idx1)

        folds0 = np.array_split(idx0, self.n_splits)
        folds1 = np.array_split(idx1, self.n_splits)

        for i in range(self.n_splits):
            val_idx = np.sort(np.concatenate([folds0[i], folds1[i]]))
            train_parts0 = [folds0[j] for j in range(self.n_splits) if j != i]
            train_parts1 = [folds1[j] for j in range(self.n_splits) if j != i]
            train_idx = np.sort(np.concatenate(train_parts0 + train_parts1))
            yield train_idx, val_idx


# ==============================================================================
# 6. GRID SEARCH THUẦN NUMPY
# ==============================================================================

class GridSearch:
    """Tìm kiếm siêu tham số qua Cross-Validation thuần NumPy."""
    def __init__(self, estimator_cls, param_grid, cv=None, scoring='pr_auc'):
        self.estimator_cls = estimator_cls
        self.param_grid = param_grid
        self.cv = cv if cv is not None else TimeSeriesSplit(n_splits=4)
        self.scoring = scoring
        self.best_params_ = None
        self.best_score_ = -float('inf')
        self.cv_results_ = []

    def _score(self, y_true, y_prob):
        if self.scoring == 'pr_auc':
            return average_precision_score(y_true, y_prob)
        if self.scoring == 'roc_auc':
            return roc_auc_score(y_true, y_prob)
        if self.scoring == 'f1':
            return f1_score(y_true, (y_prob >= 0.5).astype(int))
        raise ValueError(f"Chưa hỗ trợ hàm chấm điểm: {self.scoring}")

    def fit(self, X, y):
        import itertools
        keys = list(self.param_grid.keys())
        combos = list(itertools.product(*[self.param_grid[k] for k in keys]))

        best_score = -float('inf')
        best_params = None
        results = []

        splits = list(self.cv.split(X, y))

        for combo in combos:
            params = dict(zip(keys, combo))
            fold_scores = []
            for tr_idx, va_idx in splits:
                model = self.estimator_cls(**params)
                model.fit(X[tr_idx], y[tr_idx])
                preds = model.predict_proba(X[va_idx])
                s = self._score(y[va_idx], preds)
                fold_scores.append(s)

            mean_s = float(np.mean(fold_scores))
            std_s = float(np.std(fold_scores))
            results.append({
                'params': params,
                'mean_score': mean_s,
                'std_score': std_s,
                'fold_scores': fold_scores,
            })

            if mean_s > best_score:
                best_score = mean_s
                best_params = params

        self.best_score_ = best_score
        self.best_params_ = best_params
        self.cv_results_ = results
        return self

