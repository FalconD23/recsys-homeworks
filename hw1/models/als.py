import numpy as np


class ALSExplicit:
    def __init__(self, n_factors=16, n_iters=10, reg=0.1, seed=42):
        self.n_factors = n_factors
        self.n_iters = n_iters
        self.reg = reg
        self.seed = seed

    def fit(self, user_idx, item_idx, values, n_users, n_items):
        rng = np.random.default_rng(self.seed)
        scale = 0.1
        self.U = rng.normal(0.0, scale, size=(n_users, self.n_factors))
        self.V = rng.normal(0.0, scale, size=(n_items, self.n_factors))
        user_ids, user_slices = _groups(user_idx)
        item_ids, item_slices = _groups(item_idx)
        eye = np.eye(self.n_factors)
        for _ in range(self.n_iters):
            _update(self.U, self.V, item_idx, values, user_ids, user_slices, self.reg, eye)
            _update(self.V, self.U, user_idx, values, item_ids, item_slices, self.reg, eye)
        self.n_items = n_items
        return self

    def score_users(self, user_indices):
        return self.U[user_indices] @ self.V.T


def _groups(ids):
    order = np.argsort(ids, kind="mergesort")
    sorted_ids = ids[order]
    uniq, starts, counts = np.unique(sorted_ids, return_index=True, return_counts=True)
    slices = [order[start : start + count] for start, count in zip(starts, counts)]
    return uniq, slices


def _update(target, other, paired_ids, values, row_ids, slices, reg, eye):
    for row_id, idx in zip(row_ids, slices):
        cols = paired_ids[idx]
        y = values[idx]
        matrix = other[cols]
        gram = matrix.T @ matrix
        gram = gram + reg * eye
        target[row_id] = np.linalg.solve(gram, matrix.T @ y)
