import numpy as np
from scipy import sparse


class ItemKNN:
    def __init__(self, k=50, block=512):
        self.k = k
        self.block = block

    def fit(self, user_idx, item_idx, values, n_users, n_items):
        matrix = sparse.csr_matrix(
            (values.astype(np.float64), (user_idx, item_idx)),
            shape=(n_users, n_items),
        )
        norms = np.sqrt(np.asarray(matrix.power(2).sum(axis=0))).ravel()
        norms[norms == 0.0] = 1.0
        normalized = matrix @ sparse.diags(1.0 / norms)
        rows = []
        cols = []
        data = []
        item_vectors = normalized.T.tocsr()
        neighbor_k = min(self.k, n_items - 1)
        for start in range(0, n_items, self.block):
            stop = min(start + self.block, n_items)
            sims = (item_vectors[start:stop] @ normalized).toarray()
            for local, item_id in enumerate(range(start, stop)):
                sims[local, item_id] = -np.inf
            chosen = np.argpartition(sims, -neighbor_k, axis=1)[:, -neighbor_k:]
            for local, item_id in enumerate(range(start, stop)):
                neigh = chosen[local]
                vals = sims[local, neigh]
                mask = np.isfinite(vals) & (vals > 0.0)
                neigh = neigh[mask]
                vals = vals[mask]
                rows.append(np.full(neigh.shape[0], item_id, dtype=np.int32))
                cols.append(neigh.astype(np.int32))
                data.append(vals)
        similarity = sparse.csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_items, n_items),
        )
        self.similarity = similarity
        self.matrix = matrix
        self.n_items = n_items
        return self

    def score_users(self, user_indices):
        return (self.matrix[user_indices] @ self.similarity).toarray()
