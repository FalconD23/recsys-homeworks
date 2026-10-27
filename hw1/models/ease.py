import numpy as np
from scipy import sparse


class EASE:
    def __init__(self, lam=500.0):
        self.lam = lam

    def fit(self, user_idx, item_idx, values, n_users, n_items):
        matrix, gram = interaction_gram(user_idx, item_idx, values, n_users, n_items)
        self.fit_from_gram(matrix, gram, self.lam)
        return self

    def fit_from_gram(self, matrix, gram, lam):
        n_items = gram.shape[0]
        regularized = gram.copy()
        regularized.flat[:: n_items + 1] += lam
        precision = np.linalg.inv(regularized)
        diagonal = np.diag(precision).copy()
        weights = -precision / diagonal[:, None]
        np.fill_diagonal(weights, 0.0)
        self.weights = weights
        self.matrix = matrix
        self.lam = lam
        self.n_items = n_items
        return self

    def score_users(self, user_indices):
        return self.matrix[user_indices] @ self.weights


def interaction_gram(user_idx, item_idx, values, n_users, n_items):
    matrix = sparse.csr_matrix(
        (values.astype(np.float64), (user_idx, item_idx)),
        shape=(n_users, n_items),
    )
    gram = (matrix.T @ matrix).toarray()
    return matrix, gram
