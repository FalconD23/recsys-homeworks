import numpy as np

from models.als import ALSExplicit


def helpful_target(rating, helpful_up):
    return rating * (1.0 + np.log1p(helpful_up))


class ALSCustom:
    def __init__(self, n_factors=16, n_iters=10, reg=0.1, seed=42):
        self.als = ALSExplicit(
            n_factors=n_factors, n_iters=n_iters, reg=reg, seed=seed
        )

    def fit(self, user_idx, item_idx, rating, helpful_up, n_users, n_items):
        target = helpful_target(rating, helpful_up)
        self.als.fit(user_idx, item_idx, target, n_users, n_items)
        return self

    def score_users(self, user_indices):
        return self.als.score_users(user_indices)
