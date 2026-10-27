import numpy as np
import torch


class FactorizationMachine:
    def __init__(self, n_factors=16, n_epochs=5, lr=0.01, batch_size=4096, seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, user_idx, item_idx, cat_idx, values, n_users, n_items, n_cats, item_categories):
        torch.manual_seed(self.seed)
        self.item_categories = torch.as_tensor(item_categories, dtype=torch.long)
        self.w0 = torch.nn.Parameter(torch.tensor([float(values.mean())]))
        self.user_b = torch.nn.Embedding(n_users, 1)
        self.item_b = torch.nn.Embedding(n_items, 1)
        self.cat_b = torch.nn.Embedding(n_cats, 1)
        self.user_v = torch.nn.Embedding(n_users, self.n_factors)
        self.item_v = torch.nn.Embedding(n_items, self.n_factors)
        self.cat_v = torch.nn.Embedding(n_cats, self.n_factors)
        for embedding in (self.user_b, self.item_b, self.cat_b):
            torch.nn.init.zeros_(embedding.weight)
        for embedding in (self.user_v, self.item_v, self.cat_v):
            torch.nn.init.normal_(embedding.weight, std=0.01)
        params = [
            self.w0,
            self.user_b.weight,
            self.item_b.weight,
            self.cat_b.weight,
            self.user_v.weight,
            self.item_v.weight,
            self.cat_v.weight,
        ]
        optimizer = torch.optim.Adam(params, lr=self.lr)
        users = torch.as_tensor(user_idx, dtype=torch.long)
        items = torch.as_tensor(item_idx, dtype=torch.long)
        cats = torch.as_tensor(cat_idx, dtype=torch.long)
        target = torch.as_tensor(values, dtype=torch.float32)
        n_obs = target.shape[0]
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(self.n_epochs):
            order = torch.randperm(n_obs, generator=generator)
            for start in range(0, n_obs, self.batch_size):
                batch = order[start : start + self.batch_size]
                pred = self._predict(users[batch], items[batch], cats[batch])
                loss = torch.mean((pred - target[batch]) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.n_items = n_items
        return self

    def _predict(self, users, items, cats):
        user_v = self.user_v(users)
        item_v = self.item_v(items)
        cat_v = self.cat_v(cats)
        return (
            self.w0
            + self.user_b(users).squeeze(-1)
            + self.item_b(items).squeeze(-1)
            + self.cat_b(cats).squeeze(-1)
            + torch.sum(user_v * item_v, dim=1)
            + torch.sum(user_v * cat_v, dim=1)
            + torch.sum(item_v * cat_v, dim=1)
        )

    def score_users(self, user_indices):
        with torch.no_grad():
            users = torch.as_tensor(user_indices, dtype=torch.long)
            item_v = self.item_v.weight
            cat_v = self.cat_v.weight[self.item_categories]
            item_bias = self.item_b.weight.squeeze(-1)
            cat_bias = self.cat_b.weight.squeeze(-1)[self.item_categories]
            item_part = item_bias + cat_bias + torch.sum(item_v * cat_v, dim=1)
            chunks = []
            for start in range(0, users.shape[0], 512):
                batch = users[start : start + 512]
                user_v = self.user_v(batch)
                user_bias = self.user_b(batch).squeeze(-1)
                scores = (
                    self.w0
                    + user_bias[:, None]
                    + item_part[None, :]
                    + user_v @ item_v.T
                    + user_v @ cat_v.T
                )
                chunks.append(scores.numpy())
        return np.vstack(chunks)
