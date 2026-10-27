import numpy as np


def hit_rate_and_ndcg(scores, target, seen, k=10):
    scores = np.array(scores, dtype=np.float64, copy=True)
    n_users = scores.shape[0]
    for i in range(n_users):
        scores[i, seen[i]] = -np.inf
    top_idx = np.argpartition(scores, -k, axis=1)[:, -k:]
    top_scores = np.take_along_axis(scores, top_idx, axis=1)
    order = np.argsort(-top_scores, axis=1)
    top_idx = np.take_along_axis(top_idx, order, axis=1)
    target = np.asarray(target)
    hits = top_idx == target[:, None]
    hit = hits.any(axis=1)
    pos = hits.argmax(axis=1)
    ndcg = np.zeros(n_users, dtype=np.float64)
    ndcg[hit] = 1.0 / np.log2(pos[hit] + 2)
    return float(hit.mean()), float(ndcg.mean()), int(n_users)


def evaluate_model(model, user_ids, targets, seen, k=10, batch=1024):
    hit_sum = 0.0
    ndcg_sum = 0.0
    total = 0
    for start in range(0, len(user_ids), batch):
        stop = start + batch
        scores = model.score_users(user_ids[start:stop])
        hit, ndcg, n_users = hit_rate_and_ndcg(
            scores, targets[start:stop], seen[start:stop], k=k
        )
        hit_sum += hit * n_users
        ndcg_sum += ndcg * n_users
        total += n_users
    if total == 0:
        return {"HitRate@10": 0.0, "NDCG@10": 0.0, "users": 0}
    return {
        "HitRate@10": hit_sum / total,
        "NDCG@10": ndcg_sum / total,
        "users": total,
    }
