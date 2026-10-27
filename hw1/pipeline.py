import ast
import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

from models.als import ALSExplicit
from models.als_custom import ALSCustom
from models.ease import EASE, interaction_gram
from models.fm import FactorizationMachine
from models.item_knn import ItemKNN
from models.metrics import evaluate_model

SEED = 42
K = 10
MIN_USER_REVIEWS = 5
RELEVANT_RATING = 4.0

REVIEWS_URL = "https://huggingface.co/datasets/cmulgy/beauty/resolve/main/reviews_Beauty_5.json.gz"
REVIEWS_URL_SNAP = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz"
META_URL = "https://huggingface.co/datasets/cmulgy/beauty/resolve/main/meta_Beauty.json.gz"
META_URL_CSV = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/meta_Beauty.csv.gz"
SOURCE_PAGE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/datasets.html"


def download(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path


def _ready(path, min_bytes):
    path = Path(path)
    return path.exists() and path.stat().st_size >= min_bytes


def ensure_data(data_dir):
    data_dir = Path(data_dir)
    reviews = data_dir / "reviews_Beauty_5.json.gz"
    meta_json = data_dir / "meta_Beauty.json.gz"
    meta_csv = data_dir / "meta_Beauty.csv.gz"
    if not _ready(reviews, 40_000_000):
        try:
            download(REVIEWS_URL, reviews)
        except Exception:
            download(REVIEWS_URL_SNAP, reviews)
    if _ready(meta_json, 90_000_000):
        return reviews, meta_json
    if _ready(meta_csv, 8_700_000):
        return reviews, meta_csv
    try:
        download(META_URL, meta_json)
        return reviews, meta_json
    except Exception:
        download(META_URL_CSV, meta_csv)
        return reviews, meta_csv


def load_reviews(path):
    rows = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            helpful = record.get("helpful") or [0, 0]
            rows.append(
                (
                    record["reviewerID"],
                    record["asin"],
                    float(record["overall"]),
                    int(record["unixReviewTime"]),
                    float(helpful[0]),
                )
            )
    return pl.DataFrame(
        rows,
        schema=["user_id", "item_id", "rating", "timestamp", "helpful_up"],
        orient="row",
    )


def load_meta(path):
    path = Path(path)
    if "json" in path.name:
        rows = []
        with gzip.open(path, "rt") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    record = ast.literal_eval(line)
                categories = record.get("categories")
                if not isinstance(categories, str):
                    categories = repr(categories)
                rows.append((record.get("asin"), record.get("title") or "", categories))
        return pl.DataFrame(
            rows,
            schema=["item_id", "title", "categories"],
            orient="row",
        )
    meta = pl.read_csv(
        path,
        has_header=False,
        ignore_errors=True,
        truncate_ragged_lines=True,
        infer_schema_length=0,
    )
    names = meta.columns
    meta = meta.rename({names[0]: "item_id", names[1]: "title", names[2]: "categories"})
    return meta.select(["item_id", "title", "categories"])


def category_name(text):
    if text is None or text == "":
        return "unknown"
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return "unknown"
    if not parsed:
        return "unknown"
    path = parsed[0] if isinstance(parsed[0], list) else parsed
    if not isinstance(path, list) or len(path) == 0:
        return "unknown"
    if len(path) >= 2:
        return str(path[1])
    return str(path[0])


def aggregate_reviews(reviews):
    ordered = reviews.sort(["user_id", "item_id", "timestamp"])
    return (
        ordered.group_by(["user_id", "item_id"], maintain_order=True)
        .agg(
            pl.col("rating").mean().alias("rating"),
            pl.col("timestamp").max().alias("timestamp"),
            pl.col("helpful_up").last().alias("helpful_up"),
        )
    )


def keep_active_users(interactions, min_reviews):
    counts = interactions.group_by("user_id").len()
    active = counts.filter(pl.col("len") >= min_reviews).select("user_id")
    return interactions.join(active, on="user_id", how="inner")


def attach_categories(interactions, meta):
    meta = meta.unique(subset=["item_id"], keep="first")
    names = [category_name(text) for text in meta["categories"].to_list()]
    meta = meta.with_columns(pl.Series("category", names))
    return interactions.join(
        meta.select(["item_id", "title", "category"]),
        on="item_id",
        how="left",
    ).with_columns(pl.col("category").fill_null("unknown"), pl.col("title").fill_null(""))


def temporal_split(interactions):
    ranked = interactions.sort(["user_id", "timestamp"]).with_columns(
        pl.col("timestamp").rank(method="ordinal", descending=True).over("user_id").alias("recency")
    )
    test = ranked.filter(pl.col("recency") == 1).drop("recency")
    validation = ranked.filter(pl.col("recency") == 2).drop("recency")
    train = ranked.filter(pl.col("recency") > 2).drop("recency")
    return train, validation, test


def _codes(values, mapping):
    return np.array([mapping[value] for value in values], dtype=np.int32)


def build_matrices(train, validation, test):
    users = train["user_id"].unique().sort().to_list()
    items = (
        pl.concat([train, validation, test])
        .select("item_id")
        .unique()
        .sort("item_id")["item_id"]
        .to_list()
    )
    categories = (
        pl.concat([train, validation, test])
        .select(["item_id", "category"])
        .unique(subset=["item_id"])
        .sort("item_id")
    )
    user_map = {user: index for index, user in enumerate(users)}
    item_map = {item: index for index, item in enumerate(items)}
    cat_names = categories["category"].unique().sort().to_list()
    if "unknown" in cat_names:
        cat_names = ["unknown"] + [name for name in cat_names if name != "unknown"]
    cat_map = {name: index for index, name in enumerate(cat_names)}
    item_categories = np.zeros(len(items), dtype=np.int32)
    for item_id, category in zip(categories["item_id"].to_list(), categories["category"].to_list()):
        item_categories[item_map[item_id]] = cat_map[category]

    def pack(frame):
        user_idx = _codes(frame["user_id"].to_list(), user_map)
        item_idx = _codes(frame["item_id"].to_list(), item_map)
        rating = frame["rating"].to_numpy().astype(np.float64)
        helpful = frame["helpful_up"].to_numpy().astype(np.float64)
        return user_idx, item_idx, rating, helpful

    train_pack = pack(train)
    item_to_cat = item_categories
    train_cats = item_to_cat[train_pack[1]]

    def seen_lists(frame):
        grouped = (
            frame.select(
                pl.Series("user_code", _codes(frame["user_id"].to_list(), user_map)),
                pl.Series("item_code", _codes(frame["item_id"].to_list(), item_map)),
            )
            .group_by("user_code")
            .agg(pl.col("item_code"))
        )
        mapping = {}
        for user_code, item_codes in zip(grouped["user_code"].to_list(), grouped["item_code"].to_list()):
            mapping[int(user_code)] = np.asarray(item_codes, dtype=np.int32)
        return mapping

    train_seen = seen_lists(train)
    train_val = pl.concat([train, validation])
    test_seen = seen_lists(train_val)

    def eval_pack(frame, seen_map):
        relevant = frame.filter(pl.col("rating") >= RELEVANT_RATING)
        users_idx = _codes(relevant["user_id"].to_list(), user_map)
        items_idx = _codes(relevant["item_id"].to_list(), item_map)
        seen = [seen_map.get(int(user), np.array([], dtype=np.int32)) for user in users_idx]
        return users_idx, items_idx, seen

    val_eval = eval_pack(validation, train_seen)
    test_eval = eval_pack(test, test_seen)
    return {
        "n_users": len(users),
        "n_items": len(items),
        "n_cats": len(cat_names),
        "cat_names": cat_names,
        "item_categories": item_categories,
        "train_user": train_pack[0],
        "train_item": train_pack[1],
        "train_rating": train_pack[2],
        "train_helpful": train_pack[3],
        "train_cat": train_cats,
        "val_eval": val_eval,
        "test_eval": test_eval,
    }


def quality_report(reviews):
    full_unique = reviews.unique().height
    pair_unique = reviews.select(["user_id", "item_id"]).unique().height
    nulls = {column: int(reviews[column].null_count()) for column in reviews.columns}
    return {
        "rows": reviews.height,
        "full_duplicate_rows": reviews.height - full_unique,
        "user_item_duplicate_rows": reviews.height - pair_unique,
        "nulls": nulls,
        "users": reviews["user_id"].n_unique(),
        "items": reviews["item_id"].n_unique(),
    }


def sparsity(n_rows, n_users, n_items):
    return 1.0 - n_rows / (n_users * n_items)


def _fit_item_knn(bundle, k):
    model = ItemKNN(k=k)
    model.fit(
        bundle["train_user"],
        bundle["train_item"],
        bundle["train_rating"],
        bundle["n_users"],
        bundle["n_items"],
    )
    return model


def _fit_als(bundle, rank):
    model = ALSExplicit(n_factors=rank, n_iters=10, reg=0.1, seed=SEED)
    model.fit(
        bundle["train_user"],
        bundle["train_item"],
        bundle["train_rating"],
        bundle["n_users"],
        bundle["n_items"],
    )
    return model


def _fit_fm(bundle):
    model = FactorizationMachine(n_factors=16, n_epochs=5, lr=0.01, batch_size=4096, seed=SEED)
    model.fit(
        bundle["train_user"],
        bundle["train_item"],
        bundle["train_cat"],
        bundle["train_rating"],
        bundle["n_users"],
        bundle["n_items"],
        bundle["n_cats"],
        bundle["item_categories"],
    )
    return model


def _fit_custom(bundle, rank):
    model = ALSCustom(n_factors=rank, n_iters=10, reg=0.1, seed=SEED)
    model.fit(
        bundle["train_user"],
        bundle["train_item"],
        bundle["train_rating"],
        bundle["train_helpful"],
        bundle["n_users"],
        bundle["n_items"],
    )
    return model


def _score(model, pack):
    users, targets, seen = pack
    return evaluate_model(model, users, targets, seen, k=K)


def select_and_test(bundle):
    rows = []
    best = {}

    knn50 = _fit_item_knn(bundle, 50)
    for k, model in ((50, knn50), (20, ItemKNN(k=20))):
        if k == 20:
            model.matrix = knn50.matrix
            model.similarity = _topk_similarity(knn50.similarity, 20)
            model.n_items = knn50.n_items
        metrics = _score(model, bundle["val_eval"])
        params = f"k={k}"
        print(f"ItemKNN {params} validation NDCG@10={metrics['NDCG@10']:.4f}", flush=True)
        rows.append({"model": "ItemKNN", "params": params, "split": "validation", **metrics})
        if "ItemKNN" not in best or metrics["NDCG@10"] > best["ItemKNN"][0]:
            best["ItemKNN"] = (metrics["NDCG@10"], f"k={k}", model)

    for rank in (16, 32):
        model = _fit_als(bundle, rank)
        metrics = _score(model, bundle["val_eval"])
        params = f"rank={rank}, reg=0.1, iters=10"
        print(f"ALS {params} validation NDCG@10={metrics['NDCG@10']:.4f}", flush=True)
        rows.append({"model": "ALS", "params": params, "split": "validation", **metrics})
        if "ALS" not in best or metrics["NDCG@10"] > best["ALS"][0]:
            best["ALS"] = (metrics["NDCG@10"], params, model)

    matrix, gram = interaction_gram(
        bundle["train_user"],
        bundle["train_item"],
        bundle["train_rating"],
        bundle["n_users"],
        bundle["n_items"],
    )
    for lam in (50.0, 500.0):
        model = EASE(lam=lam).fit_from_gram(matrix, gram, lam)
        metrics = _score(model, bundle["val_eval"])
        params = f"lambda={lam:g}"
        print(f"EASE {params} validation NDCG@10={metrics['NDCG@10']:.4f}", flush=True)
        rows.append({"model": "EASE", "params": params, "split": "validation", **metrics})
        if "EASE" not in best or metrics["NDCG@10"] > best["EASE"][0]:
            best["EASE"] = (metrics["NDCG@10"], params, model)
    del gram

    fm = _fit_fm(bundle)
    fm_params = "rank=16, epochs=5, lr=0.01, batch=4096"
    metrics = _score(fm, bundle["val_eval"])
    print(f"FM {fm_params} validation NDCG@10={metrics['NDCG@10']:.4f}", flush=True)
    rows.append({"model": "FM", "params": fm_params, "split": "validation", **metrics})
    best["FM"] = (metrics["NDCG@10"], fm_params, fm)

    als_rank = 32 if "rank=32" in best["ALS"][1] else 16
    custom = _fit_custom(bundle, als_rank)
    custom_params = f"rank={als_rank}, reg=0.1, iters=10, target=rating*(1+log1p(helpful))"
    metrics = _score(custom, bundle["val_eval"])
    print(f"ALSCustom {custom_params} validation NDCG@10={metrics['NDCG@10']:.4f}", flush=True)
    rows.append({"model": "ALSCustom", "params": custom_params, "split": "validation", **metrics})
    best["ALSCustom"] = (metrics["NDCG@10"], custom_params, custom)

    for name, (_, params, model) in best.items():
        metrics = _score(model, bundle["test_eval"])
        rows.append({"model": name, "params": params, "split": "test", **metrics})
    return pl.DataFrame(rows), {name: params for name, (_, params, _) in best.items()}


def _topk_similarity(similarity, k):
    similarity = similarity.tolil()
    for row_id in range(similarity.shape[0]):
        values = similarity.data[row_id]
        if len(values) <= k:
            continue
        chosen = np.argpartition(np.asarray(values), -k)[-k:]
        similarity.rows[row_id] = [similarity.rows[row_id][index] for index in chosen]
        similarity.data[row_id] = [values[index] for index in chosen]
    return similarity.tocsr()
