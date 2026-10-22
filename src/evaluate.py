"""Offline evaluation of the recommender: time-based hold-out, Precision@K, Recall@K, coverage."""
import numpy as np
import pandas as pd

from .recommender import (CONF, build_matrix, item_similarity, itemcf_scores,
                          popularity_ranking, train_als)


def time_split(prod, test_days=90):
    """Train = everything before the last `test_days` days, test = the last `test_days` days."""
    cutoff = prod["InvoiceDate"].max() - pd.Timedelta(days=test_days)
    return prod[prod["InvoiceDate"] <= cutoff], prod[prod["InvoiceDate"] > cutoff]


def precision_recall_at_k(rec_fn, test_items, n_items, k=10):
    """Average Precision@k, Recall@k over test customers, and catalogue coverage."""
    p = r = 0.0
    seen = set()
    for u, truth in test_items.items():
        recs = rec_fn(u)[:k]
        hits = len(set(recs) & truth)
        p += hits / k
        r += hits / len(truth)
        seen.update(recs)
    n = len(test_items)
    return p / n, r / n, len(seen) / n_items


def run_offline_evaluation(prod, test_days=90, k=10):
    """Compare popularity, item-based CF and ALS on a time-based hold-out."""
    train, test = time_split(prod, test_days)
    users = np.sort(train["Customer ID"].unique())
    items = np.sort(train["StockCode"].unique())
    X, u2i, s2i = build_matrix(train, users, items)
    Xc = (X * CONF).tocsr()

    # Ground truth: products each known customer bought during the test window
    test_items = {u2i[u]: {s2i[s] for s in g if s in s2i}
                  for u, g in test.groupby("Customer ID")["StockCode"] if u in u2i}
    test_items = {u: t for u, t in test_items.items() if t}

    pop = popularity_ranking(X)
    S = item_similarity(X)
    als = train_als(Xc)

    models = {
        "Popularity (baseline)": lambda u: pop,
        "Item-based CF": lambda u: list(np.argsort(-itemcf_scores(X, S, u))),
        "ALS": lambda u: list(als.recommend(u, Xc[u], N=k, filter_already_liked_items=False)[0]),
    }
    rows = {name: precision_recall_at_k(fn, test_items, len(items), k) for name, fn in models.items()}
    out = pd.DataFrame(rows, index=[f"Precision@{k}", f"Recall@{k}", "Coverage"]).T.round(3)
    out.attrs["info"] = (f"train customers={len(users)}, products={len(items)}, "
                         f"test customers={len(test_items)}, test window={test_days} days")
    return out
