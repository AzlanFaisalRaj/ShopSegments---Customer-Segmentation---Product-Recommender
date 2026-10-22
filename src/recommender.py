"""Recommender: user-item matrix, ALS, item-based CF, popularity cold start, serving class."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.preprocessing import normalize

CONF = 40   # ALS confidence multiplier on the binary purchase matrix
SEED = 42


def build_matrix(data, users, items):
    """USERS x ITEMS binary matrix (1 = customer bought the product). Returns (X, user->row, item->col)."""
    u2i = {u: i for i, u in enumerate(users)}
    s2i = {s: i for i, s in enumerate(items)}
    d = data.drop_duplicates(["Customer ID", "StockCode"])
    d = d[d["Customer ID"].isin(u2i) & d["StockCode"].isin(s2i)]
    rows, cols = d["Customer ID"].map(u2i), d["StockCode"].map(s2i)
    X = sparse.csr_matrix((np.ones(len(d)), (rows, cols)), shape=(len(users), len(items)))
    return X, u2i, s2i


def train_als(Xc, factors=64, regularization=0.1, iterations=20):
    """Fit implicit ALS. Xc must be USERS x ITEMS (do not transpose)."""
    from implicit.als import AlternatingLeastSquares
    model = AlternatingLeastSquares(factors=factors, regularization=regularization,
                                    iterations=iterations, random_state=SEED)
    model.fit(Xc, show_progress=False)
    assert model.user_factors.shape[0] == Xc.shape[0], "ALS orientation is wrong (users/items swapped)"
    return model


def popularity_ranking(X):
    """Item indices ordered by number of distinct buyers (baseline and cold-start source)."""
    return list(np.argsort(-np.asarray(X.sum(0)).ravel()))


def item_similarity(X):
    """Item-item cosine similarity matrix (item-based collaborative filtering)."""
    Xn = normalize(X.T.tocsr())
    return (Xn @ Xn.T).tocsr()


def itemcf_scores(X, S, user_row):
    """Score all items for one user by summing similarity to items the user bought."""
    row = X[user_row]
    scores = np.asarray(S[row.indices].sum(0)).ravel()
    scores[row.indices] = -1          # do not re-recommend (used for the item-CF baseline)
    return scores


def build_descriptions(prod):
    """StockCode -> product description."""
    pm = prod.dropna(subset=["Description"]).drop_duplicates("StockCode")
    return dict(zip(pm["StockCode"], pm["Description"].str.strip()))


def cold_start_table(prod, stockcode_to_desc, n=10):
    """Top-n products by number of DISTINCT buyers (not quantity, so bulk orders do not bias it)."""
    t = (prod.groupby("StockCode")["Customer ID"].nunique()
             .sort_values(ascending=False).head(n).rename("Buyers").reset_index())
    t["Description"] = t["StockCode"].map(stockcode_to_desc)
    return t


def segment_top_products(prod, rfm, stockcode_to_desc, n=5):
    """Top-n products per segment by distinct buyers (for the dashboard segment cards)."""
    seg = rfm.set_index("CustomerID")["Segment"]
    p = prod.assign(Segment=prod["Customer ID"].map(seg)).dropna(subset=["Segment"])
    out = {}
    for name, g in p.groupby("Segment"):
        t = g.groupby("StockCode")["Customer ID"].nunique().nlargest(n).rename("Buyers").reset_index()
        t["Description"] = t["StockCode"].map(stockcode_to_desc)
        out[name] = t
    return out


class Recommender:
    """Serves recommendations from saved ALS factors (numpy only, no `implicit` needed)."""

    def __init__(self, user_factors, item_factors, matrix, customer_lookup,
                 idx_to_code, code_to_idx, descriptions, cold_start):
        self.user_f, self.item_f = user_factors, item_factors
        self.item_norm = item_factors / np.maximum(np.linalg.norm(item_factors, axis=1, keepdims=True), 1e-9)
        self.matrix, self.cust = matrix.tocsr(), customer_lookup
        self.idx_to_code, self.code_to_idx = idx_to_code, code_to_idx
        self.desc, self.cold = descriptions, cold_start

    @classmethod
    def from_dir(cls, models_dir):
        d = Path(models_dir)
        f = np.load(d / "als_factors.npz")
        return cls(f["user_factors"], f["item_factors"], joblib.load(d / "user_item_matrix.pkl"),
                   joblib.load(d / "customer_lookup.pkl"), joblib.load(d / "cat_to_stockcode.pkl"),
                   joblib.load(d / "stockcode_to_idx.pkl"), joblib.load(d / "stockcode_to_desc.pkl"),
                   joblib.load(d / "cold_start_recs.pkl"))

    def _name(self, code):
        return self.desc.get(code, f"Product {code}")

    def recommend(self, customer_id, n=10):
        """Top-n products for a customer. Unknown customers get the popular-products fallback."""
        idx = self.cust.get(int(customer_id))
        if idx is None:
            return self.cold.head(n)[["StockCode", "Description"]].assign(**{"Bought before": False})
        top = np.argsort(-(self.item_f @ self.user_f[idx]))[:n]
        bought = self.matrix[idx].toarray().ravel() > 0
        return pd.DataFrame({
            "StockCode": [self.idx_to_code[i] for i in top],
            "Description": [self._name(self.idx_to_code[i]) for i in top],
            "Bought before": [bool(bought[i]) for i in top],
        })

    def also_bought(self, stockcode, n=8):
        """Customers who bought this also bought (cosine similarity of ALS item factors)."""
        i = self.code_to_idx[str(stockcode)]
        sims = self.item_norm @ self.item_norm[i]
        top = [j for j in np.argsort(-sims) if j != i][:n]
        return pd.DataFrame({
            "StockCode": [self.idx_to_code[j] for j in top],
            "Description": [self._name(self.idx_to_code[j]) for j in top],
            "Similarity": [round(float(sims[j]), 2) for j in top],
        })
