"""Customer segmentation: scaling, choosing k, model comparison, segment naming, UMAP."""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

FEATURES = ["Recency", "Frequency", "Monetary"]
SEED = 42


def log_scale(rfm):
    """log1p (RFM is heavily right-skewed) then StandardScaler. Returns (X, fitted scaler)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(np.log1p(rfm[FEATURES]))
    return X, scaler


def evaluate_k(X, k_range=range(2, 9)):
    """Inertia (elbow) and silhouette score of K-Means for each k."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def fit_kmeans(X, k):
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
    return km, km.labels_


def compare_models(X, k, dbscan_eps=(0.3, 0.4, 0.5), min_samples=10):
    """K-Means vs Gaussian Mixture vs DBSCAN: clusters, noise %, silhouette."""
    rows = []
    km_labels = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(X)
    rows.append({"model": f"K-Means (k={k})", "clusters": k, "noise_%": 0.0,
                 "silhouette": silhouette_score(X, km_labels)})
    gm_labels = GaussianMixture(n_components=k, random_state=SEED, n_init=3).fit_predict(X)
    rows.append({"model": f"Gaussian Mixture (k={k})", "clusters": k, "noise_%": 0.0,
                 "silhouette": silhouette_score(X, gm_labels)})
    for eps in dbscan_eps:
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        mask = labels != -1
        n_clusters = len(set(labels[mask]))
        sil = silhouette_score(X[mask], labels[mask]) if n_clusters > 1 else np.nan
        rows.append({"model": f"DBSCAN (eps={eps}, min_samples={min_samples})", "clusters": n_clusters,
                     "noise_%": 100 * (~mask).mean(), "silhouette": sil})
    return pd.DataFrame(rows)


def name_segments(rfm, cluster_col="Cluster"):
    """Rule-based names from cluster means (so names stay correct if labels are re-ordered).

    For k=4:  highest Monetary -> Champions; of the rest, highest Recency -> Lost;
              of the remaining two, higher Recency -> At Risk, the other -> New Customers.
    Other k values get generic names.
    """
    means = rfm.groupby(cluster_col)[FEATURES].mean()
    if len(means) != 4:
        return {c: f"Segment {c}" for c in means.index}
    names = {}
    champions = means["Monetary"].idxmax(); names[champions] = "Champions"
    rest = means.drop(champions)
    lost = rest["Recency"].idxmax(); names[lost] = "Lost"
    rest = rest.drop(lost)
    at_risk = rest["Recency"].idxmax(); names[at_risk] = "At Risk"
    names[rest["Recency"].idxmin()] = "New Customers"
    return names


def segment_profile(rfm, segment_col="Segment"):
    """Size, share, average R/F/M and share of total spend per segment."""
    g = rfm.groupby(segment_col)
    prof = g.agg(customers=("CustomerID", "count"), avg_recency_days=("Recency", "mean"),
                 avg_orders=("Frequency", "mean"), avg_spend=("Monetary", "mean"),
                 total_spend=("Monetary", "sum"))
    prof["customers_%"] = 100 * prof["customers"] / prof["customers"].sum()
    prof["spend_%"] = 100 * prof["total_spend"] / prof["total_spend"].sum()
    return prof.drop(columns="total_spend").round(1)


def umap_embedding(X, n_neighbors=15, min_dist=0.1):
    """2D UMAP coordinates for the customer map (visualisation only, not used for clustering)."""
    import umap
    return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=SEED).fit_transform(X)
