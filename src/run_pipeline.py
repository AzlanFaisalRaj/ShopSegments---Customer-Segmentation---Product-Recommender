"""End-to-end pipeline: Excel -> clean data -> RFM -> segments -> recommender -> saved models + reports.

Usage (from the repo root):
    python -m src.run_pipeline --excel data/raw/online_retail_II.xlsx
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import data_prep, evaluate, recommender, segmentation

K_FINAL = 4   # chosen for business actionability, see README


def main(excel, models_dir, processed_dir, reports_dir):
    models_dir, processed_dir, reports_dir = map(Path, (models_dir, processed_dir, reports_dir))
    for d in (models_dir, processed_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("1/6 Loading and cleaning data ...")
    raw = data_prep.load_data(excel)
    df, log = data_prep.clean_transactions(raw, return_log=True)
    log.to_csv(reports_dir / "cleaning_log.csv", index=False)
    df.to_parquet(processed_dir / "clean_transactions.parquet", index=False)
    print(log.to_string(index=False))

    print("2/6 Building RFM ...")
    rfm = data_prep.build_rfm(df)
    rfm.to_parquet(processed_dir / "rfm.parquet", index=False)
    print("customers:", len(rfm))

    print("3/6 Scaling and choosing k ...")
    X, scaler = segmentation.log_scale(rfm)
    k_table = segmentation.evaluate_k(X)
    k_table.round(3).to_csv(reports_dir / "cluster_selection.csv", index=False)
    print(k_table.round(3).to_string(index=False))
    comparison = segmentation.compare_models(X, K_FINAL)
    comparison.round(3).to_csv(reports_dir / "model_comparison.csv", index=False)
    print(comparison.round(3).to_string(index=False))

    print("4/6 Final segments + UMAP ...")
    kmeans, labels = segmentation.fit_kmeans(X, K_FINAL)
    rfm["Cluster"] = labels
    rfm["Segment"] = rfm["Cluster"].map(segmentation.name_segments(rfm))
    emb = segmentation.umap_embedding(X)
    rfm["UMAP1"], rfm["UMAP2"] = emb[:, 0].astype("float32"), emb[:, 1].astype("float32")
    profile = segmentation.segment_profile(rfm)
    profile.to_csv(reports_dir / "segment_profile.csv")
    print(profile.to_string())

    print("5/6 Offline evaluation of recommenders ...")
    prod = data_prep.product_rows(df)
    metrics = evaluate.run_offline_evaluation(prod)
    metrics.to_csv(reports_dir / "recommender_metrics.csv")
    print(metrics.attrs["info"]); print(metrics.to_string())

    print("6/6 Training final ALS and saving artifacts ...")
    users, items = np.sort(prod["Customer ID"].unique()), np.sort(prod["StockCode"].unique())
    Xb, customer_lookup, code_to_idx = recommender.build_matrix(prod, users, items)
    Xc = (Xb * recommender.CONF).tocsr()
    als = recommender.train_als(Xc)
    desc = recommender.build_descriptions(prod)

    joblib.dump(rfm, models_dir / "rfm_clustered_df.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(kmeans, models_dir / "kmeans_model.pkl")
    joblib.dump(Xc, models_dir / "user_item_matrix.pkl")
    joblib.dump(customer_lookup, models_dir / "customer_lookup.pkl")
    joblib.dump(dict(enumerate(items)), models_dir / "cat_to_stockcode.pkl")
    joblib.dump(code_to_idx, models_dir / "stockcode_to_idx.pkl")
    joblib.dump(desc, models_dir / "stockcode_to_desc.pkl")
    joblib.dump(recommender.cold_start_table(prod, desc), models_dir / "cold_start_recs.pkl")
    joblib.dump(recommender.segment_top_products(prod, rfm, desc), models_dir / "segment_top_products.pkl")
    np.savez_compressed(models_dir / "als_factors.npz", user_factors=np.asarray(als.user_factors),
                        item_factors=np.asarray(als.item_factors))
    print("Done. Models in", models_dir, "| reports in", reports_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", default="data/raw/online_retail_II.xlsx")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--reports-dir", default="reports")
    a = ap.parse_args()
    main(a.excel, a.models_dir, a.processed_dir, a.reports_dir)
