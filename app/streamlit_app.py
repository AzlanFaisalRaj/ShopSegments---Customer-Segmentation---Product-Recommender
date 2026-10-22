"""ShopSegments: customer segments + product recommendations for a marketing team."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

MODELS = Path(__file__).resolve().parent.parent / "models"

st.set_page_config(page_title="ShopSegments", page_icon="🛍️", layout="wide")

# ---------- Business text per segment ----------
SEGMENT_INFO = {
    "Champions": {
        "what": "Best customers: bought recently, order often, spend the most.",
        "action": "Reward them: early access to new products, loyalty perks, ask for reviews and referrals. Avoid heavy discounts, they already buy.",
    },
    "New Customers": {
        "what": "Recently active, a few orders, medium spend. Still building the habit.",
        "action": "Welcome series, then nudge the next purchase with a small bundle offer and the best-sellers.",
    },
    "At Risk": {
        "what": "Used to spend well but have not bought for a while.",
        "action": "Win-back email: 'we miss you' with a time-limited offer on products they bought before.",
    },
    "Lost": {
        "what": "Long inactive, mostly a single small order.",
        "action": "One low-cost reactivation message (email/SMS). If no response, stop spending ad budget on them.",
    },
}
SEGMENT_ORDER = ["Champions", "New Customers", "At Risk", "Lost"]
COLORS = {"Champions": "#0f766e", "New Customers": "#2563eb", "At Risk": "#d97706", "Lost": "#b91c1c"}


# ---------- Load everything once ----------
@st.cache_resource
def load_artifacts():
    rfm = joblib.load(MODELS / "rfm_clustered_df.pkl")
    f = np.load(MODELS / "als_factors.npz")
    item_f = f["item_factors"]
    norm = item_f / np.maximum(np.linalg.norm(item_f, axis=1, keepdims=True), 1e-9)
    return {
        "rfm": rfm,
        "user_f": f["user_factors"],
        "item_f": item_f,
        "item_norm": norm,
        "matrix": joblib.load(MODELS / "user_item_matrix.pkl").tocsr(),
        "cust": joblib.load(MODELS / "customer_lookup.pkl"),
        "idx2code": joblib.load(MODELS / "cat_to_stockcode.pkl"),
        "code2idx": joblib.load(MODELS / "stockcode_to_idx.pkl"),
        "desc": joblib.load(MODELS / "stockcode_to_desc.pkl"),
        "cold": joblib.load(MODELS / "cold_start_recs.pkl"),
        "seg_top": joblib.load(MODELS / "segment_top_products.pkl"),
        "scaler": joblib.load(MODELS / "scaler.pkl"),
        "kmeans": joblib.load(MODELS / "kmeans_model.pkl"),
    }


A = load_artifacts()
rfm = A["rfm"]
cluster_to_segment = rfm.drop_duplicates("Cluster").set_index("Cluster")["Segment"].to_dict()


def product_name(code):
    return A["desc"].get(code, f"Product {code}")


def recommend_for_customer(customer_id, n=10):
    """ALS scores = customer taste vector x product vectors. Unknown customers get popular items."""
    idx = A["cust"].get(int(customer_id))
    if idx is None:
        return A["cold"].head(n)[["StockCode", "Description"]].assign(**{"Bought before": False})
    scores = A["item_f"] @ A["user_f"][idx]
    top = np.argsort(-scores)[:n]
    bought = A["matrix"][idx].toarray().ravel() > 0
    return pd.DataFrame({
        "StockCode": [A["idx2code"][i] for i in top],
        "Description": [product_name(A["idx2code"][i]) for i in top],
        "Bought before": [bool(bought[i]) for i in top],
    })


def also_bought(code, n=8):
    i = A["code2idx"][code]
    sims = A["item_norm"] @ A["item_norm"][i]
    top = [j for j in np.argsort(-sims) if j != i][:n]
    return pd.DataFrame({
        "StockCode": [A["idx2code"][j] for j in top],
        "Description": [product_name(A["idx2code"][j]) for j in top],
        "Similarity": [round(float(sims[j]), 2) for j in top],
    })


def predict_segment(recency, frequency, monetary):
    x = np.log1p([[recency, frequency, monetary]])
    cluster = int(A["kmeans"].predict(A["scaler"].transform(x))[0])
    return cluster_to_segment[cluster]


# ---------- Header ----------
st.title("🛍️ ShopSegments")
st.caption("Who are our customers, what should we say to each group, and what should we recommend them?")

tab_overview, tab_segments, tab_reco, tab_new = st.tabs(
    ["Overview", "Segment cards", "Recommendations", "New customer"]
)

# ---------- Tab 1: Overview ----------
with tab_overview:
    total_rev = rfm["Monetary"].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(rfm):,}")
    c2.metric("Segments", rfm["Segment"].nunique())
    c3.metric("Total spend", f"£{total_rev:,.0f}")
    c4.metric("Avg spend per customer", f"£{rfm['Monetary'].mean():,.0f}")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Customer map")
        st.caption("Each dot is one customer. Customers close together behave similarly. Hover for details.")
        plot_df = rfm.rename(columns={
            "CustomerID": "Customer ID", "Recency": "Days since last purchase",
            "Frequency": "Number of orders", "Monetary": "Total spend (£)"})
        fig = px.scatter(
            plot_df, x="UMAP1", y="UMAP2", color="Segment",
            category_orders={"Segment": SEGMENT_ORDER}, color_discrete_map=COLORS,
            hover_data={"Customer ID": True, "Days since last purchase": True,
                        "Number of orders": True, "Total spend (£)": ":.0f",
                        "UMAP1": False, "UMAP2": False},
            render_mode="webgl", opacity=0.7, height=520)
        fig.update_layout(xaxis_title="", yaxis_title="", xaxis_showticklabels=False,
                          yaxis_showticklabels=False, legend_title_text="")
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Segment sizes")
        sizes = rfm["Segment"].value_counts().reindex(SEGMENT_ORDER).reset_index()
        sizes.columns = ["Segment", "Customers"]
        fig2 = px.bar(sizes, x="Customers", y="Segment", orientation="h", color="Segment",
                      color_discrete_map=COLORS, height=260)
        fig2.update_layout(showlegend=False, yaxis_title="", yaxis_autorange="reversed")
        st.plotly_chart(fig2, width="stretch")

        share = rfm.groupby("Segment")["Monetary"].sum().reindex(SEGMENT_ORDER) / total_rev * 100
        st.subheader("Share of total spend")
        for seg in SEGMENT_ORDER:
            st.write(f"**{seg}**: {share[seg]:.0f}% of spend")

# ---------- Tab 2: Segment cards ----------
with tab_segments:
    st.subheader("What each segment looks like and what to do")
    cols = st.columns(2)
    for k, seg in enumerate(SEGMENT_ORDER):
        g = rfm[rfm["Segment"] == seg]
        with cols[k % 2]:
            with st.container(border=True):
                st.markdown(f"### {seg}")
                st.write(SEGMENT_INFO[seg]["what"])
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Customers", f"{len(g):,}", f"{len(g)/len(rfm)*100:.0f}%", delta_color="off")
                m2.metric("Avg spend", f"£{g['Monetary'].mean():,.0f}")
                m3.metric("Avg orders", f"{g['Frequency'].mean():.1f}")
                m4.metric("Days since last buy", f"{g['Recency'].mean():.0f}")
                st.markdown("**Top products**")
                top = A["seg_top"][seg][["Description", "Buyers"]].rename(columns={"Buyers": "Customers who bought"})
                st.dataframe(top, hide_index=True, width="stretch")
                st.info(f"**Suggested action:** {SEGMENT_INFO[seg]['action']}")

# ---------- Tab 3: Recommendations ----------
with tab_reco:
    left, right = st.columns(2)
    with left:
        st.subheader("Recommendations for a customer")
        cid = st.selectbox("Customer ID", options=sorted(rfm["CustomerID"].tolist()),
                           help="Type to search")
        row = rfm[rfm["CustomerID"] == cid].iloc[0]
        st.markdown(f"Segment: **{row['Segment']}**")
        a, b, c = st.columns(3)
        a.metric("Days since last buy", int(row["Recency"]))
        b.metric("Orders", int(row["Frequency"]))
        c.metric("Total spend", f"£{row['Monetary']:,.0f}")
        if int(cid) not in A["cust"]:
            st.warning("No product history for this customer, showing popular products instead.")
        recs = recommend_for_customer(cid)
        st.markdown("**Top 10 products for this customer**")
        st.dataframe(recs, hide_index=True, width="stretch")
        st.caption("'Bought before' = True means they already bought it (a good re-order reminder).")
    with right:
        st.subheader("Customers who bought this also bought")
        codes = sorted(A["code2idx"], key=lambda c: product_name(c))
        pick = st.selectbox("Product", options=codes,
                            format_func=lambda c: f"{product_name(c)} ({c})",
                            index=codes.index("85123A") if "85123A" in codes else 0)
        st.dataframe(also_bought(pick), hide_index=True, width="stretch")

# ---------- Tab 4: New customer ----------
with tab_new:
    st.subheader("New customer (no purchase history)")
    st.write("We cannot personalise without history, so new customers see the products bought by the "
             "most different customers. This is the fallback used across the app.")
    st.dataframe(A["cold"][["Description", "Buyers"]].rename(columns={"Buyers": "Customers who bought"}),
                 hide_index=True, width="stretch")

    st.divider()
    st.subheader("Which segment would a customer fall into?")
    st.caption("Enter a customer's behaviour to see the segment the model assigns.")
    x1, x2, x3 = st.columns(3)
    r = x1.number_input("Days since last purchase", min_value=1, max_value=800, value=30)
    f = x2.number_input("Number of orders", min_value=1, max_value=400, value=5)
    m = x3.number_input("Total spend (£)", min_value=1.0, max_value=600000.0, value=1500.0, step=100.0)
    seg = predict_segment(r, f, m)
    st.success(f"Segment: **{seg}**")
    st.write(SEGMENT_INFO[seg]["action"])

st.divider()
st.caption("Data: Online Retail II (UCI), UK registered customers, Dec 2009 to Dec 2011. "
           "Guest checkouts (no customer ID) are not included.")
