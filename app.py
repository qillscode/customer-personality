import os

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Segmentasi Customer Personality", page_icon="🛍️", layout="wide")

MODEL_FEATURES = ["Income_Log", "Spending_Log", "Total_Children", "Recency"]
if os.path.exists("feature_names.pkl"):
    MODEL_FEATURES = joblib.load("feature_names.pkl")

NUM_COLS = MODEL_FEATURES

CLUSTER_INFO = {
    0: {
        "name": "High Value Customer",
        "desc": "Income dan total belanja tinggi. Segmen paling menguntungkan.",
        "strategy": "Program loyalitas eksklusif, akses awal produk premium, layanan personal.",
    },
    1: {
        "name": "Potential Customer",
        "desc": "Income cukup tinggi tapi belanja belum maksimal.",
        "strategy": "Penawaran/rekomendasi produk personal untuk meningkatkan konversi.",
    },
    2: {
        "name": "Family Shopper",
        "desc": "Income & belanja menengah, jumlah anak/tanggungan relatif banyak.",
        "strategy": "Bundling produk keluarga, promo musiman.",
    },
    3: {
        "name": "Budget / Low Engagement",
        "desc": "Income dan belanja rendah, kunjungan/transaksi jarang.",
        "strategy": "Promo diskon, kampanye re-engagement (voucher comeback, survei kepuasan).",
    },
}


@st.cache_resource
def load_artifacts():
    df = pd.read_csv("customer_personality_with_cluster.csv")

    feature_cols = MODEL_FEATURES
    if os.path.exists("feature_names.pkl"):
        feature_cols = joblib.load("feature_names.pkl")

    scaler = None
    if os.path.exists("scaler.pkl"):
        scaler = joblib.load("scaler.pkl")
    else:
        scaler = StandardScaler()
        scaler.fit(df[feature_cols])

    pca = None
    if os.path.exists("pca.pkl"):
        pca = joblib.load("pca.pkl")
    else:
        pca = PCA(n_components=2, random_state=42)
        pca.fit(scaler.transform(df[feature_cols]))

    model = None
    if os.path.exists("kmeans_model.pkl"):
        model = joblib.load("kmeans_model.pkl")
    else:
        n_clusters = int(df["cluster"].nunique()) if "cluster" in df.columns else 4
        X_pca = pca.transform(scaler.transform(df[feature_cols]))
        model = KMeans(n_clusters=max(2, n_clusters), random_state=42, n_init=10)
        model.fit(X_pca)

    return model, scaler, pca


@st.cache_data
def load_data():
    return pd.read_csv("customer_personality_with_cluster.csv")


def label_cluster(cluster_id: int) -> dict:
    return CLUSTER_INFO.get(
        cluster_id,
        {
            "name": f"Cluster {cluster_id}",
            "desc": "Profil segmen mengikuti pola umum pelanggan berdasarkan karakteristiknya.",
            "strategy": "Gunakan strategi marketing umum seperti personalisasi promo, retargeting, dan penawaran loyalitas.",
        },
    )


model, scaler, pca = load_artifacts()
df = load_data()

st.title("🛍️ Segmentasi Customer Personality (K-Means Clustering)")
st.caption("Tugas Mandiri Pertemuan 4 - CRISP-DM Deployment | Universitas Gunadarma")

tab1, tab2 = st.tabs(["📊 Eksplorasi Cluster", "🔮 Prediksi Segmen Pelanggan Baru"])

with tab1:
    st.subheader("Ringkasan Segmentasi Pelanggan")

    counts = df["cluster"].value_counts().sort_index()
    cols = st.columns(len(counts))
    for c, col in zip(counts.index, cols):
        info = label_cluster(c)
        col.metric(info["name"], f"{counts[c]} pelanggan")

    st.divider()

    left, right = st.columns([1, 1])

    with left:
        st.markdown("**Visualisasi Cluster (PCA 2D)**")
        X_scaled = scaler.transform(df[NUM_COLS])
        X_pca = pca.transform(X_scaled)
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df["cluster"], cmap="viridis", alpha=0.6)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        legend_labels = [label_cluster(c)["name"] for c in sorted(df["cluster"].unique())]
        ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Segmen", fontsize=8)
        st.pyplot(fig)

    with right:
        st.markdown("**Profil Rata-rata Tiap Cluster**")
        profile = df.groupby("cluster")[NUM_COLS].mean().round(2)
        profile.index = [label_cluster(i)["name"] for i in profile.index]
        st.dataframe(profile, use_container_width=True)

    st.divider()
    st.markdown("**Deskripsi & Strategi Marketing per Segmen**")
    for c in sorted(df["cluster"].unique()):
        info = label_cluster(c)
        with st.expander(f"{info['name']} ({counts[c]} pelanggan)"):
            st.write(f"**Karakteristik:** {info['desc']}")
            st.write(f"**Rekomendasi strategi:** {info['strategy']}")

    st.divider()
    st.markdown("**Data Pelanggan (contoh)**")
    st.dataframe(df.head(20), use_container_width=True)

with tab2:
    st.subheader("Prediksi Segmen untuk Pelanggan Baru")
    st.write("Masukkan karakteristik utama pelanggan yang sesuai dengan fitur model yang digunakan dalam training.")

    c1, c2 = st.columns(2)
    with c1:
        income = st.slider("Income tahunan (USD)", 1000, 150000, 50000, step=1000)
        total_spending = st.slider("Total belanja 2 tahun terakhir (USD)", 0, 3000, 500, step=10)
    with c2:
        total_children = st.slider("Jumlah anak/tanggungan di rumah", 0, 3, 0)
        recency = st.slider("Hari sejak transaksi terakhir", 0, 100, 30)

    if st.button("Prediksi Segmen", type="primary"):
        new_data = pd.DataFrame([{
            "Income_Log": np.log1p(income),
            "Spending_Log": np.log1p(total_spending),
            "Total_Children": total_children,
            "Recency": recency,
        }])
        new_scaled = scaler.transform(new_data[NUM_COLS])
        new_pca = pca.transform(new_scaled)
        pred_cluster = int(model.predict(new_pca)[0])
        info = label_cluster(pred_cluster)

        st.success(f"Pelanggan ini termasuk segmen: **{info['name']}**")
        st.write(f"**Karakteristik segmen:** {info['desc']}")
        st.info(f"**Rekomendasi strategi marketing:** {info['strategy']}")

st.divider()
st.caption("Dataset: Customer Personality Analysis (Kaggle - imakash3011/customer-personality-analysis).")
