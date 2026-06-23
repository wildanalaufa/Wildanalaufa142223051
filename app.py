"""
Aplikasi Data Mining - Customer Segmentation dengan K-Means Clustering
Dibuat untuk tugas kuliah Data Mining
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation - Data Mining",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Customer Segmentation menggunakan K-Means Clustering")
st.markdown(
    "Aplikasi ini melakukan **clustering (pengelompokan)** pelanggan mall "
    "berdasarkan usia, pendapatan tahunan, dan skor belanja menggunakan "
    "algoritma **K-Means**."
)

# ----------------------------------------------------------------------------
# SIDEBAR - UPLOAD DATA & PENGATURAN
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Pengaturan")

uploaded_file = st.sidebar.file_uploader(
    "Upload dataset CSV (opsional)", type=["csv"]
)

@st.cache_data
def load_default_data():
    return pd.read_csv("Mall_Customers.csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset kustom berhasil dimuat.")
else:
    df = load_default_data()
    st.sidebar.info("Menggunakan dataset bawaan: Mall_Customers.csv")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

st.sidebar.subheader("Pilih Fitur untuk Clustering")
selected_features = st.sidebar.multiselect(
    "Fitur numerik",
    options=numeric_cols,
    default=[c for c in ["Age", "Annual Income (k$)", "Spending Score (1-100)"] if c in numeric_cols]
)

st.sidebar.subheader("Jumlah Cluster (K)")
k = st.sidebar.slider("Pilih nilai K", min_value=2, max_value=10, value=5)

scale_data = st.sidebar.checkbox("Standarisasi data (StandardScaler)", value=True)

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Eksplorasi Data", "📈 Tentukan K Optimal", "🔍 Hasil Clustering", "📋 Profil Cluster"]
)

# ----------------------------------------------------------------------------
# TAB 1 - EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Tinjauan Dataset")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.dataframe(df.head(10), use_container_width=True)

    with col2:
        st.metric("Jumlah Baris", df.shape[0])
        st.metric("Jumlah Kolom", df.shape[1])
        st.metric("Missing Value", int(df.isnull().sum().sum()))

    st.subheader("Statistik Deskriptif")
    st.dataframe(df[numeric_cols].describe(), use_container_width=True)

    if len(selected_features) >= 2:
        st.subheader("Distribusi Antar Fitur")
        fig = px.scatter_matrix(
            df,
            dimensions=selected_features,
            color="Gender" if "Gender" in df.columns else None,
            title="Scatter Matrix Fitur Terpilih"
        )
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# PERSIAPAN DATA UNTUK CLUSTERING
# ----------------------------------------------------------------------------
if len(selected_features) < 2:
    st.warning("⚠️ Pilih minimal 2 fitur numerik di sidebar untuk melanjutkan proses clustering.")
    st.stop()

X = df[selected_features].copy()

if scale_data:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X.values

# ----------------------------------------------------------------------------
# TAB 2 - ELBOW METHOD & SILHOUETTE SCORE
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Metode Elbow (Within-Cluster Sum of Squares)")
    st.markdown(
        "Grafik ini membantu menentukan jumlah cluster (K) yang optimal. "
        "Titik 'siku' (elbow) pada grafik biasanya menunjukkan nilai K yang baik."
    )

    wcss = []
    sil_scores = []
    k_range = range(2, 11)

    for i in k_range:
        km = KMeans(n_clusters=i, init="k-means++", random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))

    col1, col2 = st.columns(2)

    with col1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=list(k_range), y=wcss, mode="lines+markers"))
        fig_elbow.update_layout(
            title="Elbow Method",
            xaxis_title="Jumlah Cluster (K)",
            yaxis_title="WCSS (Inertia)"
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(x=list(k_range), y=sil_scores, mode="lines+markers", line=dict(color="orange")))
        fig_sil.update_layout(
            title="Silhouette Score per K",
            xaxis_title="Jumlah Cluster (K)",
            yaxis_title="Silhouette Score"
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    st.info(f"💡 Berdasarkan Silhouette Score tertinggi, nilai K yang disarankan adalah **{best_k}**.")

# ----------------------------------------------------------------------------
# CLUSTERING DENGAN K TERPILIH
# ----------------------------------------------------------------------------
kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df_result = df.copy()
df_result["Cluster"] = cluster_labels
sil_score_final = silhouette_score(X_scaled, cluster_labels)

# ----------------------------------------------------------------------------
# TAB 3 - HASIL CLUSTERING
# ----------------------------------------------------------------------------
with tab3:
    st.subheader(f"Hasil Clustering dengan K = {k}")
    st.metric("Silhouette Score", f"{sil_score_final:.4f}")

    if len(selected_features) == 2:
        fig_cluster = px.scatter(
            df_result, x=selected_features[0], y=selected_features[1],
            color=df_result["Cluster"].astype(str),
            title="Visualisasi Hasil Cluster",
            labels={"color": "Cluster"}
        )
    else:
        # Lebih dari 2 fitur -> reduksi dimensi dengan PCA untuk visualisasi
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(X_scaled)
        df_result["PCA1"] = pca_result[:, 0]
        df_result["PCA2"] = pca_result[:, 1]
        fig_cluster = px.scatter(
            df_result, x="PCA1", y="PCA2",
            color=df_result["Cluster"].astype(str),
            title="Visualisasi Hasil Cluster (Reduksi Dimensi PCA)",
            labels={"color": "Cluster"}
        )
        st.caption(
            f"Dataset memiliki {len(selected_features)} fitur, sehingga divisualisasikan "
            f"dalam 2 dimensi menggunakan PCA (variansi terjelaskan: "
            f"{pca.explained_variance_ratio_.sum()*100:.1f}%)."
        )

    st.plotly_chart(fig_cluster, use_container_width=True)

    st.subheader("Data Hasil Clustering")
    st.dataframe(df_result, use_container_width=True)

    csv_download = df_result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Unduh Hasil Clustering (CSV)",
        data=csv_download,
        file_name="hasil_clustering.csv",
        mime="text/csv"
    )

# ----------------------------------------------------------------------------
# TAB 4 - PROFIL CLUSTER
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("Profil Rata-Rata Tiap Cluster")
    profile = df_result.groupby("Cluster")[selected_features].mean().round(2)
    profile["Jumlah Anggota"] = df_result["Cluster"].value_counts().sort_index()
    st.dataframe(profile, use_container_width=True)

    st.subheader("Jumlah Anggota per Cluster")
    fig_count = px.bar(
        x=profile.index.astype(str), y=profile["Jumlah Anggota"],
        labels={"x": "Cluster", "y": "Jumlah Anggota"},
        title="Distribusi Anggota Cluster"
    )
    st.plotly_chart(fig_count, use_container_width=True)

    st.markdown(
        "💡 **Tips untuk laporan:** gunakan tabel profil di atas untuk memberi nama "
        "atau interpretasi pada tiap cluster, misalnya 'pelanggan loyal dengan "
        "pendapatan tinggi dan belanja tinggi' atau 'pelanggan hemat'."
    )

st.markdown("---")
st.caption("Dibuat untuk tugas kuliah Data Mining · K-Means Clustering · Streamlit")
