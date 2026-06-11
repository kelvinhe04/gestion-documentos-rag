"""Dashboard de estadísticas de los documentos y clustering temático (ML)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

import app_helpers as ah

ah.page_config("Dashboard", "📊")
ah.sidebar_status()

st.title("📊 Dashboard de Documentos")
st.caption("Estadísticas de la colección y agrupamiento temático con Machine Learning (KMeans).")

version = ah.data_version()
stats = ah.cached_statistics(version)

if stats["n_documents"] == 0:
    st.warning("No hay documentos indexados todavía. Ve a **📥 Ingesta de Documentos**.", icon="⚠️")
    st.stop()

# ---------------------------------------------------------------------------
# 1) Métricas generales
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("📄 Documentos", stats["n_documents"])
c2.metric("🧩 Fragmentos", stats["n_chunks"])
c3.metric("📑 Páginas totales", stats["n_pages"])
prom = round(stats["n_chunks"] / stats["n_documents"], 1) if stats["n_documents"] else 0
c4.metric("Fragmentos/doc (prom.)", prom)

st.divider()

# ---------------------------------------------------------------------------
# 2) Distribución por fuente y por documento
# ---------------------------------------------------------------------------
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Documentos por fuente")
    if stats["by_source"]:
        df_src = pd.DataFrame(
            {"Fuente": list(stats["by_source"].keys()), "Documentos": list(stats["by_source"].values())}
        )
        fig = px.pie(df_src, names="Fuente", values="Documentos", hole=0.45)
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320)
        st.plotly_chart(fig, width="stretch")

with col_b:
    st.subheader("Fragmentos por documento")
    df_docs = pd.DataFrame(stats["documents"])
    df_top = df_docs.sort_values("chunks", ascending=False).head(12)
    fig2 = px.bar(
        df_top, x="chunks", y="title", orientation="h",
        labels={"chunks": "Fragmentos", "title": ""}, color="source",
    )
    fig2.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, width="stretch")

# ---------------------------------------------------------------------------
# 3) Tabla de documentos
# ---------------------------------------------------------------------------
st.subheader("📋 Documentos indexados")
tabla = df_docs[["title", "source", "num_pages", "chunks", "author"]].rename(
    columns={
        "title": "Título", "source": "Fuente", "num_pages": "Páginas",
        "chunks": "Fragmentos", "author": "Autor",
    }
)
st.dataframe(tabla, width="stretch", hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# 4) Machine Learning: clustering temático (KMeans)
# ---------------------------------------------------------------------------
st.header("🤖 Agrupamiento temático (Machine Learning)")
st.caption(
    "Técnica de ML: **clustering con KMeans** sobre los embeddings de los "
    "documentos. Agrupa los documentos por similitud temática."
)

if stats["n_documents"] < 2:
    st.info("Se necesitan al menos 2 documentos para ejecutar el clustering.")
    st.stop()

col_k1, col_k2 = st.columns([1, 3])
modo_auto = col_k1.toggle("k automático (silueta)", value=True)
k_manual = col_k2.slider(
    "Número de clusters (k)", 2, max(2, min(10, stats["n_documents"] - 1)), 3,
    disabled=modo_auto,
)

resultado = ah.cached_clustering(version, None if modo_auto else k_manual)

if "error" in resultado:
    st.info(resultado["error"])
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Clusters (k)", resultado["k"])
m2.metric("Coef. de silueta", resultado["silhouette"])
m3.metric("Documentos agrupados", resultado["n_docs"])
st.caption(
    "El **coeficiente de silueta** (entre -1 y 1) mide la calidad del "
    "agrupamiento; cuanto más cercano a 1, mejor separados están los grupos."
)

df_clusters = pd.DataFrame(resultado["documents"])
df_clusters["cluster"] = df_clusters["cluster"].astype(str)

# Etiquetas legibles por cluster (términos TF-IDF más representativos).
etiquetas = {
    str(cid): ", ".join(terms[:4]) if terms else f"Cluster {cid}"
    for cid, terms in resultado["top_terms"].items()
}
df_clusters["tema"] = df_clusters["cluster"].map(lambda c: etiquetas.get(c, f"Cluster {c}"))

col_v1, col_v2 = st.columns([3, 2])

with col_v1:
    st.subheader("Mapa de documentos (PCA 2D)")
    fig3 = px.scatter(
        df_clusters, x="x", y="y", color="cluster", hover_name="title",
        hover_data={"tema": True, "source": True, "x": False, "y": False},
        labels={"cluster": "Cluster"}, height=420,
    )
    fig3.update_traces(marker=dict(size=14, line=dict(width=1, color="white")))
    fig3.update_layout(margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig3, width="stretch")

with col_v2:
    st.subheader("Temas por cluster")
    for cid in sorted(resultado["top_terms"].keys()):
        terms = resultado["top_terms"][cid]
        size = resultado["cluster_sizes"].get(cid, 0)
        st.markdown(f"**Cluster {cid}** · {size} doc(s)")
        st.caption(", ".join(terms) if terms else "—")

st.subheader("Documentos por cluster")
st.dataframe(
    df_clusters[["title", "tema", "cluster", "source", "n_chunks"]].rename(
        columns={
            "title": "Documento", "tema": "Tema (TF-IDF)", "cluster": "Cluster",
            "source": "Fuente", "n_chunks": "Fragmentos",
        }
    ),
    width="stretch",
    hide_index=True,
)
