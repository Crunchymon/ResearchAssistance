from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.express as px
from schemas.chunk_schema import Chunk
from schemas.document_schema import Document
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from plotly.graph_objects import Figure


def _auto_select_k(embeddings: np.ndarray) -> int:
    n_samples = len(embeddings)
    min_k, max_k = _cluster_bounds(n_samples)
    if max_k <= 1:
        return 1

    best_k = min_k
    best_score = -1.0
    for k in range(min_k, max_k + 1):
        try:
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(embeddings)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    return best_k


def _cluster_bounds(n_samples: int) -> Tuple[int, int]:
    if n_samples <= 1:
        return 1, 1
    if n_samples < 4:
        return 1, n_samples
    min_k = 4
    max_k = min(8, n_samples)
    return min_k, max_k


def _resolve_k(embeddings: np.ndarray, requested_k: Optional[int]) -> Tuple[int, int, int, int]:
    n_samples = len(embeddings)
    min_k, max_k = _cluster_bounds(n_samples)
    auto_k = _auto_select_k(embeddings)

    if requested_k is None:
        selected_k = auto_k
    else:
        selected_k = max(min_k, min(int(requested_k), max_k))

    return selected_k, auto_k, min_k, max_k


def cluster_documents(
    documents: List[Document],
    requested_k: Optional[int] = None,
) -> Tuple[List[Document], Optional[Figure], Dict[str, int]]:
    valid_docs = [doc for doc in documents if doc.embedding]
    if not valid_docs:
        return documents, None, {"selected_k": 1, "auto_k": 1, "min_k": 1, "max_k": 1, "n_docs": 0}

    embeddings = np.array([doc.embedding for doc in valid_docs])
    n_docs = len(valid_docs)
    k, auto_k, min_k, max_k = _resolve_k(embeddings, requested_k)

    if k > 1:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
    else:
        labels = np.zeros(n_docs, dtype=int)

    for doc, label in zip(valid_docs, labels):
        doc.cluster_id = int(label)

    if n_docs < 2:
        return documents, None, {"selected_k": k, "auto_k": auto_k, "min_k": min_k, "max_k": max_k, "n_docs": n_docs}

    pca = PCA(n_components=2, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings)

    cluster_rows = []
    for cluster_id in sorted(set(int(label) for label in labels)):
        cluster_points = reduced_embeddings[labels == cluster_id]
        cluster_doc_titles = [
            doc.title
            for doc, label in zip(valid_docs, labels)
            if int(label) == cluster_id
        ]
        cluster_rows.append(
            {
                "cluster_id": str(cluster_id),
                "x": float(np.mean(cluster_points[:, 0])),
                "y": float(np.mean(cluster_points[:, 1])),
                "cluster_size": int(len(cluster_points)),
                "members": "<br>".join(cluster_doc_titles[:8]),
            }
        )

    pca_fig = px.scatter(
        cluster_rows,
        x="x",
        y="y",
        size="cluster_size",
        color="cluster_id",
        hover_name="cluster_id",
        hover_data={"cluster_size": True, "members": True, "x": False, "y": False},
        title="Cluster Map (PCA Centroids)",
        labels={"x": "PCA Component 1", "y": "PCA Component 2", "cluster_id": "Cluster"},
        size_max=54,
    )
    pca_fig.update_layout(height=420)

    return documents, pca_fig, {"selected_k": k, "auto_k": auto_k, "min_k": min_k, "max_k": max_k, "n_docs": n_docs}


def cluster_chunks(
    chunks: List[Chunk],
    requested_k: Optional[int] = None,
) -> Tuple[List[Chunk], Optional[Figure], Dict[str, int]]:
    valid_chunks = [chunk for chunk in chunks if chunk.embedding]
    if not valid_chunks:
        return chunks, None, {"selected_k": 1, "auto_k": 1, "min_k": 1, "max_k": 1, "n_chunks": 0}

    embeddings = np.array([chunk.embedding for chunk in valid_chunks])
    n_chunks = len(valid_chunks)
    k, auto_k, min_k, max_k = _resolve_k(embeddings, requested_k)

    if k > 1:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(embeddings)
    else:
        labels = np.zeros(n_chunks, dtype=int)

    for chunk, label in zip(valid_chunks, labels):
        chunk.cluster_id = int(label)

    if n_chunks < 2:
        return chunks, None, {
            "selected_k": k,
            "auto_k": auto_k,
            "min_k": min_k,
            "max_k": max_k,
            "n_chunks": n_chunks,
        }

    reduced = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    plot_rows = []
    for idx, chunk in enumerate(valid_chunks):
        plot_rows.append(
            {
                "x": float(reduced[idx, 0]),
                "y": float(reduced[idx, 1]),
                "cluster_id": str(chunk.cluster_id),
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "preview": chunk.text[:160],
            }
        )

    pca_fig = px.scatter(
        plot_rows,
        x="x",
        y="y",
        color="cluster_id",
        hover_name="title",
        hover_data={"doc_id": True, "preview": True, "x": False, "y": False},
        title="Chunk Clusters (PCA)",
        labels={"x": "PCA Component 1", "y": "PCA Component 2", "cluster_id": "Cluster"},
    )
    pca_fig.update_traces(marker={"size": 9, "opacity": 0.78})
    pca_fig.update_layout(height=460)

    return chunks, pca_fig, {
        "selected_k": k,
        "auto_k": auto_k,
        "min_k": min_k,
        "max_k": max_k,
        "n_chunks": n_chunks,
    }
