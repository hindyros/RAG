"""GET /visualize — 2-D PCA projection of all chunk embeddings with k-means clustering."""
import asyncio
import logging
import numpy as np
from fastapi import APIRouter, Depends, Request
from app.api.schemas import ClusterInfo, VisualizationPoint, VisualizationResponse
from app.dependencies import get_vector_store
from app.store.vector_store import VectorStore

logger = logging.getLogger(__name__)
router = APIRouter()


def _run_pca(matrix: np.ndarray) -> np.ndarray:
    """(N, D) float32 → (N, 2) float64 via mean-centred truncated SVD."""
    X = matrix.astype(np.float64)
    X -= X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:2].T


def _kmeans(X: np.ndarray, k: int, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Simple numpy k-means on 2-D coords. Returns (labels, centroids)."""
    rng = np.random.default_rng(42)
    centroids = X[rng.choice(len(X), k, replace=False)].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)  # (N, k)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            mask = labels == i
            if mask.any():
                centroids[i] = X[mask].mean(axis=0)
    return labels, centroids


async def _label_cluster(client, excerpts: list[str]) -> str:
    if not excerpts:
        return "Unknown Topic"
    sample = "\n".join(f'{i + 1}. "{e}"' for i, e in enumerate(excerpts))
    prompt = (
        f"Given these text excerpts from a cluster of related document chunks:\n\n{sample}\n\n"
        "Provide a concise 3-5 word topic label that best describes the common theme. "
        "Respond with only the label, no punctuation or explanation."
    )
    try:
        reply = await client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=20,
        )
        return reply.strip().strip('"\'.,')
    except Exception:
        return "Unknown Topic"


@router.get("/visualize", response_model=VisualizationResponse)
async def visualize_endpoint(
    request: Request,
    store: VectorStore = Depends(get_vector_store),
) -> VisualizationResponse:
    n = store.total_chunks

    # Cache hit
    cached = request.app.state.viz_cache
    if cached is not None and cached[0] == n:
        return cached[1]

    # Empty store
    if n == 0:
        resp = VisualizationResponse(total_chunks=0, points=[], document_ids=[], clusters=[])
        request.app.state.viz_cache = (0, resp)
        return resp

    matrix = store.get_cosine_index().matrix  # (N, D) float32

    # Single-chunk edge case
    if n == 1:
        meta = store.get_chunk(0)
        resp = VisualizationResponse(
            total_chunks=1,
            points=[VisualizationPoint(
                chunk_index=0, x=0.0, y=0.0, cluster_id=0,
                document_id=meta.document_id, source_file=meta.source_file,
                page_number=meta.page_number, section_header=meta.section_header,
                excerpt=meta.text[:200].strip(),
            )],
            document_ids=[meta.document_id],
            clusters=[ClusterInfo(cluster_id=0, label="Single Chunk",
                                  centroid_x=0.0, centroid_y=0.0, chunk_count=1)],
        )
        request.app.state.viz_cache = (1, resp)
        return resp

    logger.info("Running PCA on %d × %d matrix", n, matrix.shape[1])
    coords = _run_pca(matrix)  # (N, 2) float64

    # Auto-select k: sqrt heuristic, clamped to [2, 8]
    k = max(2, min(8, round((n / 4) ** 0.5)))
    k = min(k, n)
    labels, centroids = _kmeans(coords.astype(np.float32), k)
    logger.info("K-means complete: k=%d for %d points", k, n)

    # Collect 4 representative excerpts per cluster (closest to centroid)
    cluster_excerpts: list[list[str]] = []
    for cid in range(k):
        mask = np.where(labels == cid)[0]
        if len(mask):
            dists = np.linalg.norm(coords[mask] - centroids[cid], axis=1)
            top = mask[np.argsort(dists)[:4]]
            excerpts = [store.get_chunk(int(i)).text[:150].strip() for i in top]
        else:
            excerpts = []
        cluster_excerpts.append(excerpts)

    # Label all clusters concurrently
    client = request.app.state.llm_client
    label_strs = await asyncio.gather(*[
        _label_cluster(client, excerpts) for excerpts in cluster_excerpts
    ])

    clusters = [
        ClusterInfo(
            cluster_id=cid,
            label=label_strs[cid],
            centroid_x=float(centroids[cid, 0]),
            centroid_y=float(centroids[cid, 1]),
            chunk_count=int((labels == cid).sum()),
        )
        for cid in range(k)
    ]

    points, seen_docs = [], {}
    for i in range(n):
        meta = store.get_chunk(i)
        seen_docs[meta.document_id] = None
        points.append(VisualizationPoint(
            chunk_index=i,
            x=float(coords[i, 0]), y=float(coords[i, 1]),
            cluster_id=int(labels[i]),
            document_id=meta.document_id, source_file=meta.source_file,
            page_number=meta.page_number, section_header=meta.section_header,
            excerpt=meta.text[:200].strip(),
        ))

    resp = VisualizationResponse(
        total_chunks=n, points=points,
        document_ids=list(seen_docs.keys()), clusters=clusters,
    )
    request.app.state.viz_cache = (n, resp)
    logger.info("Visualization cached — %d points, %d clusters, labels: %s",
                n, k, [c.label for c in clusters])
    return resp
