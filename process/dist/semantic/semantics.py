"""
Semantic cluster assignment — the semantic-dimension analogue of
processing.dist.lengths' short/medium/long buckets.

Partitions per-am-group English-side embeddings (processing.utils.score_embed,
mean-pooled across every English reference sharing an am — see
processing.utils.pool._am_cluster_ids) into N_CLUSTERS regions via k-means, so
processing.utils.pool.split_semantic can stratify train/val/test by topic/domain
the same way it already stratifies by length.

Single source of truth for the semantic dimension: both pool.py (stratification)
and semantic_dist.py (reporting) import N_CLUSTERS/cluster_ids from here, same
role lengths.py plays for BUCKETS/bucketize.
"""
import numpy as np

N_CLUSTERS = 16


def cluster_ids(vecs: np.ndarray, k: int = N_CLUSTERS, seed: int = 42) -> np.ndarray:
    """K-means cluster id per row of `vecs`.

    MiniBatchKMeans rather than plain KMeans: the unique-am count is in the
    hundreds of thousands, and MiniBatchKMeans fits in seconds instead of
    minutes at that scale with essentially the same result.

    `vecs` must already be L2-normalized (unit vectors) — sklearn's k-means
    minimizes Euclidean distance, not cosine similarity, but for unit vectors
    the two are monotonically related (‖a-b‖² = 2 - 2·cos_sim(a,b)), so
    minimizing Euclidean distance among unit vectors is equivalent to
    maximizing cosine similarity. Passing un-normalized vectors here would
    silently cluster by magnitude instead of semantic direction.
    """
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init="auto")
    return km.fit_predict(vecs)
