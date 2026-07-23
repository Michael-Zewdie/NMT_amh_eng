"""
Semantic clusters — k-means over sentence embeddings, for use as a second
stratification axis alongside nmt.lengths' short/medium/long buckets.

The model is persisted so cluster ids stay stable across runs: load_or_fit
loads an existing model rather than refitting by default, since a refit would
renumber every cluster and make any past per-cluster analysis noncomparable.
Delete the persisted .joblib (or call fit() directly) to force a refit —
same convention score_cache.py already uses for its own cache.
"""
import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from processing.utils.paths import MODELS

N_CLUSTERS = 20
SEED = 42


def _model_path(k: int):
    return MODELS / f"semantic_kmeans_k{k}.joblib"


def fit(embeddings: np.ndarray, k: int = N_CLUSTERS, seed: int = SEED) -> MiniBatchKMeans:
    """Fit a fresh k-means model on `embeddings` and persist it."""
    model = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init="auto")
    model.fit(embeddings)
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, _model_path(k))
    print(f"[clusters] fit k={k} on {len(embeddings)} embeddings → {_model_path(k)}")
    return model


def load(k: int = N_CLUSTERS) -> MiniBatchKMeans | None:
    """The persisted model for this k, or None if it hasn't been fit yet."""
    path = _model_path(k)
    return joblib.load(path) if path.exists() else None


def load_or_fit(embeddings: np.ndarray, k: int = N_CLUSTERS, seed: int = SEED) -> MiniBatchKMeans:
    """Load the persisted model for this k, fitting fresh only if none exists."""
    model = load(k)
    if model is not None:
        print(f"[clusters] loaded k={k} model → {_model_path(k)}")
        return model
    return fit(embeddings, k=k, seed=seed)


def bucketize(embeddings: np.ndarray, model: MiniBatchKMeans) -> np.ndarray:
    """Label each embedding with its predicted cluster id, e.g. "cluster_7"."""
    labels = model.predict(embeddings)
    return np.array([f"cluster_{i}" for i in labels])


def bucket_names(k: int = N_CLUSTERS) -> tuple[str, ...]:
    return tuple(f"cluster_{i}" for i in range(k))
