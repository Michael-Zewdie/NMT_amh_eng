"""
Short / medium / long length buckets.

"""
import numpy as np


LENGTH_CUTOFFS = (40, 120)

BUCKETS = ("short", "medium", "long")


def bucketize(lengths, cutoffs=LENGTH_CUTOFFS) -> np.ndarray:
    """Label each length short/medium/long per the cutoffs (vectorized)."""
    lo, hi = cutoffs
    lengths = np.asarray(lengths)
    return np.where(lengths < lo, "short", np.where(lengths < hi, "medium", "long"))
