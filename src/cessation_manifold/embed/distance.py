"""Distance-from-cessation-centroid readout."""
from __future__ import annotations

import numpy as np


def distance_from_cessation(Xr: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Euclidean distance in embedded space from each row to the cessation centroid.
    A geodesic (manifold-following) distance is a natural upgrade once there is
    enough real data density to estimate the manifold's local structure; for v0,
    straight-line distance in the low-dimensional embedding is the honest
    baseline and is documented as such."""
    return np.linalg.norm(Xr - centroid[None, :], axis=1)
