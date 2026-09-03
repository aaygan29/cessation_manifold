"""PCA + UMAP embedding of the feature table.

Fits on epochs labeled as "cessation-like" (from the synthetic collapse mask,
or eventually from real cessation-onset annotations) and transforms the rest
into the same space, so a cessation centroid can be defined and every other
epoch's distance to it computed.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.decomposition import PCA

try:
    import umap

    _HAS_UMAP = True
except Exception:  # pragma: no cover
    _HAS_UMAP = False


@dataclass
class ManifoldModel:
    pca: PCA
    reducer: object
    feature_mean: np.ndarray
    feature_std: np.ndarray
    cessation_centroid: np.ndarray
    feature_names: list


def fit_manifold(
    X: np.ndarray,
    cessation_mask: np.ndarray,
    feature_names: list,
    n_pca: int = 10,
    n_umap: int = 2,
    seed: int = 0,
) -> ManifoldModel:
    """X: (n_epochs, n_features). cessation_mask: bool (n_epochs,) marking
    epochs to anchor the cessation centroid on."""
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-12
    Xz = (X - mean) / std

    n_pca = min(n_pca, Xz.shape[0], Xz.shape[1])
    pca = PCA(n_components=n_pca, random_state=seed).fit(Xz)
    Xp = pca.transform(Xz)

    if _HAS_UMAP and Xp.shape[0] >= 10:
        reducer = umap.UMAP(n_components=n_umap, random_state=seed)
        Xr = reducer.fit_transform(Xp)
    else:
        reducer = PCA(n_components=min(n_umap, Xp.shape[1]), random_state=seed).fit(Xp)
        Xr = reducer.transform(Xp)

    if cessation_mask.sum() == 0:
        raise ValueError("cessation_mask has no True entries; cannot anchor centroid")
    centroid = Xr[cessation_mask].mean(axis=0)

    model = ManifoldModel(
        pca=pca,
        reducer=reducer,
        feature_mean=mean,
        feature_std=std,
        cessation_centroid=centroid,
        feature_names=feature_names,
    )
    return model, Xr


def transform(model: ManifoldModel, X: np.ndarray) -> np.ndarray:
    Xz = (X - model.feature_mean) / model.feature_std
    Xp = model.pca.transform(Xz)
    if hasattr(model.reducer, "transform"):
        return model.reducer.transform(Xp)
    return model.reducer.fit_transform(Xp)
