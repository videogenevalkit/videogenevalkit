"""Fréchet distance between two Gaussians fitted to feature sets.

Shared by FVD / VFID / CLIP-FVD. float64 throughout for reproducibility
[VIDEO_METRICS_DESIGN §10].
"""

from __future__ import annotations

import numpy as np


def compute_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu, sigma) for an (N, D) feature matrix, in float64."""
    feats = np.asarray(features, dtype=np.float64)
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray,
    mu2: np.ndarray, sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Fréchet distance between N(mu1, sigma1) and N(mu2, sigma2).

    ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 sqrt(sigma1 sigma2)).
    Uses scipy.linalg.sqrtm [double precision]; adds eps*I if the product
    matrix is near-singular [standard FID numerical guard].
    """
    from scipy import linalg

    mu1 = np.atleast_1d(np.asarray(mu1, dtype=np.float64))
    mu2 = np.atleast_1d(np.asarray(mu2, dtype=np.float64))
    sigma1 = np.atleast_2d(np.asarray(sigma1, dtype=np.float64))
    sigma2 = np.atleast_2d(np.asarray(sigma2, dtype=np.float64))

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def fid_from_features(gen_feats: np.ndarray, ref_feats: np.ndarray) -> float:
    """Convenience: FID-style Fréchet distance directly from two feature sets."""
    mu1, sigma1 = compute_statistics(gen_feats)
    mu2, sigma2 = compute_statistics(ref_feats)
    return frechet_distance(mu1, sigma1, mu2, sigma2)
