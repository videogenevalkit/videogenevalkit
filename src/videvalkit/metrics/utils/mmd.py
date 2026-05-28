"""Polynomial-kernel MMD² — the statistic behind KID / KVD.

Per Bińkowski et al. 2018 [Kernel Inception Distance], video-extended for KVD.
More stable than Fréchet distance at small sample sizes.

k(x, y) = (x·y / d + 1)^3  [cubic polynomial kernel, gamma=1/d, coef0=1].
MMD²(X, Y) = mean_{i≠j} k(x_i, x_j) + mean_{i≠j} k(y_i, y_j) - 2 mean_{i,j} k(x_i, y_j).

float64 throughout for reproducibility [VIDEO_METRICS_DESIGN §10].
"""

from __future__ import annotations

import numpy as np


def _poly_kernel(X: np.ndarray, Y: np.ndarray, degree: int = 3) -> np.ndarray:
    d = X.shape[1]
    return (X @ Y.T / d + 1.0) ** degree


def polynomial_mmd2(
    gen_feats: np.ndarray, ref_feats: np.ndarray, degree: int = 3,
) -> float:
    """Unbiased polynomial-kernel MMD² between two feature sets [float64]."""
    X = np.asarray(gen_feats, dtype=np.float64)
    Y = np.asarray(ref_feats, dtype=np.float64)
    m, n = X.shape[0], Y.shape[0]
    if m < 2 or n < 2:
        raise ValueError(
            f"KVD/MMD needs >=2 samples per set, got gen={m}, ref={n}"
        )

    kxx = _poly_kernel(X, X, degree)
    kyy = _poly_kernel(Y, Y, degree)
    kxy = _poly_kernel(X, Y, degree)

    # Unbiased: exclude diagonal for the within-set terms.
    sum_xx = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    sum_yy = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    sum_xy = kxy.mean()

    return float(sum_xx + sum_yy - 2.0 * sum_xy)
