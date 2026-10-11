"""Consistency scores and document-level features (Eqs. 6-8, 11).

Pure NumPy: no model dependencies.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# NLI probability columns are always ordered [entailment, neutral, contradiction].
E, N, C = 0, 1, 2
FEATURE_NAMES = ("min", "mean", "std", "frac_below")
SCORE_VARIANTS = ("signed", "entail", "one_minus_contra")


def pair_scores(probs: np.ndarray, variant: str = "signed") -> np.ndarray:
    """Eq. 6. ``probs`` has shape (..., 3) -> scores of shape (...)."""
    probs = np.asarray(probs, dtype=np.float64)
    if variant == "signed":
        return probs[..., E] - probs[..., C]
    if variant == "entail":
        return probs[..., E]
    if variant == "one_minus_contra":
        return 1.0 - probs[..., C]
    raise ValueError(f"unknown score variant {variant!r}; choose from {SCORE_VARIANTS}")


def sentence_consistency(
    pair_probs: Sequence[np.ndarray], k: Optional[int] = None, variant: str = "signed"
) -> np.ndarray:
    """Eq. 7: mean pairwise score over the (first ``k``) retrieved passages.

    ``pair_probs[i]`` has shape (k_i, 3), ordered by retrieval rank. A sentence
    with no retrieved evidence gets the neutral score 0.0.
    """
    out = []
    for p in pair_probs:
        p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
        if k is not None:
            p = p[:k]
        out.append(float(pair_scores(p, variant).mean()) if len(p) else 0.0)
    return np.asarray(out, dtype=np.float64)


def document_features(
    c_hat: np.ndarray, tau: float = 0.0, names: Sequence[str] = FEATURE_NAMES
) -> np.ndarray:
    """Eq. 8: [min, mean, std (population), fraction below tau] (or a subset)."""
    c_hat = np.asarray(c_hat, dtype=np.float64)
    if c_hat.size == 0:
        raise ValueError("document_features needs at least one sentence score")
    all_feats = {
        "min": c_hat.min(),
        "mean": c_hat.mean(),
        "std": c_hat.std(),  # ddof=0, matching the 1/m formula in Eq. 8
        "frac_below": float((c_hat < tau).mean()),
    }
    return np.asarray([all_feats[n] for n in names], dtype=np.float64)


def sentence_labels(c_hat: np.ndarray, tau: float = 0.0) -> np.ndarray:
    """Eq. 11: per-sentence hallucination indicator."""
    return (np.asarray(c_hat) < tau).astype(int)
