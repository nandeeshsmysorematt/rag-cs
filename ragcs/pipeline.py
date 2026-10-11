"""End-to-end RAG-CS detection (Algorithm 1)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .classifier import HallucinationClassifier
from .config import RAGCSConfig
from .features import document_features, sentence_consistency, sentence_labels
from .schema import Passage
from .segmentation import SentenceSegmenter
from .utils import set_seed

Knowledge = Sequence[Union[str, Tuple[str, str], dict]]


@dataclass
class ScoredResponse:
    sentences: List[str]
    pair_probs: List[np.ndarray]          # per sentence: (k_i, 3) [E, N, C], rank-ordered
    evidence: List[List[Passage]]
    timings_ms: Dict[str, float] = field(default_factory=dict)


@dataclass
class Detection:
    label: int                            # document-level y_hat
    probability: Optional[float]          # p(y=1|f); None without a trained classifier
    sentences: List[str]
    sentence_scores: List[float]          # c_hat_i
    sentence_labels: List[int]            # y_hat_i
    features: Optional[List[float]]
    evidence: List[List[Passage]]
    timings_ms: Dict[str, float]
    used_classifier: bool


def _normalize_knowledge(knowledge: Optional[Knowledge]) -> Optional[List[Tuple[str, str]]]:
    if knowledge is None:
        return None
    out = []
    for item in knowledge:
        if isinstance(item, str):
            out.append(("", item))
        elif isinstance(item, dict):
            out.append((item.get("title", ""), item["text"]))
        else:
            out.append((item[0], item[1]))
    return out


class RAGCS:
    """Pass ``retriever`` / ``nli`` / ``segmenter`` to inject components (used in tests)."""

    def __init__(
        self,
        cfg: Optional[RAGCSConfig] = None,
        index_dir: Optional[str] = None,
        classifier: Optional[HallucinationClassifier] = None,
        retriever=None,
        nli=None,
        segmenter=None,
    ):
        self.cfg = cfg or RAGCSConfig()
        set_seed(self.cfg.seed)
        self.classifier = classifier
        self.segmenter = segmenter or SentenceSegmenter(
            self.cfg.spacy_model, self.cfg.min_sentence_tokens)
        if retriever is None:
            from .retriever import DenseRetriever

            retriever = (DenseRetriever.load(self.cfg, index_dir) if index_dir
                         else DenseRetriever.local_only(self.cfg))
        if nli is None:
            from .nli import NLIScorer

            nli = NLIScorer(self.cfg)
        self.retriever, self.nli = retriever, nli

    # ---- Stages 1 + 2 -----------------------------------------------------
    def score_sentences(self, query: str, response: str,
                        knowledge: Optional[Knowledge] = None, k: Optional[int] = None) -> ScoredResponse:
        k = k or self.cfg.k
        t0 = time.perf_counter()
        sentences = self.segmenter.split(response)
        t1 = time.perf_counter()
        if not sentences:
            return ScoredResponse([], [], [], {"segment": (t1 - t0) * 1e3})

        evidence = self.retriever.search(
            [query] * len(sentences), sentences, k, _normalize_knowledge(knowledge))
        t2 = time.perf_counter()

        flat_s, flat_p = [], []
        for s, ps in zip(sentences, evidence):
            for p in ps:
                flat_s.append(s)
                flat_p.append(p.full_text if self.cfg.prepend_title_for_nli else p.text)
        probs = self.nli.probs(flat_s, flat_p)   # one batched pass over all m*k pairs
        t3 = time.perf_counter()

        pair_probs, pos = [], 0
        for ps in evidence:
            pair_probs.append(probs[pos:pos + len(ps)])
            pos += len(ps)
        return ScoredResponse(sentences, pair_probs, evidence, {
            "segment": (t1 - t0) * 1e3, "retrieve": (t2 - t1) * 1e3, "nli": (t3 - t2) * 1e3})

    # ---- Stage 3 (Algorithm 1) -------------------------------------------
    def detect(self, query: str, response: str,
               knowledge: Optional[Knowledge] = None) -> Detection:
        scored = self.score_sentences(query, response, knowledge)
        if not scored.sentences:                      # empty output -> y_hat = 0
            return Detection(0, 0.0, [], [], [], None, [], scored.timings_ms, False)

        t0 = time.perf_counter()
        c_hat = sentence_consistency(scored.pair_probs)
        names = self.classifier.feature_names if self.classifier else None
        feats = (document_features(c_hat, self.cfg.tau, names) if names
                 else document_features(c_hat, self.cfg.tau))
        y_i = sentence_labels(c_hat, self.cfg.tau)
        if self.classifier is not None:
            prob = float(self.classifier.predict_proba(feats)[0])
            label = int(prob >= 0.5)
        else:  # untrained fallback: flag the response if any sentence is unsupported
            prob, label = None, int(y_i.any())
        timings = dict(scored.timings_ms)
        timings["classify"] = (time.perf_counter() - t0) * 1e3
        return Detection(label, prob, scored.sentences, c_hat.tolist(), y_i.tolist(),
                         feats.tolist(), scored.evidence, timings, self.classifier is not None)
