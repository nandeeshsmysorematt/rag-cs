"""Sentence segmentation (Sec. 4.3).

spaCy ``en_core_web_sm`` is used when available; otherwise a regex fallback is
used (with a warning). Sentences shorter than ``min_tokens`` are merged with the
immediately following sentence. A short trailing sentence is merged into the
previous one. A response that is only one short sentence is kept as-is, so only
empty / whitespace-only responses yield zero sentences.
"""
from __future__ import annotations

import re
import warnings
from typing import List, Optional, Tuple

_FALLBACK_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


class SentenceSegmenter:
    def __init__(self, spacy_model: Optional[str] = "en_core_web_sm", min_tokens: int = 5):
        self.min_tokens = min_tokens
        self.nlp = None
        if spacy_model:
            try:
                import spacy

                self.nlp = spacy.load(spacy_model, disable=["ner", "lemmatizer"])
            except (ImportError, OSError):
                warnings.warn(
                    f"spaCy model '{spacy_model}' unavailable; using regex sentence "
                    "splitter. Run: python -m spacy download en_core_web_sm"
                )

    def _raw_split(self, text: str) -> List[Tuple[str, int]]:
        if self.nlp is not None:
            spans = [(s.text.strip(), len(s)) for s in self.nlp(text).sents]
        else:
            spans = [(p.strip(), len(p.split())) for p in _FALLBACK_SPLIT.split(text)]
        return [(t, n) for t, n in spans if t]

    def split(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        merged: List[str] = []
        carry: Optional[Tuple[str, int]] = None
        for sent, n in self._raw_split(text):
            if carry is not None:
                sent, n = f"{carry[0]} {sent}", carry[1] + n
                carry = None
            if n < self.min_tokens:
                carry = (sent, n)
            else:
                merged.append(sent)
        if carry is not None:
            if merged:
                merged[-1] = f"{merged[-1]} {carry[0]}"
            else:
                merged.append(carry[0])
        return merged
