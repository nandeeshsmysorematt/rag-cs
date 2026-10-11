"""Stage 2: cross-encoder NLI scoring (Sec. 3.4)."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import RAGCSConfig
from .utils import resolve_device


class NLIScorer:
    """Returns softmax probabilities ordered [entailment, neutral, contradiction]."""

    def __init__(self, cfg: RAGCSConfig):
        if cfg.nli_input_order not in ("sentence_first", "premise_first"):
            raise ValueError("nli_input_order must be 'sentence_first' or 'premise_first'")
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.tok = AutoTokenizer.from_pretrained(cfg.nli_model, revision=cfg.nli_revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.nli_model, revision=cfg.nli_revision)
        if cfg.fp16 and self.device.startswith("cuda"):
            self.model.half()
        self.model.to(self.device).eval()

        # Never hard-code label order: read it from the checkpoint config.
        id2label = {int(i): str(l).lower() for i, l in self.model.config.id2label.items()}
        label2id = {v: k for k, v in id2label.items()}
        try:
            self._cols = [label2id["entailment"], label2id["neutral"], label2id["contradiction"]]
        except KeyError as e:
            raise ValueError(f"Unexpected NLI label set: {id2label}") from e

    def _truncate_passage(self, text: str) -> str:
        if len(text.split()) <= 200:  # 100-word passages never need truncation
            return text
        ids = self.tok(text, add_special_tokens=False, truncation=True,
                       max_length=self.cfg.max_passage_tokens)["input_ids"]
        return self.tok.decode(ids)

    @torch.no_grad()
    def probs(self, sentences: Sequence[str], passages: Sequence[str]) -> np.ndarray:
        """Score aligned (sentence, passage) pairs -> array of shape (n, 3)."""
        assert len(sentences) == len(passages)
        passages = [self._truncate_passage(p) for p in passages]
        out: List[np.ndarray] = []
        bs = self.cfg.nli_batch_size
        sf = self.cfg.nli_input_order == "sentence_first"
        for i in range(0, len(sentences), bs):
            s, p = list(sentences[i:i + bs]), list(passages[i:i + bs])
            a, b = (s, p) if sf else (p, s)
            trunc = "only_second" if sf else "only_first"  # keep the sentence intact
            kw = dict(max_length=self.cfg.max_nli_tokens, padding=True, return_tensors="pt")
            try:
                enc = self.tok(a, b, truncation=trunc, **kw)
            except ValueError:
                enc = self.tok(a, b, truncation="longest_first", **kw)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits.float()
            out.append(torch.softmax(logits, dim=-1)[:, self._cols].cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 3), dtype=np.float32)
