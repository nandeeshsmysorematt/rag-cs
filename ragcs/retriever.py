"""Stage 1: dense evidence retrieval (Sec. 3.3).

Query for sentence s_i is the pair (q, s_i) encoded jointly by the DPR question
encoder (Eq. 1-2). Passages are encoded offline by the DPR context encoder and
searched with an exact FAISS ``IndexFlatIP`` (Eq. 3-4).
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizerFast,
)

from .config import RAGCSConfig
from .schema import Passage
from .utils import resolve_device


class DPREncoders:
    """Lazily loads the DPR question and/or context encoder."""

    def __init__(self, cfg: RAGCSConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.dtype = torch.float16 if (cfg.fp16 and self.device.startswith("cuda")) else torch.float32
        self._q_tok = self._q_enc = self._c_tok = self._c_enc = None

    def _ensure_q(self):
        if self._q_enc is None:
            c = self.cfg
            self._q_tok = DPRQuestionEncoderTokenizerFast.from_pretrained(
                c.dpr_question_model, revision=c.dpr_question_revision)
            self._q_enc = DPRQuestionEncoder.from_pretrained(
                c.dpr_question_model, revision=c.dpr_question_revision, torch_dtype=self.dtype
            ).to(self.device).eval()

    def _ensure_c(self):
        if self._c_enc is None:
            c = self.cfg
            self._c_tok = DPRContextEncoderTokenizerFast.from_pretrained(
                c.dpr_context_model, revision=c.dpr_context_revision)
            self._c_enc = DPRContextEncoder.from_pretrained(
                c.dpr_context_model, revision=c.dpr_context_revision, torch_dtype=self.dtype
            ).to(self.device).eval()

    @torch.no_grad()
    def encode_queries(self, questions: Sequence[str], sentences: Sequence[str],
                       batch_size: int = 64) -> np.ndarray:
        """Encode q_i = [CLS] q [SEP] s_i [SEP]; q is truncated first, s_i preserved."""
        self._ensure_q()
        out = []
        for i in range(0, len(sentences), batch_size):
            qs, ss = list(questions[i:i + batch_size]), list(sentences[i:i + batch_size])
            kw = dict(max_length=self.cfg.max_query_tokens, padding=True, return_tensors="pt")
            try:
                enc = self._q_tok(qs, ss, truncation="only_first", **kw)
            except ValueError:  # sentence alone exceeds the budget
                enc = self._q_tok(qs, ss, truncation="longest_first", **kw)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out.append(self._q_enc(**enc).pooler_output.float().cpu().numpy())
        if not out:
            return np.zeros((0, self.cfg.embedding_dim), dtype=np.float32)
        return np.concatenate(out).astype(np.float32)

    @torch.no_grad()
    def encode_passages(self, titles: Sequence[str], texts: Sequence[str],
                        batch_size: int = 128) -> np.ndarray:
        self._ensure_c()
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self._c_tok(
                list(titles[i:i + batch_size]), list(texts[i:i + batch_size]),
                truncation=True, max_length=self.cfg.max_passage_encode_tokens,
                padding=True, return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out.append(self._c_enc(**enc).pooler_output.float().cpu().numpy())
        if not out:
            return np.zeros((0, self.cfg.embedding_dim), dtype=np.float32)
        return np.concatenate(out).astype(np.float32)


class PassageStore:
    """Random access to passages in a JSONL file via a byte-offset table."""

    def __init__(self, jsonl_path: str, offsets_path: str):
        self.offsets = np.load(offsets_path, mmap_mode="r")
        self._fh = open(jsonl_path, "rb")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, i: int) -> dict:
        self._fh.seek(int(self.offsets[i]))
        return json.loads(self._fh.readline())


class DenseRetriever:
    """Top-k retrieval from a FAISS index, or from a per-example passage list."""

    def __init__(self, encoders: DPREncoders, index=None, store: Optional[PassageStore] = None):
        self.encoders = encoders
        self.index = index
        self.store = store

    @classmethod
    def load(cls, cfg: RAGCSConfig, index_dir: str) -> "DenseRetriever":
        import faiss

        with open(os.path.join(index_dir, "meta.json")) as f:
            meta = json.load(f)
        index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        store = PassageStore(meta["passages_path"], os.path.join(index_dir, "offsets.npy"))
        return cls(DPREncoders(cfg), index, store)

    @classmethod
    def local_only(cls, cfg: RAGCSConfig) -> "DenseRetriever":
        """No global index: only usable with ``knowledge=`` passed to ``search``."""
        return cls(DPREncoders(cfg))

    def search(self, questions: Sequence[str], sentences: Sequence[str], k: int,
               knowledge: Optional[Sequence[Tuple[str, str]]] = None) -> List[List[Passage]]:
        if knowledge is not None:
            return self._rank_local(questions, sentences, k, knowledge)
        if self.index is None:
            raise RuntimeError("No FAISS index loaded; pass index_dir or per-example knowledge.")
        emb = np.ascontiguousarray(self.encoders.encode_queries(questions, sentences), dtype=np.float32)
        scores, ids = self.index.search(emb, k)
        results = []
        for srow, irow in zip(scores, ids):
            ps = []
            for s, i in zip(srow, irow):
                if i < 0:
                    continue
                rec = self.store[int(i)]
                ps.append(Passage(int(i), rec["text"], rec.get("title", ""), float(s)))
            results.append(ps)
        return results

    def _rank_local(self, questions, sentences, k, knowledge) -> List[List[Passage]]:
        """Rank a small provided passage list with the same DPR inner-product score."""
        titles = [t for t, _ in knowledge]
        texts = [x for _, x in knowledge]
        if not texts:
            return [[] for _ in sentences]
        ctx = self.encoders.encode_passages(titles, texts)
        q = self.encoders.encode_queries(questions, sentences)
        sims = q @ ctx.T
        results = []
        for row in sims:
            top = np.argsort(-row)[:k]
            results.append([Passage(int(j), texts[j], titles[j], float(row[j])) for j in top])
        return results
