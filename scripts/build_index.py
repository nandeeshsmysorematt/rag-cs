#!/usr/bin/env python
"""Encode passages with the DPR context encoder and build a FAISS IndexFlatIP.

Output directory: index.faiss, offsets.npy (byte offset of each passage line),
meta.json (path to the passages JSONL). The full 21M-passage Wikipedia index
needs ~64 GB disk / ~58 GB RAM and ~24 GPU-hours (paper, Sec. 4.3).
"""
import argparse
import json
import os
from array import array

import faiss
import numpy as np
from tqdm import tqdm

from ragcs.config import RAGCSConfig
from ragcs.retriever import DPREncoders
from ragcs.utils import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", required=True, help="JSONL from prepare_corpus.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-passages", type=int, default=None, help="debug: stop early")
    args = ap.parse_args()

    cfg = RAGCSConfig()
    set_seed(cfg.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    enc = DPREncoders(cfg)
    index = faiss.IndexFlatIP(cfg.embedding_dim)
    offsets = array("q")
    titles, texts = [], []

    def flush():
        if texts:
            index.add(np.ascontiguousarray(enc.encode_passages(titles, texts), dtype=np.float32))
            titles.clear()
            texts.clear()

    pos = 0
    with open(args.passages, "rb") as f:
        for line in tqdm(f, desc="encoding"):
            start, pos = pos, pos + len(line)
            if not line.strip():
                continue
            rec = json.loads(line)
            offsets.append(start)
            titles.append(rec.get("title", ""))
            texts.append(rec["text"])
            if len(texts) >= args.batch_size:
                flush()
            if args.max_passages and len(offsets) >= args.max_passages:
                break
    flush()

    faiss.write_index(index, os.path.join(args.out_dir, "index.faiss"))
    np.save(os.path.join(args.out_dir, "offsets.npy"), np.frombuffer(offsets, dtype=np.int64))
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({"passages_path": os.path.abspath(args.passages),
                   "dim": cfg.embedding_dim, "num_passages": index.ntotal}, f, indent=2)
    print(f"indexed {index.ntotal} passages -> {args.out_dir}")


if __name__ == "__main__":
    main()
