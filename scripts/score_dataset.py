#!/usr/bin/env python
"""Run Stages 1-2 (retrieval + NLI) over a dataset and cache the raw NLI
probabilities. This is the only expensive step; classifier training, ablations
over k / tau / score variant / feature subsets then run on CPU in seconds.

Input JSONL fields: id, query, response, label (0/1), optional knowledge.
"""
import argparse
import json
import os

from tqdm import tqdm

from ragcs.config import RAGCSConfig
from ragcs.pipeline import RAGCS
from ragcs.utils import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--index-dir", default=None, help="FAISS index dir (global corpus)")
    ap.add_argument("--use-example-knowledge", action="store_true",
                    help="retrieve from each example's own 'knowledge' (e.g. FaithDial)")
    ap.add_argument("--k", type=int, default=5, help="score this many passages; ablate smaller k later")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--nli-input-order", choices=["sentence_first", "premise_first"],
                    default="sentence_first")
    ap.add_argument("--nli-batch-size", type=int, default=32)
    ap.add_argument("--no-fp16", action="store_true")
    args = ap.parse_args()

    if not args.index_dir and not args.use_example_knowledge:
        ap.error("provide --index-dir or --use-example-knowledge")

    cfg = RAGCSConfig(k=args.k, nli_input_order=args.nli_input_order,
                      nli_batch_size=args.nli_batch_size, fp16=not args.no_fp16)
    rag = RAGCS(cfg, index_dir=None if args.use_example_knowledge else args.index_dir)

    done = set()
    if os.path.exists(args.out):  # resume
        done = {r["id"] for r in read_jsonl(args.out)}

    with open(args.out, "a", encoding="utf-8") as out:
        for n, ex in enumerate(tqdm(read_jsonl(args.data), desc="scoring")):
            if args.limit and n >= args.limit:
                break
            if ex["id"] in done:
                continue
            knowledge = ex.get("knowledge") if args.use_example_knowledge else None
            sc = rag.score_sentences(ex.get("query", ""), ex["response"], knowledge, k=args.k)
            out.write(json.dumps({
                "id": ex["id"], "label": int(ex["label"]), "sentences": sc.sentences,
                "probs": [p.round(5).tolist() for p in sc.pair_probs],
                "passage_ids": [[p.id for p in ps] for ps in sc.evidence],
                "timings_ms": {k: round(v, 2) for k, v in sc.timings_ms.items()},
            }) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
