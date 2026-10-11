#!/usr/bin/env python
"""Run RAG-CS on a single (query, response) pair and print per-sentence results."""
import argparse
import json

from ragcs.classifier import HallucinationClassifier
from ragcs.config import RAGCSConfig
from ragcs.pipeline import RAGCS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--response", required=True)
    ap.add_argument("--index-dir", default=None, help="FAISS index (global corpus)")
    ap.add_argument("--knowledge-file", default=None, help="text file, one passage per line")
    ap.add_argument("--classifier", default=None, help="joblib from train_eval.py --model-out")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--nli-input-order", choices=["sentence_first", "premise_first"],
                    default="sentence_first")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    knowledge = None
    if args.knowledge_file:
        with open(args.knowledge_file, encoding="utf-8") as f:
            knowledge = [l.strip() for l in f if l.strip()]
    elif not args.index_dir:
        ap.error("provide --index-dir or --knowledge-file")

    clf = HallucinationClassifier.load(args.classifier) if args.classifier else None
    cfg = RAGCSConfig(k=args.k, tau=args.tau, nli_input_order=args.nli_input_order)
    rag = RAGCS(cfg, index_dir=None if knowledge else args.index_dir, classifier=clf)
    det = rag.detect(args.query, args.response, knowledge)

    if args.json:
        print(json.dumps({k: v for k, v in det.__dict__.items() if k != "evidence"}, indent=2))
        return
    for s, c, y in zip(det.sentences, det.sentence_scores, det.sentence_labels):
        print(f"[{'HALLUCINATED' if y else 'supported   '}] c_hat={c:+.3f}  {s}")
    prob = f"{det.probability:.3f}" if det.probability is not None else "n/a (no classifier)"
    print(f"\ndocument label: {det.label} ({'hallucinated' if det.label else 'faithful'}), p={prob}")
    print("timings (ms):", {k: round(v, 1) for k, v in det.timings_ms.items()})


if __name__ == "__main__":
    main()
