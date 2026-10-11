#!/usr/bin/env python
"""Train the logistic-regression head on a labeled subset and evaluate on the rest
(Sec. 4.3 'Classifier Training' / 'Evaluation Protocol'). CPU only.

Ablation switches reproduce the study designs of Table 4 without re-running NLI:
  --k {1,3,5,10,20}                         retrieval depth (needs scores made with k >= value)
  --score-variant {signed,entail,one_minus_contra}
  --features min,mean,std,frac_below        any subset
  --tau                                     neutrality threshold
"""
import argparse
import json

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ragcs.classifier import HallucinationClassifier
from ragcs.features import FEATURE_NAMES, SCORE_VARIANTS, document_features, sentence_consistency
from ragcs.utils import read_jsonl, set_seed


def featurize(ex, k, variant, tau, names):
    pp = [np.asarray(p, dtype=float).reshape(-1, 3) for p in ex["probs"]]
    if not pp:
        return None  # empty response
    return document_features(sentence_consistency(pp, k, variant), tau, names)


def metrics(y, prob, pred):
    out = {"f1": f1_score(y, pred, pos_label=1, zero_division=0),
           "precision": precision_score(y, pred, pos_label=1, zero_division=0),
           "recall": recall_score(y, pred, pos_label=1, zero_division=0),
           "n": int(len(y))}
    out["auroc"] = float(roc_auc_score(y, prob)) if len(set(y)) > 1 else None
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="output of score_dataset.py")
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--score-variant", choices=SCORE_VARIANTS, default="signed")
    ap.add_argument("--features", default=",".join(FEATURE_NAMES))
    ap.add_argument("--no-tune", action="store_true", help="fix C=1.0 instead of 5-fold CV")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-out", default=None)
    ap.add_argument("--metrics-out", default=None)
    ap.add_argument("--load-classifier", default=None, help="skip training; evaluate on all data")
    args = ap.parse_args()

    set_seed(args.seed)
    names = tuple(n.strip() for n in args.features.split(","))
    data = list(read_jsonl(args.scores))
    feats = [featurize(ex, args.k, args.score_variant, args.tau, names) for ex in data]
    y_all = np.array([ex["label"] for ex in data])
    valid = np.array([f is not None for f in feats])
    X_all = np.stack([f if f is not None else np.zeros(len(names)) for f in feats])
    idx = np.arange(len(data))

    if args.load_classifier:
        clf = HallucinationClassifier.load(args.load_classifier)
        eval_idx = idx
    else:
        train_idx, eval_idx = train_test_split(
            idx, train_size=args.n_train, stratify=y_all, random_state=args.seed)
        train_idx = train_idx[valid[train_idx]]  # empty responses carry no signal
        clf = HallucinationClassifier(C=1.0, seed=args.seed, feature_names=names)
        clf.fit(X_all[train_idx], y_all[train_idx], tune=not args.no_tune)
        if args.model_out:
            clf.save(args.model_out)

    prob = np.where(valid[eval_idx], clf.predict_proba(X_all[eval_idx]), 0.0)
    pred = (prob >= 0.5).astype(int)  # empty response -> y_hat = 0
    report = {"config": vars(args), "cv": clf.cv_results_, "C": clf.C,
              "eval": metrics(y_all[eval_idx], prob, pred)}
    print(json.dumps(report, indent=2, default=str))
    if args.metrics_out:
        with open(args.metrics_out, "w") as f:
            json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
    main()
