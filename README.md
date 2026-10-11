# RAG-CS: Retrieval-Augmented Consistency Scoring

Reference implementation of **"Retrieval augmented consistency scoring for model agnostic
hallucination detection in large language models"** (Nandeesh S Mysore Matt, Pavithra N, Preethi).

RAG-CS is a post-hoc, black-box detector. It needs only the query `q` and the LLM response `r`:

1. **Segment** `r` into sentences (spaCy `en_core_web_sm`, sentences < 5 tokens merged forward).
2. **Retrieve** top-`k` passages per sentence with DPR (query = `[q ; s_i]`) over a FAISS `IndexFlatIP`.
3. **Score** each (sentence, passage) pair with `cross-encoder/nli-deberta-v3-large`:
   `c_ij = pi_E - pi_C`, then `c_i = mean_j c_ij`.
4. **Aggregate** `[min, mean, std, frac(c_i < tau)]` and classify with logistic regression;
   per-sentence labels are `c_i < tau`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pip install -e .
```

For the full 21M-passage index use a GPU build of FAISS (`conda install -c pytorch faiss-gpu`)
and budget ~64 GB disk / ~58 GB RAM (see paper, Sec. 4.3).

## Quickstart (no big index needed)

Verify a response against a few passages you supply:

```bash
python scripts/detect.py \
  --query "What is the mechanism of action of aspirin?" \
  --response "Aspirin irreversibly inhibits COX enzymes, preventing prostaglandin synthesis. It also directly inhibits the serotonin receptor 5-HT2A, which contributes to its analgesic effects." \
  --knowledge-file data/aspirin_knowledge.txt
```

Without a trained classifier, the document label falls back to "any sentence has `c_i < tau`".

## Full pipeline

```bash
# 1. Corpus: 100-word passages, 20-word overlap (or convert DPR's psgs_w100.tsv)
python scripts/prepare_corpus.py --input wiki_articles.jsonl --output passages.jsonl
# 2. Index (exact inner-product search)
python scripts/build_index.py --passages passages.jsonl --out-dir indices/wiki
# 3. Dataset in the unified JSONL format (see below), e.g. HaluEval-QA
python scripts/prepare_halueval.py --input qa_data.json --output data/halueval_qa.jsonl
# 4. Retrieval + NLI (the only GPU-heavy step; resumable). Score at k=20 to allow k-ablations.
python scripts/score_dataset.py --data data/halueval_qa.jsonl --out scores/halueval_qa.jsonl \
       --index-dir indices/wiki --k 20
# 5. Train head on 2,000 examples, evaluate on the rest (CPU, seconds)
python scripts/train_eval.py --scores scores/halueval_qa.jsonl --n-train 2000 --k 5 \
       --model-out artifacts/halueval_clf.joblib --metrics-out artifacts/halueval_metrics.json
```

`train_eval.py` also covers the Table 4 ablations without re-running NLI:
`--k`, `--tau`, `--score-variant {signed,entail,one_minus_contra}`, `--features min,mean,...`.

### Unified dataset format (JSONL)

```json
{"id": "ex-1", "query": "...", "response": "...", "label": 1, "knowledge": ["optional passage", "..."]}
```

`label` = 1 means hallucinated. `knowledge` is used only with `--use-example-knowledge`
(FaithDial-style, where the task provides its own knowledge source).
Converters are included for HaluEval-QA only; TruthfulQA (responses must be generated with
GPT-3.5-turbo), FEVER and FaithDial need a small converter to this format.

## Python API

```python
from ragcs import RAGCS, RAGCSConfig
from ragcs.classifier import HallucinationClassifier

rag = RAGCS(RAGCSConfig(k=5), index_dir="indices/wiki",
            classifier=HallucinationClassifier.load("artifacts/halueval_clf.joblib"))
det = rag.detect("Who wrote Hamlet?", "Hamlet was written by Christopher Marlowe in 1599.")
print(det.label, det.probability, det.sentence_scores, det.sentence_labels)
```

## Things to check before reporting results

* **NLI input order.** The paper specifies `[CLS] s_i [SEP] p_j [SEP]` (sentence first), which is the
  default here (`nli_input_order="sentence_first"`). NLI checkpoints are trained as
  *premise then hypothesis*, so the standard direction is passage first. Both are implemented
  (`--nli-input-order premise_first`); compare them on a dev split and keep the paper text and code consistent.
* **Retriever checkpoint.** The paper's text says the bi-encoder is pre-trained on MS-MARCO, but the
  implementation details use the Natural Questions DPR checkpoints; the code follows the latter.
* **Pin model revisions.** Set `dpr_*_revision` / `nli_revision` in `ragcs/config.py` to commit hashes.
* **Not included:** baseline re-implementations (Perplexity, ROUGE-L, BERTScore, SelfCheckGPT, SAPLMA,
  FActScore), McNemar tests, and the paper's figures. No result in the paper is reproduced by the
  tests; the tests cover logic only.

## Tests

```bash
pip install pytest && PYTHONPATH=. pytest -q   # no GPU / model downloads required
```

## Layout

```
ragcs/        config, segmentation, retriever (DPR + FAISS), nli, features, classifier, pipeline
scripts/      prepare_corpus, build_index, prepare_halueval, score_dataset, train_eval, detect
tests/        unit tests with fake retriever / NLI
data/         tiny demo inputs
```
