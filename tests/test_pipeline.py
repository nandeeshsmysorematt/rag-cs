"""Pipeline logic tested with fake retriever / NLI (no models, no GPU)."""
import numpy as np

from ragcs.classifier import HallucinationClassifier
from ragcs.config import RAGCSConfig
from ragcs.pipeline import RAGCS
from ragcs.schema import Passage
from ragcs.segmentation import SentenceSegmenter


class FakeRetriever:
    def search(self, questions, sentences, k, knowledge=None):
        return [[Passage(j, f"evidence {j}") for j in range(k)] for _ in sentences]


class FakeNLI:
    """Entails sentences containing 'true', contradicts those containing 'false'."""
    def probs(self, sentences, passages):
        rows = [[0.9, 0.05, 0.05] if "true" in s else [0.05, 0.05, 0.9] for s in sentences]
        return np.array(rows, dtype=np.float32).reshape(-1, 3)


def make(classifier=None):
    return RAGCS(RAGCSConfig(k=3), classifier=classifier, retriever=FakeRetriever(),
                 nli=FakeNLI(), segmenter=SentenceSegmenter(None, 3))


def test_empty_response_is_not_hallucinated():
    d = make().detect("q", "")
    assert d.label == 0 and d.sentences == []


def test_per_sentence_labels_and_fallback_label():
    d = make().detect("q", "This statement is true today. This statement is false today.")
    assert d.sentence_labels == [0, 1]
    assert d.label == 1 and d.probability is None and not d.used_classifier
    assert d.sentence_scores[0] > 0 > d.sentence_scores[1]


def test_trained_classifier_is_used():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 100)
    X = np.column_stack([np.where(y == 1, -0.8, 0.8)] * 4) + rng.normal(0, 0.1, (100, 4))
    clf = HallucinationClassifier().fit(X, y, tune=False)
    good = make(clf).detect("q", "This statement is true today.")
    bad = make(clf).detect("q", "This statement is false today.")
    assert good.used_classifier and good.label == 0 and bad.label == 1
