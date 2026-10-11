import numpy as np
import pytest

from ragcs.features import (document_features, pair_scores, sentence_consistency,
                            sentence_labels)


def test_pair_scores_variants():
    p = np.array([[0.7, 0.2, 0.1]])
    assert pair_scores(p, "signed")[0] == pytest.approx(0.6)
    assert pair_scores(p, "entail")[0] == pytest.approx(0.7)
    assert pair_scores(p, "one_minus_contra")[0] == pytest.approx(0.9)


def test_sentence_consistency_mean_and_k():
    sent = [np.array([[1, 0, 0], [0, 0, 1]], dtype=float)]  # +1, -1
    assert sentence_consistency(sent)[0] == pytest.approx(0.0)
    assert sentence_consistency(sent, k=1)[0] == pytest.approx(1.0)


def test_sentence_without_evidence_is_neutral():
    assert sentence_consistency([np.zeros((0, 3))])[0] == 0.0


def test_document_features_values():
    f = document_features(np.array([0.5, -0.5, 0.1]), tau=0.0)
    assert f[0] == pytest.approx(-0.5)                       # min
    assert f[1] == pytest.approx(0.1 / 3)                    # mean
    assert f[2] == pytest.approx(np.std([0.5, -0.5, 0.1]))   # population std
    assert f[3] == pytest.approx(1 / 3)                      # fraction below tau


def test_feature_subset_order():
    f = document_features(np.array([0.2, 0.4]), names=("mean", "min"))
    assert f.tolist() == pytest.approx([0.3, 0.2])


def test_sentence_labels_threshold():
    assert sentence_labels(np.array([-0.1, 0.0, 0.3]), tau=0.0).tolist() == [1, 0, 0]
