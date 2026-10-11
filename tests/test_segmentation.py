from ragcs.segmentation import SentenceSegmenter

seg = SentenceSegmenter(spacy_model=None, min_tokens=5)  # regex fallback


def test_empty_response_has_no_sentences():
    assert seg.split("   ") == []


def test_basic_split():
    text = "The Eiffel Tower is in Paris France. It was completed in the year 1889."
    assert len(seg.split(text)) == 2


def test_short_sentence_merges_with_next():
    out = seg.split("Yes. The Eiffel Tower is located in the city of Paris.")
    assert len(out) == 1 and out[0].startswith("Yes.")


def test_short_trailing_sentence_merges_with_previous():
    out = seg.split("The Eiffel Tower is located in the city of Paris. Yes.")
    assert len(out) == 1 and out[0].endswith("Yes.")


def test_single_short_response_kept():
    assert seg.split("Paris.") == ["Paris."]
