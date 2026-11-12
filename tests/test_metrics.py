from masquener.labels import ID2LABEL, LABEL2ID, check_config
from masquener.metrics import bio_spans, score

import pytest


def test_bio_spans_lenient_like_conlleval():
    tags = ["B-PER", "I-PER", "O", "I-LOC", "B-LOC", "I-ORG", "B-ORG"]
    assert bio_spans(tags) == {("PER", 0, 2), ("LOC", 3, 4), ("LOC", 4, 5), ("ORG", 5, 6), ("ORG", 6, 7)}


def test_score_exact_match_per_type():
    gold = [{("PER", 0, 2), ("LOC", 3, 4)}]
    pred = [{("PER", 0, 2), ("LOC", 3, 5), ("DATE", 7, 8)}]
    s = score(gold, pred, ("PER", "LOC"))
    assert s["PER"]["f1"] == 1.0
    assert s["LOC"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}
    assert s["micro"]["precision"] == 0.5 and s["micro"]["recall"] == 0.5


def test_labels_are_the_reference_ones():
    check_config({"id2label": {str(k): v for k, v in ID2LABEL.items()}, "label2id": LABEL2ID})
    with pytest.raises(ValueError):
        check_config({"id2label": {"0": "O", "1": "B-PER"}, "label2id": {"O": 0, "B-PER": 1}})


def test_token_labels_from_char_spans():
    from masquener.labels import IGNORE, token_labels

    # « [CLS] KON ##AN Yao à Abidjan [SEP] » ; entités : « KONAN Yao » (PER), « Abidjan » (LOC)
    offsets = [(0, 0), (0, 3), (3, 5), (6, 9), (10, 11), (12, 19), (0, 0)]
    labels = token_labels(offsets, [[0, 9, "PER"], [12, 19, "LOC"]])
    names = [ID2LABEL[i] if i != IGNORE else None for i in labels]
    assert names == [None, "B-PER", "I-PER", "I-PER", "O", "B-LOC", None]
