"""Tests sur le modèle de référence téléchargé (python -m masquener.reference) ; ignorés s'il est absent."""

import json

import pytest

from masquener.evaluate import gold_spans
from masquener.labels import check_config
from masquener.onnx_ner import OnnxNer
from masquener.reference import REFERENCE_DIR

pytestmark = pytest.mark.skipif(not (REFERENCE_DIR / "config.json").exists(), reason="modèle de référence absent")


def test_reference_labels_unchanged():
    check_config(json.loads((REFERENCE_DIR / "config.json").read_text(encoding="utf-8")))


def test_spans_in_text_positions():
    text = "Rapport\nKouassi Yao est à Abidjan."
    spans = {(text[s:e], t) for s, e, t, _ in OnnxNer(REFERENCE_DIR).spans(text)}
    assert ("Kouassi Yao", "PER") in spans and ("Abidjan", "LOC") in spans


def test_gold_spans_every_occurrence_and_nested_dropped():
    case = {
        "id": "x",
        "text": "Abidjan, BP 12 Abidjan 08 ; Konan à Abidjan.",
        "entities": [
            {"text": "Abidjan", "type": "LIEU"},
            {"text": "BP 12 Abidjan 08", "type": "LIEU"},
            {"text": "Konan", "type": "PERSONNE"},
            {"text": "BP 12", "type": "REFERENCE"},
        ],
    }
    assert gold_spans(case) == {("LOC", 0, 7), ("LOC", 9, 25), ("PER", 28, 33), ("LOC", 36, 43)}
