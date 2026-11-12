"""Générateur : positions justes, jeux étanches (modèles et noms), sortie reproductible."""

import random
from pathlib import Path

from masquener.generate import SPLITS, Country, fill, generate, load_country

CI = Path("countries/ci")


def test_entities_point_at_their_values():
    for row in generate("train", 300, seed=1):
        text = row["text"]
        assert "{" not in text, text
        for start, end, kind in row["entities"]:
            value = text[start:end]
            assert value == value.strip() and value, (text, start, end)
            assert kind in {"PER", "LOC", "ORG"}


def test_splits_share_no_template_and_no_name():
    parts = {s: load_country(CI, s) for s in SPLITS}
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                assert not set(parts[a].templates["fr"]) & set(parts[b].templates["fr"])
                for name in ("prenoms", "noms", "villes", "entreprises"):
                    assert not set(parts[a].lists[name]) & set(parts[b].lists[name]), name


def test_same_seed_same_sentences():
    assert generate("dev", 50, seed=3) == generate("dev", 50, seed=3)
    assert generate("dev", 50, seed=3) != generate("dev", 50, seed=4)


def test_elision_before_vowel():
    country = Country("xx", {"institutions": ["Ageroute"]}, {"institutions": "ORG"}, {"{nom}": 1}, {})
    text, spans = fill("Le dossier de {ORG.institutions} est prêt.", country, "fr", random.Random(0))
    assert text == "Le dossier d'Ageroute est prêt."
    assert [text[s:e] for s, e, _ in spans] == ["Ageroute"]
