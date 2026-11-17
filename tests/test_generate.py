"""Générateur : positions justes, jeux étanches (modèles et noms), sortie reproductible."""

import random
import re
from pathlib import Path

from masquener.generate import SLOT, SPLITS, Country, fill, generate, load_country

CI = Path("countries/ci")


def test_entities_point_at_their_values():
    for row in generate("train", 300, seed=1):
        text = row["text"]
        assert not SLOT.search(text) and "<" not in text, text  # aucune case ni morceau de nom laissé vide
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


def test_multiline_template_and_comma_name():
    country = Country("xx", {"noms": ["Konan"], "prenoms": ["Aya"]}, {"noms": "PER", "prenoms": "PER"}, {"{NOM}, {prenoms}": 1}, {})
    text, spans = fill(r"Nom : {PER}\nMatricule : 1234\nCordialement", country, "fr", random.Random(0))
    assert text == "Nom : KONAN, Aya\nMatricule : 1234\nCordialement"
    assert [text[s:e] for s, e, _ in spans] == ["KONAN, Aya"]


def test_uppercase_sentences_keep_their_entities_and_none_is_lowercased():
    rows = generate("train", 2000, seed=2)
    upper = [r for r in rows if r["text"] == r["text"].upper() and r["entities"]]
    assert upper, "aucune phrase en capitales"
    assert not [r for r in rows if r["text"] == r["text"].lower() and r["entities"]]
    for row in upper:
        for start, end, _ in row["entities"]:
            assert row["text"][start:end] == row["text"][start:end].strip() != ""


def test_multiline_documents_and_negatives_are_generated():
    rows = generate("train", 3000, seed=0)
    assert any("\n" in r["text"] and r["entities"] for r in rows)
    assert any("\n" in r["text"] and not r["entities"] for r in rows)


def test_elision_before_vowel():
    country = Country("xx", {"institutions": ["Ageroute"]}, {"institutions": "ORG"}, {"{nom}": 1}, {})
    text, spans = fill("Le dossier de {ORG.institutions} est prêt.", country, "fr", random.Random(0))
    assert text == "Le dossier d'Ageroute est prêt."
    assert [text[s:e] for s, e, _ in spans] == ["Ageroute"]


def test_composed_slots_and_initials():
    rows = generate("train", 4000, seed=3)
    composed = [r for r in rows if "{ORG:" in r["template"] or "{LOC:" in r["template"] or "{PER:" in r["template"]]
    assert composed, "aucune case composée"
    for row in composed:
        assert "<" not in row["text"]
    # Initiale d'un prénom dans une entité : « K. KONAN », « Kassi J.-P. ».
    initials = re.compile(r"(?<![\w.])\w\.(?:-\w\.)?(?= \w)|(?<=\w )\w\.(?:-\w\.)?$")
    assert any(initials.search(row["text"][s:e]) for row in rows for s, e, t in row["entities"] if t == "PER")


def test_recased_sentences_keep_their_entities():
    from masquener.generate import recase

    assert recase("Appelle Sophie chez Ford demain.") == "Appelle Sophie chez Ford Demain."
    assert recase("Le dossier de N'Guessan") == "le Dossier de n'guessan"  # mots outils et élisions en minuscules
    rows = generate("train", 3000, seed=4)
    recased = [r for r in rows if r["entities"] and r["text"] == recase(r["text"]) and r["text"] != r["text"].upper()]
    assert recased, "aucune phrase remise en majuscules"
    for row in recased:
        for start, end, _ in row["entities"]:
            assert row["text"][start:end] == row["text"][start:end].strip() != ""
