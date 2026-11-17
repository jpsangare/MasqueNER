"""Test par famille : lecture du fichier, entités bien placées, familles négatives et positives présentes."""

from pathlib import Path

from masquener.evaluate import parse_families


def test_family_file_parses_with_exact_positions():
    families = parse_families(Path("familles/familles.txt"))
    assert len(families) >= 15
    for name, rows in families.items():
        assert rows, f"famille vide : {name}"
        kinds = {bool(spans) for _, spans in rows}
        assert len(kinds) == 1, f"{name} : phrases positives et négatives mêlées"
        for text, spans in rows:
            assert "[" not in text or "](" not in text
            for _, start, end in spans:
                assert text[start:end] == text[start:end].strip() != ""
    assert any(not rows[0][1] for rows in families.values()), "aucune famille négative"


def test_markup_and_newlines(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("# x\n== a ==\nDe : [Aya Koné](PER)\\nÀ : [Daloa](LOC)\n== b ==\nBonjour,\n", encoding="utf-8")
    families = parse_families(path)
    text, spans = families["a"][0]
    assert text == "De : Aya Koné\nÀ : Daloa"
    assert sorted((text[s:e], t) for t, s, e in spans) == [("Aya Koné", "PER"), ("Daloa", "LOC")]
    assert families["b"] == [("Bonjour,", set())]
