"""Contrôle : seuil de −0,5 point, mesures manquantes bloquantes, ORG du corpus et WikiANN indicatifs."""

import json

from masquener.gate import compare


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def gold(macro, blocking=True, dataset="conll"):
    return {"dataset": dataset, "blocking": blocking, "limit": None, "results": {"en": {"micro": {"f1": macro}}, "macro_f1": macro}}


def corpus(per, loc, org, masking):
    f = lambda v: {"f1": v}  # noqa: E731
    return {"corpus": "c.json", "strict": {"PER": f(per), "LOC": f(loc), "ORG": f(org)}, "masking_rate": masking}


def failures(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return [line for line in compare("reference", "new") if line.startswith("ÉCHEC")]


def test_threshold_half_point(tmp_path, monkeypatch):
    write(tmp_path / "reports/reference/conll.json", gold(0.900))
    write(tmp_path / "reports/new/conll.json", gold(0.896))
    assert failures(tmp_path, monkeypatch) == []
    write(tmp_path / "reports/new/conll.json", gold(0.894))
    assert len(failures(tmp_path, monkeypatch)) == 1


def test_missing_report_blocks(tmp_path, monkeypatch):
    write(tmp_path / "reports/reference/masakhaner.json", gold(0.44, dataset="masakhaner"))
    assert len(failures(tmp_path, monkeypatch)) == 1


def test_partial_measure_blocks(tmp_path, monkeypatch):
    write(tmp_path / "reports/reference/conll.json", gold(0.900))
    write(tmp_path / "reports/new/conll.json", {**gold(0.95), "limit": 100})
    assert len(failures(tmp_path, monkeypatch)) == 1


def test_corpus_org_and_wikiann_do_not_block(tmp_path, monkeypatch):
    write(tmp_path / "reports/local/reference/corpus-ci.json", corpus(0.8, 0.7, 0.6, 0.9))
    write(tmp_path / "reports/local/new/corpus-ci.json", corpus(0.8, 0.7, 0.1, 0.9))
    write(tmp_path / "reports/reference/wikiann.json", gold(0.56, blocking=False, dataset="wikiann"))
    write(tmp_path / "reports/new/wikiann.json", gold(0.40, blocking=False, dataset="wikiann"))
    assert failures(tmp_path, monkeypatch) == []
    write(tmp_path / "reports/local/new/corpus-ci.json", corpus(0.8, 0.7, 0.6, 0.89))
    assert len(failures(tmp_path, monkeypatch)) == 1


def test_synthetic_negatives_folder_counts_intact_sentences(tmp_path, monkeypatch):
    def synthetic(hit):
        neg = {"strict": {"micro": {"f1": 0.0}}, "masking_rate": None, "masking_precision": 0.0, "negative_cases": 100, "negative_cases_hit": hit}
        return {"dataset": "synthetic", "results": {"commun": neg}}

    write(tmp_path / "reports/reference/synthetic.json", synthetic(40))
    write(tmp_path / "reports/new/synthetic.json", synthetic(20))
    assert failures(tmp_path, monkeypatch) == []  # 60 % → 80 % de phrases intactes
    write(tmp_path / "reports/new/synthetic.json", synthetic(45))
    assert len(failures(tmp_path, monkeypatch)) == 1  # 60 % → 55 %


def test_corpus_masking_precision_blocks(tmp_path, monkeypatch):
    # Plus de noms trouvés, mais des mots ordinaires masqués à tort : refusé (cas de la v1).
    write(tmp_path / "reports/local/reference/corpus-ci.json", {**corpus(0.4, 0.6, 0.1, 0.7), "masking_precision": 0.90})
    write(tmp_path / "reports/local/new/corpus-ci.json", {**corpus(0.7, 0.65, 0.1, 0.85), "masking_precision": 0.84})
    assert len(failures(tmp_path, monkeypatch)) == 1
    # Mesure absente du nouveau rapport : bloquant aussi.
    write(tmp_path / "reports/local/new/corpus-ci.json", corpus(0.7, 0.65, 0.1, 0.85))
    assert len(failures(tmp_path, monkeypatch)) == 1
