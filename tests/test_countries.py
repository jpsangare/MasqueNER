"""Cohérence des dossiers pays : listes déclarées présentes, cases des modèles connues, prénoms et noms en mots isolés."""

import re
from pathlib import Path

import pytest
import yaml

COUNTRIES = sorted(p for p in Path("countries").iterdir() if (p / "country.yaml").exists())
SLOT = re.compile(r"\{([^}]+)\}")


def entries(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


PARTICLES = {"de", "du", "des", "d", "la", "le", "les", "van", "von", "da", "di", "del", "dos", "ben", "el", "al", "mac", "mc"}


def single_name(entry: str, particles: bool) -> bool:
    words = entry.split()
    if any(c.isdigit() for c in entry) or not words:
        return False
    return len(words) == 1 or (particles and all(w.lower().rstrip("'") in PARTICLES for w in words[:-1]))


@pytest.mark.parametrize("country", COUNTRIES, ids=lambda p: p.name)
def test_country_folder(country: Path):
    config = yaml.safe_load((country / "country.yaml").read_text(encoding="utf-8"))
    assert set(config["lists"].values()) <= {"PER", "LOC", "ORG"}
    for name in config["lists"]:
        assert entries(country / "lists" / f"{name}.txt"), f"liste vide : {name}"
    for name in ("prenoms", "noms"):
        if name not in config["lists"]:
            continue
        # Un seul mot ; pour un nom de famille, des particules peuvent le précéder (« Le Gall », « Da Silva »).
        bad = [e for e in entries(country / "lists" / f"{name}.txt") if not single_name(e, particles=name == "noms")]
        assert not bad, f"{name} : un seul mot par ligne ({bad[:5]})"

    allowed = {"PER", "DATE", "NUM"} | {f"{t}.{n}" for n, t in config["lists"].items() if t != "PER"}
    for lang in config["languages"]:
        for line in entries(country / "templates" / f"{lang}.txt"):
            unknown = set(SLOT.findall(line)) - allowed
            assert not unknown, f"case inconnue {unknown} dans : {line}"
