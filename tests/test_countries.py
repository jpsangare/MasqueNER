"""Cohérence des dossiers pays : listes déclarées présentes, cases des modèles connues, prénoms et noms en mots isolés."""

import re
from pathlib import Path

import pytest
import yaml

COUNTRIES = sorted(p for p in Path("countries").iterdir() if (p / "country.yaml").exists())
SLOT = re.compile(r"\{([^}]+)\}")


def entries(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


@pytest.mark.parametrize("country", COUNTRIES, ids=lambda p: p.name)
def test_country_folder(country: Path):
    config = yaml.safe_load((country / "country.yaml").read_text(encoding="utf-8"))
    assert set(config["lists"].values()) <= {"PER", "LOC", "ORG"}
    for name in config["lists"]:
        assert entries(country / "lists" / f"{name}.txt"), f"liste vide : {name}"
    for name in ("prenoms", "noms"):
        bad = [e for e in entries(country / "lists" / f"{name}.txt") if " " in e or any(c.isdigit() for c in e)]
        assert not bad, f"{name} : un seul mot par ligne ({bad[:5]})"

    allowed = {"PER", "DATE", "NUM"} | {f"{t}.{n}" for n, t in config["lists"].items() if t != "PER"}
    for lang in config["languages"]:
        for line in entries(country / "templates" / f"{lang}.txt"):
            unknown = set(SLOT.findall(line)) - allowed
            assert not unknown, f"case inconnue {unknown} dans : {line}"
