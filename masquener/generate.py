"""Phrases synthétiques étiquetées, à partir des dossiers pays (countries/<code>/).

    python -m masquener.generate [--train 20000] [--dev 1000] [--test 2000] [--seed 0]

Chaque modèle de phrase voit ses cases remplies au hasard ; on connaît donc la position exacte de chaque entité.
Un modèle peut tenir sur plusieurs lignes (« \n » : fiche, courriel, en-tête), et une petite part des phrases est
écrite tout en minuscules (LOWERCASE_RATE).
Sortie : data/synthetic/{train,dev,test}.jsonl, une phrase par ligne :
    {"id", "country", "lang", "template", "text", "entities": [[début, fin, type], ...]}
(positions en caractères, fin exclue ; types PER, LOC, ORG).

Dates et nombres sont écrits mais non étiquetés : ils relèvent de règles à part, et le modèle de départ ne
prédit jamais DATE (la distillation lui donne donc « O » partout). Ils servent d'exemples de ce qu'il ne faut pas
prendre pour un nom.

Séparation : chaque modèle de phrase et chaque ligne de liste appartient à un seul jeu (train, dev ou test), tiré
d'un hachage stable. Le test ne contient donc que des modèles et des noms jamais vus à l'entraînement : il mesure la
généralisation, pas la mémoire. Une liste trop courte pour avoir des lignes dans chaque jeu est partagée en entier.
"""

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

COUNTRIES = Path("countries")
OUT = Path("data/synthetic")
SPLITS = ("train", "dev", "test")
SLOT = re.compile(r"\{([^}]+)\}")
PERSON_PART = re.compile(r"\{(nom|NOM|prenom|prenoms)\}")
# Nombre de prénoms pour {prenoms} et poids : un prénom le plus souvent, parfois deux ou trois.
GIVEN_COUNT = {1: 5, 2: 4, 3: 1}
# Part des phrases écrites tout en minuscules (messages tapés vite : « appelle james chez ford »), entités comprises.
LOWERCASE_RATE = 0.05
# Dans un modèle de phrase, « \n » marque un retour à la ligne : fiches, courriels, en-têtes sur plusieurs lignes.
NEWLINE = "\\n"

MONTHS = {
    "fr": "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split(),
}
WEEKDAYS = {"fr": "lundi mardi mercredi jeudi vendredi samedi dimanche".split()}
# Élision devant une valeur qui commence par une voyelle : « de Ageroute » → « d'Ageroute ».
ELISION = {"fr": re.compile(r"(?<![\w'])([dDlLjJ]e|[lL]a|[qQ]ue) $")}
VOWEL = re.compile(r"^[AEIOUYÀÂÉÈÊËÎÏÔÛÙaeiouyàâéèêëîïôûù]")


def bucket(*parts: str) -> str:
    """Jeu (train / dev / test) d'un élément, stable d'une exécution à l'autre : 80 / 10 / 10."""
    n = int(hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:8], 16) % 10
    return "test" if n == 0 else "dev" if n == 1 else "train"


def entries(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


@dataclass
class Country:
    code: str
    lists: dict[str, list[str]]  # nom de liste → lignes du jeu
    types: dict[str, str]  # nom de liste → type d'entité
    person_formats: dict[str, int]
    templates: dict[str, list[str]]  # langue → modèles du jeu


def load_country(folder: Path, split: str) -> Country:
    """Dossier pays restreint aux modèles et lignes de listes d'un jeu."""
    config = yaml.safe_load((folder / "country.yaml").read_text(encoding="utf-8"))
    code = config["code"]
    lists = {}
    for name in config["lists"]:
        values = entries(folder / "lists" / f"{name}.txt")
        parts = {s: [v for v in values if bucket(code, name, v) == s] for s in SPLITS}
        lists[name] = parts[split] if all(parts.values()) else values
    templates = {}
    for lang in config["languages"]:
        if lang not in MONTHS:
            raise ValueError(f"{code} : langue {lang} sans noms de mois dans masquener/generate.py")
        templates[lang] = [t for t in entries(folder / "templates" / f"{lang}.txt") if bucket(code, lang, t) == split]
    return Country(code, lists, config["lists"], config["person_formats"], templates)


def person(country: Country, rng: random.Random) -> str:
    fmt = rng.choices(list(country.person_formats), weights=list(country.person_formats.values()))[0]

    def part(m: re.Match) -> str:
        kind = m.group(1)
        if kind in ("nom", "NOM"):
            name = rng.choice(country.lists["noms"])
            return name.upper() if kind == "NOM" else name
        k = 1 if kind == "prenom" else rng.choices(list(GIVEN_COUNT), weights=list(GIVEN_COUNT.values()))[0]
        return " ".join(rng.sample(country.lists["prenoms"], k))

    return PERSON_PART.sub(part, fmt)


def date(lang: str, rng: random.Random) -> str:
    day, month, year = rng.randint(1, 28), rng.choice(MONTHS[lang]), rng.randint(2015, 2027)
    first = "1er" if day == 1 and lang == "fr" else str(day)
    return rng.choices(
        [f"{first} {month} {year}", f"{month} {year}", f"{rng.choice(WEEKDAYS[lang])} {first} {month} {year}"],
        weights=[6, 2, 2],
    )[0]


def fill(template: str, country: Country, lang: str, rng: random.Random) -> tuple[str, list[list]]:
    """Remplit les cases ; renvoie le texte et ses entités (positions en caractères)."""
    template = template.replace(NEWLINE, "\n")
    text, spans, at = "", [], 0
    for m in SLOT.finditer(template):
        text += template[at : m.start()]
        at = m.end()
        slot = m.group(1)
        if slot == "PER":
            kind, value = "PER", person(country, rng)
        elif slot == "DATE":
            kind, value = None, date(lang, rng)
        elif slot == "NUM":
            kind, value = None, str(rng.randint(10, 999999))
        else:
            kind, name = slot.split(".")
            value = rng.choice(country.lists[name])
        if VOWEL.match(value) and (e := ELISION[lang].search(text)):
            text = text[: e.start()] + e.group(1)[:-1] + "'"
        if kind:
            spans.append([len(text), len(text) + len(value), kind])
        text += value
    return text + template[at:], spans


def generate(split: str, n: int, seed: int) -> list[dict]:
    """n phrases d'un jeu, réparties à parts égales entre les pays puis entre leurs langues."""
    folders = sorted(p for p in COUNTRIES.iterdir() if (p / "country.yaml").exists())
    countries = [load_country(folder, split) for folder in folders]
    groups = [(c, lang) for c in countries for lang in sorted(c.templates)]
    rows = []
    for g, (country, lang) in enumerate(groups):
        rng = random.Random(f"{seed}/{split}/{country.code}/{lang}")
        for i in range(n // len(groups) + (g < n % len(groups))):
            template = rng.choice(country.templates[lang])
            text, spans = fill(template, country, lang, rng)
            # Minuscules : mêmes positions tant que chaque caractère garde sa longueur (vrai pour le français).
            if rng.random() < LOWERCASE_RATE and len(text.lower()) == len(text):
                text = text.lower()
            rows.append({"id": f"{country.code}-{lang}-{split}-{i}", "country": country.code, "lang": lang,
                         "template": template, "text": text, "entities": spans})  # fmt: skip
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=int, default=20000)
    parser.add_argument("--dev", type=int, default=1000)
    parser.add_argument("--test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        rows = generate(split, getattr(args, split), args.seed)
        path = OUT / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"{path} : {len(rows)} phrases, {sum(len(r['entities']) for r in rows)} entités")


if __name__ == "__main__":
    main()
