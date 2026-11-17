"""Phrases synthétiques étiquetées, à partir des dossiers pays (countries/<code>/).

    python -m masquener.generate [--train 20000] [--dev 1000] [--test 2000] [--seed 0]

Chaque modèle de phrase voit ses cases remplies au hasard ; on connaît donc la position exacte de chaque entité.
Un modèle peut tenir sur plusieurs lignes (« \n » : fiche, courriel, en-tête), et une petite part des phrases est
écrite tout en capitales (UPPERCASE_RATE : en-têtes, formulaires). Jamais en minuscules : le texte tapé sans
majuscules relève du moteur qui emploie le modèle (il rend les majuscules avant de l'interroger) ; l'apprendre au
modèle lui fait prendre des noms communs pour des noms (v3).
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
# Case : type en capitales, sans accolade à l'intérieur ; les autres accolades sont du texte (JSON : {"nom": "{PER}"}).
SLOT = re.compile(r"\{([A-Z]+(?:[.:][^{}]+)?)\}")
PERSON_PART = re.compile(r"\{(nom|NOM|prenom|prenoms|initiale)\}")
# Case composée d'un modèle de phrase, {TYPE:forme} : les morceaux de nom s'y écrivent <nom>, <prenom>…
# (« {ORG:Pharmacie <nom>} », « rue {LOC:<prenom> <nom>} », « {PER:<prenom> Ould <nom>} »).
COMPOSED_PART = re.compile(r"<(nom|NOM|prenom|prenoms|initiale)>")
# Nombre de prénoms pour {prenoms} et poids : un prénom le plus souvent, parfois deux ou trois.
GIVEN_COUNT = {1: 5, 2: 4, 3: 1}
# Part des phrases écrites tout en capitales (« ATTESTATION DÉLIVRÉE À KONÉ ADAMA »), entités comprises.
UPPERCASE_RATE = 0.04
# Part des phrases « remises en majuscules » : un texte tapé sans majuscules (« appelle sophie chez ford ») est rendu
# au modèle avec une majuscule à chaque mot sauf les mots outils (« Appelle Sophie chez Ford ») ; le modèle doit y
# trouver les noms par le contexte, la majuscule n'y distinguant plus rien.
RECASED_RATE = 0.04
# Mots outils (FR, EN) laissés en minuscules dans une phrase remise en majuscules.
FUNCTION_WORDS = set(
    (
        "le la les l un une des du de d et ou mais donc or ni car à a au aux en dans par pour sur sous avec sans chez vers "
        "je j tu il elle on nous vous ils elles me m te t se s lui leur y ce c cet cette ces ça mon ma mes ton ta tes son sa ses "
        "notre nos votre vos qui que qu quoi dont où ne n pas plus est es suis sont être été ai as avons avez ont avoir fait faire "
        "peux peut veux veut dois doit va vais vas aller jusqu jusque puis très bien merci salut bonjour bonsoir oui non "
        "the a an and or but so of to in on at by for from with without into about as is am are was were be been i im "
        "you he she it we they me him her us them my your his its our their this that these those who what which where "
        "not no yes can could will would should do does did have has had please thanks thank hi hello hey until"
    ).split()
)
# Un mot : une lettre, puis lettres, apostrophes et traits d'union (« aujourd'hui », « jean-marc »).
WORD = re.compile(r"[^\W\d_](?:[^\W\d_]|['’-])*")


def recase(text: str) -> str:
    """Texte en minuscules, puis majuscule initiale à chaque mot qui n'est pas un mot outil (longueur inchangée)."""

    def word(m: re.Match) -> str:
        w = m.group(0)
        head = re.split(r"['’]", w)[0]
        if len(w) < 2 or head in FUNCTION_WORDS or w in FUNCTION_WORDS:
            return w
        return w[0].upper() + w[1:] if len(w[0].upper()) == 1 else w

    return WORD.sub(word, text.lower())
# Types des listes de noms propres : une ligne sans majuscule (« audit france ») en est retirée, une ligne qui commence
# par une minuscule (« centre hospitalier de Mâcon », libellé Wikidata) prend une majuscule initiale.
PROPER_TYPES = ("ORG", "LOC")
# Dans un modèle de phrase, « \n » marque un retour à la ligne : fiches, courriels, en-têtes sur plusieurs lignes.
NEWLINE = "\\n"

MONTHS = {
    "fr": "janvier février mars avril mai juin juillet août septembre octobre novembre décembre".split(),
    "en": "January February March April May June July August September October November December".split(),
    "de": "Januar Februar März April Mai Juni Juli August September Oktober November Dezember".split(),
    "it": "gennaio febbraio marzo aprile maggio giugno luglio agosto settembre ottobre novembre dicembre".split(),
    "es": "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split(),
    "ar": "يناير فبراير مارس أبريل مايو يونيو يوليو أغسطس سبتمبر أكتوبر نوفمبر ديسمبر".split(),
}
WEEKDAYS = {
    "fr": "lundi mardi mercredi jeudi vendredi samedi dimanche".split(),
    "en": "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split(),
    "de": "Montag Dienstag Mittwoch Donnerstag Freitag Samstag Sonntag".split(),
    "it": "lunedì martedì mercoledì giovedì venerdì sabato domenica".split(),
    "es": "lunes martes miércoles jueves viernes sábado domingo".split(),
    "ar": "الاثنين الثلاثاء الأربعاء الخميس الجمعة السبت الأحد".split(),
}
# Élision devant une valeur qui commence par une voyelle : « de Ageroute » → « d'Ageroute ».
ELISION = {"fr": re.compile(r"(?<![\w'])([dDlLjJ]e|[lL]a|[qQ]ue) $")}
VOWEL = re.compile(r"^[AEIOUYÀÂÉÈÊËÎÏÔÛÙaeiouyàâéèêëîïôûù]")


def bucket(*parts: str) -> str:
    """Jeu (train / dev / test) d'un élément, stable d'une exécution à l'autre : 80 / 10 / 10."""
    n = int(hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:8], 16) % 10
    return "test" if n == 0 else "dev" if n == 1 else "train"


def proper(value: str) -> str | None:
    """Ligne d'une liste de noms propres, avec majuscule initiale ; None si elle n'a aucune majuscule."""
    if not any(ch.isupper() for ch in value):
        return None
    return value[0].upper() + value[1:]


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
    weight: float = 1.0  # part des phrases, relative aux autres dossiers (country.yaml : weight)


def load_country(folder: Path, split: str) -> Country:
    """Dossier pays restreint aux modèles et lignes de listes d'un jeu."""
    config = yaml.safe_load((folder / "country.yaml").read_text(encoding="utf-8"))
    code = config["code"]
    lists = {}
    for name in config["lists"]:
        values = entries(folder / "lists" / f"{name}.txt")
        if config["lists"][name] in PROPER_TYPES:
            values = sorted({v for v in map(proper, values) if v})
        parts = {s: [v for v in values if bucket(code, name, v) == s] for s in SPLITS}
        lists[name] = parts[split] if all(parts.values()) else values
    templates = {}
    for lang in config["languages"]:
        if lang not in MONTHS:
            raise ValueError(f"{code} : langue {lang} sans noms de mois dans masquener/generate.py")
        templates[lang] = [t for t in entries(folder / "templates" / f"{lang}.txt") if bucket(code, lang, t) == split]
    return Country(code, lists, config["lists"], config["person_formats"], templates, float(config.get("weight", 1)))


def name_part(kind: str, country: Country, rng: random.Random) -> str:
    """Un morceau de nom de personne : nom, NOM (capitales), prenom, prenoms (1 à 3) ou initiale."""
    if kind in ("nom", "NOM"):
        name = rng.choice(country.lists["noms"])
        return name.upper() if kind == "NOM" else name
    if kind == "initiale":  # « K. », « J.-P. » pour un prénom composé
        return "-".join(f"{w[0]}." for w in rng.choice(country.lists["prenoms"]).split("-"))
    k = 1 if kind == "prenom" else rng.choices(list(GIVEN_COUNT), weights=list(GIVEN_COUNT.values()))[0]
    return " ".join(rng.sample(country.lists["prenoms"], k))


def person(country: Country, rng: random.Random) -> str:
    fmt = rng.choices(list(country.person_formats), weights=list(country.person_formats.values()))[0]
    return PERSON_PART.sub(lambda m: name_part(m.group(1), country, rng), fmt)


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
        elif ":" in slot:  # entité composée autour d'un nom de personne : {ORG:Cabinet <nom> & Associés}
            kind, form = slot.split(":", 1)
            value = COMPOSED_PART.sub(lambda m: name_part(m.group(1), country, rng), form)
        else:
            kind, name = slot.split(".")
            value = rng.choice(country.lists[name])
        if VOWEL.match(value) and lang in ELISION and (e := ELISION[lang].search(text)):
            text = text[: e.start()] + e.group(1)[:-1] + "'"
        if kind:
            spans.append([len(text), len(text) + len(value), kind])
        text += value
    return text + template[at:], spans


def generate(split: str, n: int, seed: int) -> list[dict]:
    """n phrases d'un jeu, réparties entre les dossiers selon leur poids (weight), puis à parts égales entre leurs
    langues. Une langue sans modèle dans ce jeu (liste trop courte) n'en reçoit aucune."""
    folders = sorted(p for p in COUNTRIES.iterdir() if (p / "country.yaml").exists())
    countries = [load_country(folder, split) for folder in folders]
    groups = [(c, lang) for c in countries for lang in sorted(c.templates) if c.templates[lang]]
    weights = [c.weight / sum(1 for k, lang in groups if k is c) for c, _ in groups]
    counts = [int(n * w / sum(weights)) for w in weights]
    for g in range(n - sum(counts)):  # reste de l'arrondi : un de plus aux premiers groupes
        counts[g % len(groups)] += 1
    rows = []
    for (country, lang), count in zip(groups, counts, strict=True):
        rng = random.Random(f"{seed}/{split}/{country.code}/{lang}")
        for i in range(count):
            template = rng.choice(country.templates[lang])
            text, spans = fill(template, country, lang, rng)
            # Capitales : mêmes positions tant que chaque caractère garde sa longueur (« ß » → « SS » l'empêche).
            draw = rng.random()
            if draw < UPPERCASE_RATE and len(text.upper()) == len(text):
                text = text.upper()
            elif draw < UPPERCASE_RATE + RECASED_RATE and len(text.lower()) == len(text):
                text = recase(text)
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
