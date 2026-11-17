"""Construit les listes de la France (countries/fr/lists/*.txt) à partir de sources publiques.

    uv run python countries/fr/build_lists.py            # réutilise les téléchargements de data/raw/fr/
    uv run python countries/fr/build_lists.py --refresh  # retélécharge les sources

Sources (détail et licences dans country.yaml) :
- Insee, base des prénoms 2024 (Licence Ouverte 2.0, version Parquet publiée sur data.gouv.fr) : prénoms ;
- Insee, fichier des personnes décédées 2024 (Licence Ouverte 2.0) : noms de famille, comptés seulement ;
- geo.api.gouv.fr (Licence Ouverte 2.0, Code officiel géographique de l'Insee) : communes, départements, régions ;
- Wikidata (CC0) : entreprises, hôpitaux, universités et établissements d'enseignement situés en France ;
- countries/fr/manual/ : listes rédigées à la main (institutions).

Règle sur les noms de personnes : on ne garde que des mots isolés. Un prénom doit avoir été donné au moins
MIN_GIVEN fois depuis 1940 ; un nom de famille doit être porté par au moins MIN_FAMILY personnes du fichier des décès.
Le fichier des décès n'est lu que pour compter les noms : aucun nom complet, prénom, date ni lieu n'en est gardé.
Les organisations dont le nom commence par un prénom (« Christian Dior ») sont écartées.

Les listes produites sont versionnées : ce sont elles, et non les sources (qui changent), qui garantissent la
reproductibilité de la génération.
"""

import argparse
import json
import re
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path

HERE = Path(__file__).parent
RAW = Path("data/raw/fr")
LISTS = HERE / "lists"
MANUAL = HERE / "manual"
USER_AGENT = "masquener/0.1 (generation de donnees NER synthetiques)"

PRENOMS_URL = "https://static.data.gouv.fr/resources/base-prenoms-2024-insee/20250723-094355/prenoms-2024.parquet"
DECES_URL = "https://static.data.gouv.fr/resources/fichier-des-personnes-decedees/20250210-094840/deces-2024.txt"
GEO = "https://geo.api.gouv.fr"
WIKIDATA = "https://query.wikidata.org/sparql"

MIN_GIVEN = 2000  # naissances depuis 1940
MIN_FAMILY = 25  # personnes du fichier des décès 2024
MIN_POPULATION = 2000  # communes retenues comme « villes »

# Classes Wikidata → liste. Les entreprises sont trop nombreuses pour une requête sans limite : deux requêtes bornées.
WIKIDATA_ORGS = {
    "entreprises": [("Q6881511", ""), ("Q4830453", "LIMIT 4000")],
    "sante": [("Q16917", "")],
    "etablissements": [("Q3918", ""), ("Q9842", ""), ("Q38723", "")],
}


# --- Téléchargement -------------------------------------------------------------------------------------------------


def fetch(url: str, path: Path, refresh: bool) -> Path:
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=600) as response:
            path.write_bytes(response.read())
    return path


def fetch_json(url: str, name: str, refresh: bool):
    return json.loads(fetch(url, RAW / name, refresh).read_text(encoding="utf-8"))


def wikidata_labels(cls: str, limit: str, refresh: bool) -> list[str]:
    """Libellés français des éléments de la classe `cls` situés en France (P17 = Q142)."""
    sparql = f"""SELECT DISTINCT ?l WHERE {{
      ?e wdt:P31 wd:{cls}; wdt:P17 wd:Q142; rdfs:label ?l. FILTER(LANG(?l) = "fr")
    }} {limit}"""
    url = f"{WIKIDATA}?format=json&query={urllib.parse.quote(sparql)}"
    rows = fetch_json(url, f"wikidata_{cls}.json", refresh)["results"]["bindings"]
    return [r["l"]["value"] for r in rows]


# --- Nettoyage ------------------------------------------------------------------------------------------------------

PARTICLES = {"de", "du", "des", "d", "la", "le", "les", "van", "von", "da", "di", "del", "dos", "ben", "el", "al"}


def key(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


def name_case(word: str, family: bool = False) -> str:
    """« MARIE-CLAIRE » → « Marie-Claire » ; « LE GALL » → « Le Gall » ; « D'ARTOIS » → « d'Artois »."""
    parts = re.split(r"([ '\-])", word.strip().lower())
    out = []
    for i, p in enumerate(parts):
        if p in " '-" or not p:
            out.append(p)
        elif family and i > 0 and p in PARTICLES:
            out.append(p)
        elif family and i == 0 and p == "d" and len(parts) > 1 and parts[1] == "'":
            out.append(p)
        else:
            out.append(p[:1].upper() + p[1:])
    return "".join(out)


def unique(values, keep=lambda v: True) -> list[str]:
    """Dédoublonne sans accents ni casse, en gardant la forme accentuée."""
    best: dict[str, str] = {}
    for v in (re.sub(r"\s+", " ", x).strip() for x in values):
        if len(v) < 2 or len(v) > 70 or not re.search(r"[^\W\d_]", v) or not keep(v):
            continue
        k = key(v)
        if k not in best or (v != key(v) and best[k] == key(best[k])):
            best[k] = v
    return sorted(best.values(), key=key)


def read_manual(name: str) -> list[str]:
    path = MANUAL / f"{name}.txt"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def write_list(name: str, values: list[str], sources: list[str]) -> None:
    header = [f"# {name}.txt — généré par countries/fr/build_lists.py le {date.today().isoformat()}"]
    header += [f"# source : {s}" for s in sources]
    LISTS.mkdir(parents=True, exist_ok=True)
    (LISTS / f"{name}.txt").write_text("\n".join(header + values) + "\n", encoding="utf-8")
    print(f"  {name:15} {len(values):6}")


# --- Construction ---------------------------------------------------------------------------------------------------


def given_names(refresh: bool) -> list[str]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(fetch(PRENOMS_URL, RAW / "prenoms-2024.parquet", refresh))
    table = table.filter(pc.and_(pc.equal(table["niveau_geographique"], "FRANCE"), pc.greater_equal(table["periode"], "1940")))
    counts = Counter()
    for name, n in zip(table["prenom"].to_pylist(), table["valeur"].to_pylist(), strict=True):
        counts[name] += n
    return unique(name_case(n) for n, c in counts.items() if c >= MIN_GIVEN and not n.startswith("_"))


def family_names(refresh: bool) -> list[str]:
    """Noms de famille du fichier des décès (colonnes 1-80 : « NOM*PRÉNOMS/ »), comptés ; rien d'autre n'est lu."""
    counts = Counter()
    with fetch(DECES_URL, RAW / "deces-2024.txt", refresh).open(encoding="utf-8", errors="replace") as f:
        for line in f:
            family = line[:80].split("*", 1)[0].strip()
            if family and re.fullmatch(r"[A-ZÀ-ÖØ-Ý' \-]+", family):
                counts[family] += 1
    # Un mot, ou des particules puis un mot (« Le Gall ») : « PRUD HOMME » (apostrophe perdue) est écarté.
    def single(n: str) -> bool:
        words = n.split()
        return len(words) == 1 or all(w.lower() in PARTICLES for w in words[:-1])

    return unique(name_case(n, family=True) for n, c in counts.items() if c >= MIN_FAMILY and len(n) > 1 and single(n))


def build(refresh: bool) -> None:
    prenoms = given_names(refresh)
    noms = family_names(refresh)
    first_names = {key(p) for p in prenoms}

    communes = fetch_json(f"{GEO}/communes?fields=nom,population&format=json", "communes.json", refresh)
    villes = unique(c["nom"] for c in communes if (c.get("population") or 0) >= MIN_POPULATION)
    departements = unique(d["nom"] for d in fetch_json(f"{GEO}/departements", "departements.json", refresh))
    regions = unique(r["nom"] for r in fetch_json(f"{GEO}/regions", "regions.json", refresh))

    def org(v: str) -> bool:
        # Pas d'identifiant Wikidata (« Q123 ») ni de nom qui commence par un prénom (« Christian Dior »).
        # Bâtiments historiques (« ancien hôpital Saint-Marcoul ») : pas des établissements en activité.
        first = key(v.split()[0]).strip(".,")
        return not re.fullmatch(r"Q\d+", v) and first not in first_names and first not in {"ancien", "ancienne"}

    orgs = {}
    for name, classes in WIKIDATA_ORGS.items():
        orgs[name] = unique((v for cls, limit in classes for v in wikidata_labels(cls, limit, refresh)), keep=org)

    print("Listes :")
    write_list("prenoms", prenoms, [f"Insee, base des prénoms 2024 (Licence Ouverte 2.0) : donnés ≥ {MIN_GIVEN} fois depuis 1940"])
    write_list("noms", noms, [f"Insee, fichier des personnes décédées 2024 (Licence Ouverte 2.0) : noms portés par ≥ {MIN_FAMILY} personnes"])
    write_list("villes", villes, [f"geo.api.gouv.fr (Licence Ouverte 2.0) : communes de {MIN_POPULATION} habitants et plus"])
    write_list("departements", departements, ["geo.api.gouv.fr (Licence Ouverte 2.0)"])
    write_list("regions", regions, ["geo.api.gouv.fr (Licence Ouverte 2.0)"])
    write_list("entreprises", orgs["entreprises"], ["Wikidata (CC0) : entreprises situées en France"])
    write_list("sante", orgs["sante"], ["Wikidata (CC0) : hôpitaux situés en France"])
    write_list("etablissements", orgs["etablissements"], ["Wikidata (CC0) : universités, écoles et lycées situés en France"])
    write_list("institutions", unique(read_manual("institutions")), ["manual/institutions.txt"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true")
    build(parser.parse_args().refresh)


if __name__ == "__main__":
    main()
