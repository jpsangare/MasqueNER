"""Construit les listes de la Côte d'Ivoire (countries/ci/lists/*.txt) à partir de sources publiques.

    uv run python countries/ci/build_lists.py            # réutilise les téléchargements de data/raw/ci/
    uv run python countries/ci/build_lists.py --refresh  # retélécharge les sources

Sources (détail et licences dans country.yaml) :
- data.gouv.ci (producteur data354-CI) : circonscriptions, RGPH, sites radioélectriques, santé, banques, hôtels,
  écoles, entreprises agréées, chaînes de télévision, parcs ;
- Wikidata (CC0) : prénoms et noms de famille des personnalités ivoiriennes ;
- countries/ci/manual/ : listes rédigées à la main (noms courants, institutions, entreprises connues).

Règle sur les noms de personnes : on ne garde que des mots isolés (un prénom, un nom de famille), portés par au moins
deux personnes différentes, sans aucun lien (nom complet, lieu, fonction). Les noms d'organisations qui contiennent un
nom de personne (« Pharmacie Kouassi », « Clinique Dr Koné ») sont écartés.

Les listes produites sont versionnées : ce sont elles, et non les sources (qui changent), qui garantissent la
reproductibilité de la génération. Le script documente leur provenance et permet de les mettre à jour.
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
RAW = Path("data/raw/ci")
LISTS = HERE / "lists"
MANUAL = HERE / "manual"
USER_AGENT = "masquener/0.1 (generation de donnees NER synthetiques)"

DATAGOUV = "https://data.gouv.ci/data-fair/api/v1/datasets/"
WIKIDATA = "https://query.wikidata.org/sparql"


# --- Téléchargement -------------------------------------------------------------------------------------------------


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def datagouv(slug: str, fields: list[str], refresh: bool) -> list[dict]:
    """Toutes les lignes d'un jeu data.gouv.ci (colonnes choisies), mises en cache dans data/raw/ci/."""
    cache = RAW / f"{slug.replace(chr(39), '_')}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    query = urllib.parse.urlencode({"size": 10000, "select": ",".join(fields)})
    url, rows = f"{DATAGOUV}{urllib.parse.quote(slug)}/lines?{query}", []
    while url:
        page = _get_json(url)
        rows += [{f: r.get(f) for f in fields} for r in page["results"]]
        url = page.get("next")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def wikidata_names(prop: str, refresh: bool) -> Counter:
    """Mot (prénom P735 ou nom de famille P734) → nombre de personnalités ivoiriennes (P27 = Q1008) qui le portent."""
    cache = RAW / f"wikidata_{prop}.json"
    if cache.exists() and not refresh:
        return Counter(json.loads(cache.read_text(encoding="utf-8")))
    sparql = f"""SELECT ?n (SAMPLE(?l) AS ?label) (COUNT(DISTINCT ?p) AS ?c) WHERE {{
      ?p wdt:P31 wd:Q5; wdt:P27 wd:Q1008; wdt:{prop} ?n.
      ?n rdfs:label ?l. FILTER(LANG(?l) IN ("mul", "fr", "en"))
    }} GROUP BY ?n"""
    rows = _get_json(f"{WIKIDATA}?format=json&query={urllib.parse.quote(sparql)}")["results"]["bindings"]
    counts = Counter()
    for r in rows:
        counts[r["label"]["value"]] += int(r["c"]["value"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(counts, ensure_ascii=False), encoding="utf-8")
    return counts


# --- Nettoyage ------------------------------------------------------------------------------------------------------

# Mots outils laissés en minuscules quand on remet en casse un texte tout en capitales.
_LOWER = {"de", "du", "des", "la", "le", "les", "et", "en", "au", "aux", "sur", "pour", "à", "d", "l"}
# Accents perdus dans les sources en capitales.
_ACCENTS = {
    "college": "Collège", "lycee": "Lycée", "ecole": "École", "prive": "Privé", "privee": "Privée",
    "eglise": "Église", "evangelique": "Évangélique", "methodiste": "Méthodiste", "protestant": "Protestant",
    "catholique": "Catholique", "superieur": "Supérieur", "superieure": "Supérieure", "general": "Général",
    "generale": "Générale", "republique": "République", "ivoirienne": "Ivoirienne", "societe": "Société",
    "cooperative": "Coopérative", "medical": "Médical", "medicale": "Médicale", "sante": "Santé",
    "hopital": "Hôpital", "universite": "Université", "etudes": "Études", "complexe": "Complexe",
}  # fmt: skip
# Numéro de téléphone collé au nom (« Hotel Ideal 08541498 », « Sanguine30 64 08 69 ») : 8 ou 10 chiffres en fin de nom.
_PHONE = re.compile(r"\s*(?:\+?225\s*)?\d{2}(?:[ .]?\d{2}){3,4}\s*$")
_SIGLE = re.compile(r"^(?:[A-Z]{2,5}|[A-Z]+-[A-Z]{1,3}|S\.?A\.?|SARL|SA)$")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def key(s: str) -> str:
    """Clé de dédoublonnage : sans accents, sans casse, ponctuation réduite."""
    return re.sub(r"[^a-z0-9]+", " ", strip_accents(s).lower().replace("’", "'")).strip()


def _cap(word: str) -> str:
    """Majuscule initiale, et après trait d'union ou apostrophe (Grand-Bassam, N'Guessan) ; accents usuels rétablis."""
    lw = word.lower()
    if strip_accents(lw) in _ACCENTS:
        return _ACCENTS[strip_accents(lw)]
    return re.sub(r"(^|[-'])(\w)", lambda m: m.group(1) + m.group(2).upper(), lw)


def tidy(s: str | None, place: bool = False) -> str:
    """Espaces et apostrophes normalisés ; casse uniforme.

    - place (lieux) : chaque mot en majuscule initiale, mots outils en minuscules, parenthèses retirées ;
    - sinon (organisations) : les mots en capitales sont remis en casse, sauf les sigles courts (CIE, SARL).
    """
    if not s:
        return ""
    s = re.sub(r"\s+", " ", str(s).replace("’", "'").replace("_", " "))
    s = _PHONE.sub("", s)
    if place:
        s = re.sub(r"\s*\(.*?\)", "", s)
    s = s.strip(" .,;:-\"'")
    words = []
    for i, w in enumerate(s.split(" ")):
        letters = [c for c in w if c.isalpha()]
        caps = bool(letters) and all(c.isupper() for c in letters)
        if i and w.lower() in _LOWER and (caps or place):
            words.append(w.lower())
        elif place or (caps and (len(letters) > 5 or strip_accents(w.lower()) in _ACCENTS) and not _SIGLE.match(w)):
            words.append(_cap(w))
        else:
            words.append(w)
    return " ".join(words)


# Mots seuls trop vagues pour être un nom de lieu ou d'organisation.
_GENERIC = {
    "cite", "residentiel", "residence", "mairie", "village", "quartier", "centre", "zone", "gare", "marche",
    "hotel", "restaurant", "pharmacie", "clinique", "maquis", "le succes", "la paix", "amour du prochain",
}  # fmt: skip


def unique(values, keep=lambda v: True, place: bool = False, max_words: int = 8) -> list[str]:
    """Dédoublonne (sans accents ni casse) en préférant la forme accentuée et en casse mixte."""
    best: dict[str, str] = {}
    for v in (tidy(x, place) for x in values):
        if len(v) < 3 or len(v.split()) > max_words or key(v) in _GENERIC:
            continue
        if not v or not re.search(r"[^\W\d_]", v) or len(v) > 70 or not keep(v):
            continue
        k = key(v)
        score = (v != v.upper(), v != strip_accents(v))
        if k not in best or score > (best[k] != best[k].upper(), best[k] != strip_accents(best[k])):
            best[k] = v
    return sorted(best.values(), key=key)


def read_manual(name: str) -> list[str]:
    path = MANUAL / f"{name}.txt"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def write_list(name: str, values: list[str], sources: list[str]) -> None:
    header = [f"# {name}.txt — généré par countries/ci/build_lists.py le {date.today().isoformat()}"]
    header += [f"# source : {s}" for s in sources]
    LISTS.mkdir(parents=True, exist_ok=True)
    (LISTS / f"{name}.txt").write_text("\n".join(header + values) + "\n", encoding="utf-8")
    print(f"  {name:13} {len(values):6}")


# --- Construction ---------------------------------------------------------------------------------------------------

_PERSON_MARKERS = re.compile(r"\b(?:dr|docteur|pr|professeur|me|maître|maitre|mme|mr|m|chez|soeur|sœur|feu|el hadj)\b\.?", re.I)


def build(refresh: bool) -> None:
    # Noms de personnes : mots isolés portés par au moins deux personnalités (Wikidata), plus la liste manuelle.
    given, family = wikidata_names("P735", refresh), wikidata_names("P734", refresh)
    one_word = lambda v: " " not in v.strip() and not any(c.isdigit() for c in v)  # noqa: E731
    prenoms = unique([n for n, c in given.items() if c >= 2] + read_manual("prenoms"), one_word)
    noms = unique([n for n, c in family.items() if c >= 2] + read_manual("noms"), one_word)
    # Pour écarter les organisations nommées d'après une personne : tous les mots de noms connus, même rares.
    person_words = {key(w) for w in [*given, *family, *prenoms, *noms] if len(w) > 2}

    # Lieux.
    circ = datagouv(
        "liste-des-circonscriptions-administratives-et-des-communes",
        ["COMMUNE", "SOUS-PREFECTURE", "CHEF-LIEU", "REGION", "DEPARTEMENT"],
        refresh,
    )
    rgph = datagouv(
        "recensement-de-la-population-ivoirienne", ["DISTRICT", "REGION", "DEPARTEMENT", "SOUS-PREFECTURE_OU_COMMUNE"], refresh
    )
    villes = unique(
        [r[c] for r in circ for c in ("COMMUNE", "SOUS-PREFECTURE", "CHEF-LIEU", "DEPARTEMENT")]
        + [r[c] for r in rgph for c in ("DEPARTEMENT", "SOUS-PREFECTURE_OU_COMMUNE")],
        place=True,
    )
    regions = unique(
        [r["REGION"] for r in circ] + [r[c] for r in rgph for c in ("DISTRICT", "REGION")],
        lambda v: key(v) not in {key(x) for x in villes} or len(v.split()) > 1,
        place=True,
    )
    town_keys = {key(v) for v in villes}

    radio = datagouv(
        "liste-des-sites-radioelectriques-orange-mtn-moov", ["Localité_selon_l'INS", "Département"], refresh
    )
    health = datagouv(
        "hospitaux-de-cote-d'ivoire", ["Ville_et_commune", "Quartier", "Catégorie", "Nom_de_l'établissement"], refresh
    )
    houses = datagouv("prix-des-maisons-a-abdjan", ["location"], refresh)
    # Localités INS d'Abidjan = quartiers ; ailleurs = villages et quartiers des villes de l'intérieur.
    in_abidjan = lambda r: key(r["Département"] or "") == "abidjan"  # noqa: E731
    # Le champ « Quartier » d'OpenStreetMap est libre : on écarte les repères (« face pharmacie… ») et les noms de personnes.
    landmark = re.compile(r"\b(?:face|pr[eè]s|derri[eè]re|pharmacie|h[oô]pital|lyc[ée]e|[ée]coles?|commerce|carrefour)\b", re.I)
    osm_quarters = [q for r in health if (q := tidy(r["Quartier"], place=True)) and not landmark.search(q)]
    quartiers = unique(
        [r["Localité_selon_l'INS"] for r in radio if in_abidjan(r)]
        + [q for q in osm_quarters if not any(key(w) in person_words for w in q.split())]
        + [r["location"] for r in houses if r["location"]],
        lambda v: key(v) not in town_keys and not re.search(r"\b(?:inconnu|n/?a|none)\b", v, re.I),
        place=True,
        max_words=4,
    )
    villages = unique(
        [r["Localité_selon_l'INS"] for r in radio if not in_abidjan(r)],
        lambda v: key(v) not in town_keys and key(v) not in {key(q) for q in quartiers},
        place=True,
        max_words=4,
    )
    parks = datagouv("parcs-nationaux-et-reserves-naturelles-et-protegees-de-cote-d'ivoire", ["name"], refresh)
    sites = unique([r["name"] for r in parks])

    # Organisations : sans nom de personne.
    def org(v: str) -> bool:
        words = [key(w) for w in re.split(r"[\s'-]+", v) if w]
        return not _PERSON_MARKERS.search(v) and not any(w in person_words and w not in town_keys for w in words)

    sante = unique([r["Nom_de_l'établissement"] for r in health], org)
    cmu = datagouv("liste-des-etablissements-proposant-le-service-cmu", ["nom_pharmacie"], refresh)
    not_admin = lambda v: not re.match(r"(?:sous[ -])?pr[ée]fecture|mairie|universit", v, re.I)  # noqa: E731
    sante = unique(sante + [r["nom_pharmacie"] for r in cmu], lambda v: org(v) and not_admin(v))
    schools = datagouv("liste-detablissements-scolaires-dabidjan-etab-ci", ["nom_detablissement"], refresh)
    schools_abj = datagouv("liste-detablissements-scolaires-dabidjan-en-francais", ["nom_de_letablissement"], refresh)
    ecoles = unique([r["nom_detablissement"] for r in schools] + [r["nom_de_letablissement"] for r in schools_abj], org)
    banks = datagouv("bank", ["Etablissement"], refresh)
    hotels = datagouv("hotels-et-restauration-en-cote-divoire", ["Catégorie", "Nom_de_l'établissement"], refresh)
    agreed = datagouv(
        "liste-des-consign-maritimes-manut-aviateurs-portuaires-agrees-2007", ["RAISON_SOCIALE"], refresh
    )
    oil = datagouv("liste-des-blocs-en-exploration-petroliere-et-leurs-operateurs-en-2017", ["Opérateurs"], refresh)
    tv = datagouv(
        "repertoire-des-chaines-de-television-et-bouquets-autorises-de-cote-d'ivoire",
        ["NOM_DU_SERVICE", "TITULAIRE_D’AUTORISATION"],
        refresh,
    )
    entreprises = unique(
        [r["Etablissement"] for r in banks]
        + [r["Nom_de_l'établissement"] for r in hotels if r["Catégorie"] in ("Hôtel", "Restaurant")]
        + [r["RAISON_SOCIALE"] for r in agreed]
        + [o for r in oil for o in re.split(r"\s*[/,]\s*", r["Opérateurs"] or "")]
        + [r[c] for r in tv for c in ("NOM_DU_SERVICE", "TITULAIRE_D’AUTORISATION")]
        + read_manual("entreprises"),
        org,
    )
    institutions = unique(read_manual("institutions"))

    dg = "data.gouv.ci (data354-CI)"
    osm = f"{dg}, données issues d'OpenStreetMap (ODbL, © contributeurs OpenStreetMap)"
    print("Listes :")
    write_list("prenoms", prenoms, ["Wikidata (CC0), prénoms (P735) portés par ≥ 2 personnalités ivoiriennes", "manual/prenoms.txt"])
    write_list("noms", noms, ["Wikidata (CC0), noms de famille (P734) portés par ≥ 2 personnalités ivoiriennes", "manual/noms.txt"])
    write_list("villes", villes, [f"{dg}, circonscriptions administratives et communes ; RGPH 2021 (Licence Ouverte 2.0)"])
    write_list("regions", regions, [f"{dg}, circonscriptions administratives ; RGPH 2021 (Licence Ouverte 2.0)"])
    write_list("quartiers", quartiers, [f"{dg}, sites radioélectriques (localités INS, Abidjan) ; prix des maisons à Abidjan (Licence Ouverte 2.0)", f"{osm} : quartiers des établissements de santé"])
    write_list("villages", villages, [f"{dg}, sites radioélectriques (localités INS hors Abidjan) (Licence Ouverte 2.0)"])
    write_list("sites", sites, [f"{osm} : parcs nationaux et réserves"])
    write_list("sante", sante, [f"{osm} : hôpitaux, cliniques, pharmacies", f"{dg}, établissements proposant la CMU (Licence Ouverte 2.0)"])
    write_list("ecoles", ecoles, [f"{dg}, établissements secondaires ; établissements scolaires d'Abidjan (Licence Ouverte 2.0)"])
    write_list("entreprises", entreprises, [f"{osm} : banques, hôtels, restaurants", f"{dg}, consignataires et manutentionnaires agréés, opérateurs pétroliers, chaînes de télévision (Licence Ouverte 2.0)", "manual/entreprises.txt"])
    write_list("institutions", institutions, ["manual/institutions.txt"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="retélécharger les sources")
    build(parser.parse_args().refresh)
