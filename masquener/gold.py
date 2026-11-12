"""Jeux de test publics annotés, à révision figée (fichiers parquet du Hub).

Rôle dans la décision :
- conll, masakhaner : annotés à la main → une régression est bloquante ;
- wikiann : étiquettes produites automatiquement, bruitées → simple indicateur.
"""

import json
from dataclasses import dataclass

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

WIKIANN = ("unimelb-nlp/wikiann", "f0a3be6dc5564c0cc4150bb660144800a1f539d4")
# Branches refs/convert/parquet (les dépôts d'origine sont des scripts).
CONLL2003 = ("eriktks/conll2003", "ce85b39f9dd99f552d0739d456814e95fb6a39b0")
CONLL2002 = ("eriktks/conll2002", "cc09aa3e64654924d2bec21b294bd79c885bafd6")
MASAKHANER = ("masakhane/masakhaner2", "a11b3e2dc7b4425c6d9a77b67d269823154eea00")


@dataclass(frozen=True)
class GoldSet:
    files: dict[str, tuple[str, str, str]]  # langue → (dépôt, révision, fichier parquet du test)
    blocking: bool


GOLD_SETS = {
    # Langues du modèle de départ.
    "wikiann": GoldSet(
        {lang: (*WIKIANN, f"{lang}/test-00000-of-00001.parquet") for lang in "ar de en es fr it lv nl pt zh".split()},
        blocking=False,
    ),
    # Données d'entraînement du modèle de départ ; l'allemand de CoNLL 2003 n'est pas diffusé (licence).
    "conll": GoldSet(
        {
            "en": (*CONLL2003, "conll2003/test/0000.parquet"),
            "es": (*CONLL2002, "es/test/0000.parquet"),
            "nl": (*CONLL2002, "nl/test/0000.parquet"),
        },
        blocking=True,
    ),
    # MasakhaNER 2.0 : 20 langues africaines annotées à la main (AFL-3.0).
    "masakhaner": GoldSet(
        {
            lang: (*MASAKHANER, f"{lang}/test/0000.parquet")
            for lang in "bam bbj ewe fon hau ibo kin lug luo mos nya pcm sna swa tsn twi wol xho yor zul".split()
        },
        blocking=True,
    ),
}


def load_test(gold: GoldSet, lang: str) -> tuple[list[list[str]], list[list[str]]]:
    return load_split(gold, lang, "test")


def load_split(gold: GoldSet, lang: str, split: str) -> tuple[list[list[str]], list[list[str]]]:
    """Phrases (mots) et étiquettes BIO d'un jeu (test, train, validation), à la même révision que le test.

    Les noms d'étiquettes viennent des métadonnées du fichier.
    """
    repo, revision, filename = gold.files[lang]
    filename = filename.replace("test", split)
    path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    features = json.loads(pq.read_schema(path).metadata[b"huggingface"])["info"]["features"]
    names = features["ner_tags"]["feature"]["names"]
    table = pq.read_table(path, columns=["tokens", "ner_tags"]).to_pydict()
    return table["tokens"], [[names[t] for t in tags] for tags in table["ner_tags"]]
