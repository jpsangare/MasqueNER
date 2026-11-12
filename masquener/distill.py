"""Phrases de révision étiquetées par le modèle de départ (distillation), contre l'oubli.

    python -m masquener.distill [--model artifacts/reference] [--train 20000] [--dev 1000] [--seed 0]

Textes : phrases d'entraînement (train) et de validation (dev) de CoNLL (en, es, nl) et de WikiANN (10 langues),
aux révisions figées dans masquener/gold.py. On ne garde que le texte : les étiquettes d'origine sont jetées et
remplacées par ce que prédit le modèle de départ, avec le regroupement d'OnnxNer.spans. L'élève
apprend ainsi à refaire ce que faisait le professeur. Les jeux de test, qui servent au contrôle, ne sont jamais lus.

Sortie : data/distill/{train,dev}.jsonl, même format que data/synthetic (texte et entités en caractères).
"""

import argparse
import json
import random
from pathlib import Path

from .evaluate import NER_TYPES
from .gold import GOLD_SETS, load_split
from .onnx_ner import OnnxNer

SOURCES = ("conll", "wikiann")
OUT = Path("data/distill")
FILE_SPLIT = {"train": "train", "dev": "validation"}
MAX_TOKENS = 128  # longueur d'entraînement ; les phrases plus longues sont écartées
NO_SPACE = {"zh"}  # WikiANN découpe le chinois en caractères : on les recolle sans espace.


def sentences(source: str, lang: str, split: str) -> list[str]:
    words, _ = load_split(GOLD_SETS[source], lang, FILE_SPLIT[split])
    sep = "" if lang in NO_SPACE else " "
    texts = (sep.join(w).strip() for w in words if w and w != ["-DOCSTART-"])
    return sorted({t for t in texts if t})


def build(ner: OnnxNer, split: str, n: int, seed: int) -> list[dict]:
    """n phrases, réparties à parts égales entre les couples (jeu, langue)."""
    groups = [(source, lang) for source in SOURCES for lang in GOLD_SETS[source].files]
    rows = []
    for g, (source, lang) in enumerate(groups):
        pool = sentences(source, lang, split)
        pool = [t for t, e in zip(pool, ner.tokenizer.encode_batch(pool)) if len(e.ids) <= MAX_TOKENS]
        picked = random.Random(f"{seed}/{split}/{source}/{lang}").sample(pool, min(len(pool), n // len(groups) + (g < n % len(groups))))
        for i, text in enumerate(picked):
            spans = sorted([s, e, t] for s, e, t, _ in ner.spans(text) if t in NER_TYPES)
            rows.append({"id": f"{source}-{lang}-{split}-{i}", "source": source, "lang": lang, "text": text, "entities": spans})
        print(f"  {source}/{lang} : {len(picked)} phrases")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("artifacts/reference"), help="professeur, format Xenova")
    parser.add_argument("--train", type=int, default=20000)
    parser.add_argument("--dev", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    ner = OnnxNer(args.model)
    OUT.mkdir(parents=True, exist_ok=True)
    for split in FILE_SPLIT:
        rows = build(ner, split, getattr(args, split), args.seed)
        path = OUT / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"{path} : {len(rows)} phrases, {sum(len(r['entities']) for r in rows)} entités")


if __name__ == "__main__":
    main()
