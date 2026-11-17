"""Évaluation d'un modèle au format Xenova.

    python -m masquener.evaluate --model artifacts/reference --name reference gold {conll,masakhaner,wikiann} [--limit N]
    python -m masquener.evaluate --model artifacts/reference --name reference corpus CHEMIN/corpus-ci.json
    python -m masquener.evaluate --model artifacts/reference --name reference synthetic [data/synthetic/test.jsonl]
    python -m masquener.evaluate --model artifacts/reference --name reference familles [familles/familles.txt]

- gold : jeu de test public annoté (masquener/gold.py) → reports/<nom>/<jeu>.json (versionné).
- corpus : corpus réel annoté (format JSON : cases, text, entities), lu sur place et jamais copié → reports/local/<nom>/corpus.json
  (ignoré par git). Le rapport ne contient que des chiffres.
- synthetic : test synthétique des pays (modèles et noms absents de l'entraînement) → reports/<nom>/synthetic.json.
- familles : phrases écrites à la main, une famille de cas à la fois (prénoms qui sont des mots, formules de
  politesse…) → reports/<nom>/familles.json.
"""

import argparse
import json
import re
from pathlib import Path

from .gold import GOLD_SETS, load_test
from .labels import ENTITY_TYPES
from .metrics import bio_spans, score
from .onnx_ner import OnnxNer, char_spans
from .reference import sha256

# Types du corpus réel → étiquettes du modèle (noms français ou déjà au format du modèle). Les autres types
# (numéros, dates…) relèvent des motifs : ils ne sont pas notés, mais un masquage qui tombe dessus n'est pas une erreur.
CORPUS_TYPES = {"PERSONNE": "PER", "LIEU": "LOC", "ORGANISATION": "ORG", "PER": "PER", "LOC": "LOC", "ORG": "ORG"}
NER_TYPES = ("PER", "ORG", "LOC")


def eval_gold(ner: OnnxNer, name: str, limit: int | None) -> dict:
    """Scores sur les types communs au jeu et au modèle : les autres (MISC du jeu, DATE absente du jeu) valent « O »."""
    gold_set = GOLD_SETS[name]
    results = {}
    for lang in gold_set.files:
        sentences, gold = load_test(gold_set, lang)
        sentences, gold = sentences[:limit], gold[:limit]
        types = tuple(t for t in ENTITY_TYPES if any(tag.endswith("-" + t) for tags in gold for tag in tags))
        keep = lambda tags: [tag if tag[2:] in types else "O" for tag in tags]  # noqa: E731
        pred = [keep(tags) for tags in ner.tag_words(sentences)]
        results[lang] = score([bio_spans(keep(g)) for g in gold], [bio_spans(p) for p in pred], types)
        print(f"  {lang} : F1 {results[lang]['micro']['f1']:.4f} ({len(sentences)} phrases, {'/'.join(types)})")
    results["macro_f1"] = round(sum(results[lang]["micro"]["f1"] for lang in gold_set.files) / len(gold_set.files), 4)
    return {"dataset": name, "files": gold_set.files, "blocking": gold_set.blocking, "limit": limit, "results": results}


def gold_spans(case: dict) -> set:
    """Chaque occurrence de chaque valeur annotée ; une valeur incluse dans une autre du même type n'est pas comptée."""
    spans = set()
    for e in case["entities"]:
        kind = CORPUS_TYPES.get(e["type"])
        if not kind:
            continue
        at = case["text"].find(e["text"])
        if at < 0:
            raise ValueError(f"cas {case['id']} : une valeur annotée ne figure pas dans le texte")
        while at >= 0:
            spans.add((kind, at, at + len(e["text"])))
            at = case["text"].find(e["text"], at + 1)
    return {s for s in spans if not any(o != s and o[0] == s[0] and o[1] <= s[1] and s[2] <= o[2] for o in spans)}


def annotated_spans(case: dict) -> set:
    """Positions de toutes les valeurs annotées du cas, de tout type (numéros, dates… compris)."""
    spans = set()
    for e in case["entities"]:
        at = case["text"].find(e["text"])
        while at >= 0:
            spans.add((at, at + len(e["text"])))
            at = case["text"].find(e["text"], at + 1)
    return spans


def text_scores(ner: OnnxNer, texts: list[str], gold: list[set], types: tuple[str, ...], annotated: list[set] | None = None) -> dict:
    """Scores stricts (type et limites exacts), taux de masquage (valeur attendue entièrement couverte, quel que soit le
    type) et précision de masquage (masquages qui touchent une valeur annotée, de tout type si `annotated` est donné).
    Un masquage qui ne touche aucune valeur annotée est « en trop » : un mot ordinaire caché à tort (« Bonjour »)."""
    pred = [{s for s in char_spans(ner.spans(text)) if s[0] in types} for text in texts]
    annotated = annotated or [{(s, e) for _, s, e in g} for g in gold]
    masked = expected = predicted = spurious = 0
    negatives = negatives_hit = 0
    for g, p, a in zip(gold, pred, annotated, strict=True):
        covered = set().union(*(range(s, e) for _, s, e in p)) if p else set()
        expected += len(g)
        masked += sum(all(i in covered for i in range(s, e)) for _, s, e in g)
        extra = sum(not any(s < ae and as_ < e for as_, ae in a) for _, s, e in p)
        predicted += len(p)
        spurious += extra
        if not a:
            negatives += 1
            negatives_hit += bool(p)
    return {
        "cases": len(texts),
        "strict": score(gold, pred, types),
        "masking_rate": round(masked / expected, 4) if expected else None,
        "masking_precision": round(1 - spurious / predicted, 4) if predicted else 1.0,
        "spurious": spurious,
        "negative_cases": negatives,
        "negative_cases_hit": negatives_hit,
    }


def eval_corpus(ner: OnnxNer, path: Path) -> dict:
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    texts = [c["text"] for c in cases]
    return {"corpus": path.name, **text_scores(ner, texts, [gold_spans(c) for c in cases], NER_TYPES, [annotated_spans(c) for c in cases])}


def eval_synthetic(ner: OnnxNer, path: Path) -> dict:
    """Phrases synthétiques (masquener/generate.py), notées pays par pays, sur les types du corpus (PER, ORG, LOC)."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    results = {}
    for country in sorted({r["country"] for r in rows}):
        mine = [r for r in rows if r["country"] == country]
        gold = [{(t, s, e) for s, e, t in r["entities"]} for r in mine]
        results[country] = text_scores(ner, [r["text"] for r in mine], gold, NER_TYPES)
        r = results[country]
        # Dossier d'exemples négatifs (sans entité) : ni F1 ni taux de masquage, seulement les masquages en trop.
        if r["masking_rate"] is None:
            print(f"  {country} : phrases intactes {r['negative_cases'] - r['negative_cases_hit']}/{r['negative_cases']} ({r['spurious']} en trop)")
        else:
            f1, rate, precision = r["strict"]["micro"]["f1"], r["masking_rate"], r["masking_precision"]
            print(f"  {country} : F1 {f1:.4f}, masquage {rate:.4f}, précision de masquage {precision:.4f} ({r['spurious']} en trop)")
    return {"dataset": "synthetic", "file": path.name, "file_sha256": sha256(path), "blocking": True, "results": results}


MARKED = re.compile(r"\[([^\]]+)\]\((PER|LOC|ORG)\)")


def parse_families(path: Path) -> dict[str, list[tuple[str, set]]]:
    """familles/familles.txt → {famille: [(texte, {(type, début, fin)})]} ; voir l'en-tête du fichier."""
    families, current = {}, None
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if m := re.fullmatch(r"== (\S+) ==", line):
            current = families.setdefault(m.group(1), [])
            continue
        if current is None:
            raise ValueError(f"{path}:{n} : phrase avant la première famille")
        text, spans, at = "", set(), 0
        line = line.replace("\\n", "\n")
        for m in MARKED.finditer(line):
            text += line[at : m.start()]
            spans.add((m.group(2), len(text), len(text) + len(m.group(1))))
            text += m.group(1)
            at = m.end()
        text += line[at:]
        if "[" in text and "](" in text:
            raise ValueError(f"{path}:{n} : entité mal marquée")
        current.append((text, spans))
    return families


def eval_families(ner: OnnxNer, path: Path) -> dict:
    """Une famille positive est notée sur le masquage et sa précision ; une famille négative, sur ses phrases intactes."""
    results = {}
    for family, rows in parse_families(path).items():
        r = results[family] = text_scores(ner, [t for t, _ in rows], [g for _, g in rows], NER_TYPES)
        if r["masking_rate"] is None:
            print(f"  {family} : phrases intactes {r['negative_cases'] - r['negative_cases_hit']}/{r['negative_cases']}")
        else:
            print(f"  {family} : masquage {r['masking_rate']:.4f}, précision de masquage {r['masking_precision']:.4f}")
    return {"dataset": "familles", "file": path.name, "file_sha256": sha256(path), "blocking": True, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="dossier au format Xenova")
    parser.add_argument("--name", required=True, help="nom du modèle dans les rapports")
    sub = parser.add_subparsers(dest="target", required=True)
    g = sub.add_parser("gold", help="jeu de test public annoté")
    g.add_argument("dataset", choices=sorted(GOLD_SETS))
    g.add_argument("--limit", type=int, help="phrases par langue (essai rapide ; les décisions se prennent sans)")
    c = sub.add_parser("corpus")
    c.add_argument("path", type=Path)
    s = sub.add_parser("synthetic", help="phrases synthétiques des pays (masquener/generate.py)")
    s.add_argument("path", type=Path, nargs="?", default=Path("data/synthetic/test.jsonl"))
    f = sub.add_parser("familles", help="test par famille de cas (familles/familles.txt)")
    f.add_argument("path", type=Path, nargs="?", default=Path("familles/familles.txt"))
    args = parser.parse_args()

    ner = OnnxNer(args.model)
    meta = {"model": args.name, "onnx_sha256": sha256(args.model / "onnx" / "model_quantized.onnx")}
    if args.target == "gold":
        report, out = eval_gold(ner, args.dataset, args.limit), Path("reports") / args.name / f"{args.dataset}.json"
        print(f"{args.dataset}, F1 moyen sur les langues : {report['results']['macro_f1']:.4f}")
    elif args.target == "synthetic":
        report, out = eval_synthetic(ner, args.path), Path("reports") / args.name / "synthetic.json"
    elif args.target == "familles":
        report, out = eval_families(ner, args.path), Path("reports") / args.name / "familles.json"
    else:
        report, out = eval_corpus(ner, args.path), Path("reports/local") / args.name / f"corpus-{args.path.stem}.json"
        print(
            f"F1 strict {report['strict']['micro']['f1']:.4f}, masquage {report['masking_rate']:.4f}, "
            f"précision de masquage {report['masking_precision']:.4f} ({report['spurious']} en trop)"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**meta, **report}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Rapport : {out}")


if __name__ == "__main__":
    main()
