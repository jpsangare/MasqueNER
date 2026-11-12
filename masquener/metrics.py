"""Précision, rappel et F1 au niveau des entités (correspondance exacte du type et des limites)."""

from collections import Counter

Span = tuple[str, int, int]  # (type, début, fin exclue)


def bio_spans(tags: list[str]) -> set[Span]:
    """Entités d'une suite BIO, à la manière de conlleval : un I- qui ne prolonge rien ouvre une entité."""
    spans, start, kind = set(), None, None
    for i, tag in enumerate([*tags, "O"]):
        prefix, _, t = tag.partition("-")
        if kind is not None and (tag == "O" or prefix == "B" or t != kind):
            spans.add((kind, start, i))
            kind = None
        if tag != "O" and kind is None:
            start, kind = i, t
    return spans


def score(gold: list[set[Span]], pred: list[set[Span]], types: tuple[str, ...]) -> dict:
    """Scores par type et globaux (micro) ; seuls les types listés comptent, des deux côtés."""
    tp, n_gold, n_pred = Counter(), Counter(), Counter()
    for g, p in zip(gold, pred, strict=True):
        g = {s for s in g if s[0] in types}
        p = {s for s in p if s[0] in types}
        for s in g & p:
            tp[s[0]] += 1
        for s in g:
            n_gold[s[0]] += 1
        for s in p:
            n_pred[s[0]] += 1

    def prf(t: int, g: int, p: int) -> dict:
        precision = t / p if p else 0.0
        recall = t / g if g else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4), "support": g}

    result = {t: prf(tp[t], n_gold[t], n_pred[t]) for t in types}
    result["micro"] = prf(sum(tp.values()), sum(n_gold.values()), sum(n_pred.values()))
    return result
