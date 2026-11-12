"""Contrôle bloquant : le nouveau modèle remplace la référence seulement s'il ne recule nulle part.

    python -m masquener.gate --name v1 [--baseline reference]

Lit les rapports déjà produits par masquener.evaluate (reports/<nom>/ et reports/local/<nom>/) et compare, pour chaque
mesure bloquante de la référence, le nouveau score à l'ancien. Seuil : −0,5 point de F1 (ou de taux de masquage).
- jeux publics bloquants (conll, masakhaner) : F1 moyen sur les langues ; le détail par langue est affiché ;
- phrases synthétiques : F1 de chaque pays ;
- corpus réels (reports/local) : F1 PER, F1 LOC et taux de masquage. ORG n'y est pas bloquant (le corpus n'annote
  pas les institutions, que le modèle doit pourtant trouver) ;
- wikiann : indicatif, affiché seulement.
Une mesure de la référence sans équivalent pour le nouveau modèle est un échec. Code de sortie 1 si refusé.
"""

import argparse
import json
import sys
from pathlib import Path

THRESHOLD = 0.005


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def measures(report: dict) -> tuple[dict[str, float], dict[str, float]]:
    """(mesures bloquantes, mesures indicatives) d'un rapport, sous forme nom → score."""
    if "corpus" in report:
        s = report["strict"]
        return {"PER F1": s["PER"]["f1"], "LOC F1": s["LOC"]["f1"], "masquage": report["masking_rate"]}, {"ORG F1": s["ORG"]["f1"]}
    r = report["results"]
    if report["dataset"] == "synthetic":
        return {f"{c} F1": v["strict"]["micro"]["f1"] for c, v in r.items()}, {f"{c} masquage": v["masking_rate"] for c, v in r.items()}
    langs = {f"{lang} F1": v["micro"]["f1"] for lang, v in r.items() if lang != "macro_f1"}
    return ({"F1 moyen": r["macro_f1"]}, langs) if report["blocking"] else ({}, {"F1 moyen": r["macro_f1"], **langs})


def compare(baseline: str, name: str) -> list[str]:
    """Lignes du rapport ; celles qui commencent par « ÉCHEC » bloquent."""
    lines = []
    for root in (Path("reports"), Path("reports/local")):
        for ref_path in sorted((root / baseline).glob("*.json")):
            ref, new = load(ref_path), load(root / name / ref_path.name)
            title = f"{root.name}/{ref_path.stem}"
            if new is None:
                lines.append(f"ÉCHEC  {title} : pas mesuré pour {name}")
                continue
            if new.get("limit") or ref.get("limit"):
                lines.append(f"ÉCHEC  {title} : mesure partielle (--limit), à refaire en entier")
                continue
            (ref_block, ref_info), (new_block, new_info) = measures(ref), measures(new)
            for kind, ref_m, new_m in (("bloquant", ref_block, new_block), ("indicatif", ref_info, new_info)):
                for key, old in ref_m.items():
                    value = new_m.get(key)
                    if value is None:
                        lines.append(f"ÉCHEC  {title} {key} : absent" if kind == "bloquant" else f"       {title} {key} : absent")
                        continue
                    delta = value - old
                    flag = "ÉCHEC " if kind == "bloquant" and delta < -THRESHOLD else "ok    " if kind == "bloquant" else "      "
                    lines.append(f"{flag} {title} {key} : {old:.4f} -> {value:.4f} ({delta * 100:+.2f} pt){'' if kind == 'bloquant' else ' · indicatif'}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--baseline", default="reference")
    args = parser.parse_args()
    lines = compare(args.baseline, args.name)
    print("\n".join(lines))
    failed = [line for line in lines if line.startswith("ÉCHEC")]
    print(f"\n{'REFUSÉ' if failed else 'ACCEPTÉ'} : {len(failed)} échec(s) sur {sum(not l.startswith(' ') for l in lines)} mesures bloquantes")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
