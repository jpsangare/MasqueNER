"""Étiquettes du modèle de départ, figées : les applications lisent id2label dans config.json, rien ne doit changer."""

ID2LABEL = {
    0: "O",
    1: "B-DATE",
    2: "I-DATE",
    3: "B-PER",
    4: "I-PER",
    5: "B-ORG",
    6: "I-ORG",
    7: "B-LOC",
    8: "I-LOC",
}
LABEL2ID = {label: i for i, label in ID2LABEL.items()}
ENTITY_TYPES = ("DATE", "PER", "ORG", "LOC")


def check_config(config: dict) -> None:
    """Refuse un config.json dont les étiquettes diffèrent de celles du modèle de départ."""
    id2label = {int(k): v for k, v in config["id2label"].items()}
    if id2label != ID2LABEL or config.get("label2id") != LABEL2ID:
        raise ValueError(f"étiquettes différentes du modèle de départ : {config['id2label']}")


IGNORE = -100  # jeton spécial : exclu de la perte


def token_labels(offsets: list[tuple[int, int]], entities: list[list]) -> list[int]:
    """Étiquette de chaque jeton d'un texte brut, d'après ses entités en caractères ([début, fin, type]).

    Tous les sous-mots d'une entité sont étiquetés (B- sur le premier, I- ensuite), comme les lit OnnxNer.spans :
    un mot est une entité si l'un de ses sous-mots l'est. Les jetons spéciaux ([CLS], [SEP]) sont ignorés.
    """
    labels = []
    for start, end in offsets:
        if start == end:
            labels.append(IGNORE)
            continue
        inside = [(s, t) for s, e, t in entities if s < end and start < e]
        if not inside:
            labels.append(LABEL2ID["O"])
        else:
            s, t = inside[0]
            labels.append(LABEL2ID[("B-" if start <= s else "I-") + t])
    return labels
