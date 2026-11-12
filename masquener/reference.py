"""Modèle de départ, figé : version ONNX de Xenova, à révision et empreintes figées.

    python -m masquener.reference   # télécharge et vérifie les 4 fichiers dans artifacts/reference/
"""

import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download

# Version ONNX pour le navigateur : référence des mesures, et source du tokenizer réutilisé tel quel.
XENOVA_REPO = "Xenova/distilbert-base-multilingual-cased-ner-hrl"
XENOVA_REVISION = "c2a4dbf593c57f47004c5bc2d3770d311aee9c43"
# Fichier → (SHA-256, taille).
XENOVA_FILES = {
    "config.json": ("38847be4dc6699b1218a749ed69f888c2ccc7b4deba98e3c4a1cac8cb34d54c8", 927),
    "tokenizer.json": ("bf1b59b7b11c95f194f51708d918eea378e09d05f84c0e1656dc5180e8117088", 2_919_362),
    "tokenizer_config.json": ("2d61ce6c7646881e0e7ef08e3b5dd655a19553ab85c41b1a3c27090a63ff6f49", 373),
    "onnx/model_quantized.onnx": ("24a0b98f4dd4cd92842f5a541272f86f760225a64a29928eddef14bdb2edb986", 135_359_829),
}

# Original PyTorch, point de départ de l'affinage (licence AFL-3.0).
DAVLAN_REPO = "Davlan/distilbert-base-multilingual-cased-ner-hrl"
DAVLAN_REVISION = "d421f57d5b1d36b375408588669e9340f9b11a89"

REFERENCE_DIR = Path("artifacts/reference")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_reference(dest: Path = REFERENCE_DIR) -> Path:
    """Télécharge les 4 fichiers du modèle Xenova à la révision figée et vérifie leur empreinte."""
    for name, (expected, _) in XENOVA_FILES.items():
        path = Path(hf_hub_download(XENOVA_REPO, name, revision=XENOVA_REVISION, local_dir=dest))
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"{name} : SHA-256 {actual}, attendu {expected}")
    return dest


if __name__ == "__main__":
    print(f"Modèle de référence vérifié : {fetch_reference()}")
