# MasqueNER

Affinage du modèle NER multilingue `Davlan/distilbert-base-multilingual-cased-ner-hrl` (version ONNX
`Xenova/…` à la révision `c2a4dbf5…`) pour mieux reconnaître les noms de personnes, de lieux et d'organisations
propres à chaque pays. Un seul modèle multilingue, entraîné sur tous les pays de `countries/`, mesuré pays par pays.

Règles : aucune donnée personnelle réelle à l'entraînement (phrases synthétiques uniquement) ; étiquettes du modèle
de départ inchangées ; corpus réels d'évaluation lus sur place, jamais copiés ni publiés.

## Installation (Windows, sans GPU)

```bash
uv sync
uv run python -m masquener.reference      # modèle de départ, vérifié par SHA-256 → artifacts/reference/
uv run pytest
```

## Mesure

```bash
uv run python -m masquener.evaluate --model artifacts/reference --name reference gold conll
uv run python -m masquener.evaluate --model artifacts/reference --name reference corpus CHEMIN/corpus-ci.json
```

- `reports/<nom>/` : rapports versionnés (jeux publics).
- `reports/local/<nom>/` : rapports sur corpus réels, ignorés par git, chiffres seulement.
- `reports/reference/synthetic.json` : test synthétique des pays (voir ci-dessous).

## Données d'entraînement

```bash
uv run python -m masquener.generate   # phrases pays → data/synthetic/{train,dev,test}.jsonl
uv run python -m masquener.distill    # révision : CoNLL + WikiANN (texte seul) étiquetés par la référence → data/distill/
uv run python -m masquener.evaluate --model artifacts/reference --name reference synthetic
```

- Chaque modèle de phrase et chaque ligne de liste appartient à un seul jeu (hachage stable, 80/10/10) : le test
  synthétique ne contient que des phrases et des noms absents de l'entraînement.
- Dates et nombres sont écrits mais non étiquetés : le modèle de départ ne prédit jamais DATE, et
  ces données relèvent de règles à part (motifs).
- Distillation : les étiquettes d'origine de CoNLL et WikiANN sont jetées ; seuls leurs jeux train et validation sont lus.

## Entraînement (Kaggle, GPU)

Dans un carnet Kaggle avec GPU et accès Internet, à la racine d'une copie du dépôt :

```bash
pip install -q onnx onnxruntime && pip install -q -e . --no-deps   # garder torch et transformers de Kaggle
python -m masquener.generate && python -m masquener.distill         # ou reprendre data/ déjà produit
python -m masquener.train --name v1            # → artifacts/v1/ (format Xenova, ONNX quantifié comme la référence)
```

Rapatrier `artifacts/v1/` (sans `v1-torch/`), puis mesurer et contrôler en local :

```bash
for d in conll masakhaner wikiann; do uv run python -m masquener.evaluate --model artifacts/v1 --name v1 gold $d; done
uv run python -m masquener.evaluate --model artifacts/v1 --name v1 synthetic
uv run python -m masquener.evaluate --model artifacts/v1 --name v1 corpus CHEMIN/corpus-ci.json   # chaque corpus réel
uv run python -m masquener.gate --name v1     # ACCEPTÉ / REFUSÉ, seuil −0,5 pt sur chaque mesure bloquante
```

Essai rapide sur processeur : `uv sync --extra train` puis `uv run python -m masquener.train --name essai --max-steps 4 --batch 8`.

## Licences

- Code : MIT (`LICENSE`).
- Données des pays : licences de leurs sources ; listes issues d'OpenStreetMap sous ODbL 1.0 (`countries/LICENSE-DATA.md`).
- Modèle affiné : dérivé de `Davlan/distilbert-base-multilingual-cased-ner-hrl` (AFL-3.0), diffusé sous AFL-3.0.
- Non diffusés : `data/distill/` (contient du texte CoNLL 2003, Reuters, non redistribuable ; `masquener.distill` le
  reconstruit), les corpus réels et `reports/local/`.

