# MasqueNER

Fine-tuning of the multilingual NER model `Davlan/distilbert-base-multilingual-cased-ner-hrl` (ONNX version
`Xenova/…` at revision `c2a4dbf5…`) to better recognize the names of people, places and organizations
specific to each country. A single multilingual model, trained on all countries in `countries/`, measured country by country.

Rules: no real personal data in training (synthetic sentences only); base model labels left unchanged;
real evaluation corpora are read in place, never copied or published.

## Installation (Windows, no GPU)

```bash
uv sync
uv run python -m masquener.reference      # base model, verified by SHA-256 → artifacts/reference/
uv run pytest
```

## Evaluation

```bash
uv run python -m masquener.evaluate --model artifacts/reference --name reference gold conll
uv run python -m masquener.evaluate --model artifacts/reference --name reference corpus PATH/corpus-ci.json
```

- `reports/<name>/`: versioned reports (public datasets).
- `reports/local/<name>/`: reports on real corpora, ignored by git, figures only.
- `reports/reference/synthetic.json`: synthetic country test (see below).

## Training data

```bash
uv run python -m masquener.generate   # country sentences → data/synthetic/{train,dev,test}.jsonl
uv run python -m masquener.distill    # rehearsal: CoNLL + WikiANN (text only) labeled by the reference → data/distill/
uv run python -m masquener.evaluate --model artifacts/reference --name reference synthetic
```

- Each sentence template and each list entry belongs to exactly one split (stable hashing, 80/10/10): the synthetic
  test set only contains sentences and names absent from training.
- Dates and numbers are written but not labeled: the base model never predicts DATE, and this kind of data
  is handled by separate rules (patterns).
- Distillation: the original CoNLL and WikiANN labels are discarded; only their train and validation splits are read.

## Training (Kaggle, GPU)

In a Kaggle notebook with GPU and Internet access, at the root of a copy of the repository:

```bash
pip install -q onnx onnxruntime && pip install -q -e . --no-deps   # keep Kaggle's torch and transformers
python -m masquener.generate && python -m masquener.distill        # or reuse already generated data/
python -m masquener.train --name v1                                # → artifacts/v1/ (Xenova format, quantized ONNX like the reference)
```

Bring back `artifacts/v1/` (without `v1-torch/`), then evaluate and check locally:

```bash
for d in conll masakhaner wikiann; do uv run python -m masquener.evaluate --model artifacts/v1 --name v1 gold $d; done
uv run python -m masquener.evaluate --model artifacts/v1 --name v1 synthetic
uv run python -m masquener.evaluate --model artifacts/v1 --name v1 corpus PATH/corpus-ci.json   # each real corpus
uv run python -m masquener.gate --name v1     # ACCEPTED / REJECTED, threshold −0.5 pt on each blocking metric
```

Quick CPU test: `uv sync --extra train` then `uv run python -m masquener.train --name test --max-steps 4 --batch 8`.

## Licenses

- Code: MIT (`LICENSE`).
- Country data: licenses of their sources; lists from OpenStreetMap under ODbL 1.0 (`countries/LICENSE-DATA.md`).
- Fine-tuned model: derived from `Davlan/distilbert-base-multilingual-cased-ner-hrl` (AFL-3.0), released under AFL-3.0.
- Not distributed: `data/distill/` (contains CoNLL 2003 Reuters text, which cannot be redistributed; `masquener.distill`
  rebuilds it), the real corpora and `reports/local/`.
