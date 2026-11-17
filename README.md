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
- `reports/local/<name>/`: reports on real corpora, ignored by git, figures only. Accepted types:
  `PERSONNE`/`LIEU`/`ORGANISATION` or `PER`/`LOC`/`ORG`; other annotated types (numbers, dates…) are not scored, but
  a masked span that falls on them is not counted as spurious.
- Corpora: besides F1 and masking rate, the **masking precision** (share of masked spans that touch an annotated
  value) and the number of spurious spans. A model that finds more names but also hides “Bonjour”, “Matricule” or
  “Nationalité” lowers it.
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

The gate only compares reports that exist for the baseline: evaluate the reference (and the previous version) on
**the same** corpora and **the same** synthetic test as the candidate. Any change to sentence templates or lists
changes the synthetic test produced by `generate`: re-evaluate `reference` and the previous version with
`evaluate … synthetic`.

**The gate measures the model alone.** An application using it adds its own rules and filters: a model accepted
here must still be measured inside that application before being adopted there (v1, better on its own, hid more
ordinary words once combined with an application's rules).

## Version 2 (prepared, to be trained)

Fixes the v1 shortcomings found in real use:

- **Field labels and courtesy phrases taken for names** (“Matricule”, “Né”, “Civilité”, “Nationalité”, “Bonjour”,
  “Cordialement”, “Veuillez”): v1 had only seen one-line sentences. Sentence templates now accept `\n`, and
  multi-line forms, e-mails, notes and headers have been added, half of them with nothing to mask (about a quarter of
  the training sentences).
- **“SURNAME, Given names”** (“LEFORT, Éric”): a format missing from v1, which spilled over onto the next word.
  Added to the country's person name formats.
- **All-lowercase writing** (“ford”): 5% of the sentences are written entirely in lowercase, entities included.
- **Gate**: masking precision is blocking on every corpus.

```bash
uv run python -m masquener.generate && uv run python -m masquener.distill
# Kaggle: python -m masquener.train --name v2, then bring back artifacts/v2/
for d in conll masakhaner wikiann; do uv run python -m masquener.evaluate --model artifacts/v2 --name v2 gold $d; done
uv run python -m masquener.evaluate --model artifacts/v2 --name v2 synthetic
uv run python -m masquener.evaluate --model artifacts/v2 --name v2 corpus PATH/corpus.json   # each corpus, as for reference and v1
uv run python -m masquener.gate --name v2                  # against the reference
uv run python -m masquener.gate --name v2 --baseline v1    # and against v1
```

Quick CPU test: `uv sync --extra train` then `uv run python -m masquener.train --name test --max-steps 4 --batch 8`.

## Published versions

| Version | Countries | Hugging Face | Revision | Status |
|---|---|---|---|---|
| v1 | Côte d'Ivoire | [`JPMSA1212/distilbert-ner-masquener`](https://huggingface.co/JPMSA1212/distilbert-ner-masquener) | `17efe1a7ee07d7074923c63f5a1280606f6b02e5` | ACCEPTED by `gate`; published ONNX identical to `artifacts/v1` (SHA-256 `5ea5d525…`) |

v1, Ivorian control corpus: PER F1 0.46 → 0.74, masking rate 0.73 → 0.86; no regression on CoNLL, WikiANN or
MasakhaNER. The evaluation is still narrow (25 fictional documents, some names also present in the training lists):
it should be confirmed on a larger corpus.

The published `tokenizer_config.json` (saved by transformers) differs from the Xenova reference one: check that it
loads with transformers.js before using the model in a browser.

## Adding a country

1. Copy `countries/ci/` to `countries/<code>/` and adapt `country.yaml` (languages, lists → entity types, person
   name formats), the `lists/` (public sources, license recorded in `countries/LICENSE-DATA.md`) and the sentence
   templates `templates/<language>.txt`. No shared code needs to change.
2. Build a control corpus for the country (realistic documents, with names **absent** from the training lists),
   read in place and never copied into the repository.
3. Regenerate the data (`generate`, `distill`), retrain on **all** countries (`train --name vN`), evaluate each
   country and the public datasets, then run `gate --name vN --baseline v(N-1)`: no country already covered may
   regress (by default, `gate` compares with `reference`).
4. Publish to the same Hugging Face repository (new revision) and add a row to the table above. Applications that
   pin a revision and file hashes must be updated for each new revision.

## Licenses

- Code: MIT (`LICENSE`).
- Country data: licenses of their sources; lists from OpenStreetMap under ODbL 1.0 (`countries/LICENSE-DATA.md`).
- Fine-tuned model: derived from `Davlan/distilbert-base-multilingual-cased-ner-hrl` (AFL-3.0), released under AFL-3.0.
- Not distributed: `data/distill/` (contains CoNLL 2003 Reuters text, which cannot be redistributed; `masquener.distill`
  rebuilds it), the real corpora and `reports/local/`.
