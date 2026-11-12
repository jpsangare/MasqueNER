"""Affinage du modèle de départ puis export au format navigateur (Xenova : ONNX quantifié).

    pip install -q onnx onnxruntime && pip install -q -e . --no-deps   # Kaggle : garder torch et transformers fournis
    python -m masquener.train --name v1 [--epochs 3] [--lr 3e-5] [--batch 32] [--max-steps N]

Entrées (produites avant, voir README) : data/synthetic/{train,dev}.jsonl et data/distill/{train,dev}.jsonl,
mélangés tels quels (autant de phrases pays que de phrases de révision).
Sortie : artifacts/<nom>/ au même format que artifacts/reference/, mesurable avec masquener.evaluate :
    config.json, tokenizer.json, tokenizer_config.json (ceux du modèle de départ), onnx/model_quantized.onnx,
    training.json (réglages, empreintes des données, pertes).
Le modèle PyTorch affiné reste dans artifacts/<nom>-torch/ (non versionné).
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from .labels import check_config, token_labels
from .reference import DAVLAN_REPO, DAVLAN_REVISION, REFERENCE_DIR, fetch_reference, sha256

DATA = {"train": ["data/synthetic/train.jsonl", "data/distill/train.jsonl"],
        "dev": ["data/synthetic/dev.jsonl", "data/distill/dev.jsonl"]}  # fmt: skip
MAX_TOKENS = 128


class Rows(torch.utils.data.Dataset):
    def __init__(self, files: list[str], tokenizer):
        self.items = []
        for f in files:
            for line in Path(f).read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                enc = tokenizer(row["text"], truncation=True, max_length=MAX_TOKENS, return_offsets_mapping=True)
                labels = token_labels(enc.pop("offset_mapping"), row["entities"])
                self.items.append({**enc, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class _Logits(torch.nn.Module):
    """Le modèle réduit à (input_ids, attention_mask) → logits : les entrées et la sortie du format Xenova."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def to_onnx(torch_dir: Path, path: Path) -> None:
    """Export ONNX par torch (exportateur TorchScript, opset 14), longueurs de lot et de phrase libres.

    Vérifie que l'ONNX non quantifié redonne les logits de PyTorch.
    """
    import numpy as np
    import onnxruntime as ort

    model = AutoModelForTokenClassification.from_pretrained(torch_dir).eval()
    tokenizer = AutoTokenizer.from_pretrained(torch_dir)
    sample = tokenizer(["Kouassi Yao travaille à Abidjan.", "Le dossier de N'Goran a été transmis à l'Ageroute le 3 mars."],
                       padding=True, return_tensors="pt")  # fmt: skip
    wrapped = _Logits(model).eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    axes = {0: "batch", 1: "sequence"}
    with torch.no_grad():
        torch.onnx.export(wrapped, (sample["input_ids"], sample["attention_mask"]), str(path),
                          input_names=["input_ids", "attention_mask"], output_names=["logits"],
                          dynamic_axes={"input_ids": axes, "attention_mask": axes, "logits": axes},
                          opset_version=14, dynamo=False)  # fmt: skip
        expected = wrapped(sample["input_ids"], sample["attention_mask"]).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (got,) = session.run(["logits"], {k: sample[k].numpy() for k in ("input_ids", "attention_mask")})
    diff = float(np.abs(got - expected).max())
    if diff > 1e-3:
        raise RuntimeError(f"export ONNX infidèle : écart maximal {diff} sur les logits")


def export(torch_dir: Path, out: Path) -> None:
    """ONNX puis quantification dynamique comme le modèle de départ, fichiers d'accompagnement de celui-ci.

    Réglages de la référence (poids int8 par canal, plage réduite) : le modèle de départ réexporté ainsi retrouve
    ses scores (CoNLL 0,912 contre 0,909 sur 400 phrases par langue) ; quantifié par tenseur, il tombe à 0,73 ;
    sans plage réduite, à 0.
    """
    fetch_reference()
    raw = out / "onnx-raw"
    to_onnx(torch_dir, raw / "model.onnx")
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    quantize_dynamic(raw / "model.onnx", out / "onnx" / "model_quantized.onnx",
                     weight_type=QuantType.QInt8, per_channel=True, reduce_range=True)  # fmt: skip
    shutil.rmtree(raw)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(REFERENCE_DIR / name, out / name)
    config = json.loads((torch_dir / "config.json").read_text(encoding="utf-8"))
    check_config(config)
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="nom du modèle (artifacts/<nom>, reports/<nom>)")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=-1, help="essai rapide (les vrais modèles s'entraînent sans)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out, torch_dir = Path("artifacts") / args.name, Path("artifacts") / f"{args.name}-torch"

    tokenizer = AutoTokenizer.from_pretrained(DAVLAN_REPO, revision=DAVLAN_REVISION)
    model = AutoModelForTokenClassification.from_pretrained(DAVLAN_REPO, revision=DAVLAN_REVISION)
    check_config(model.config.to_dict())
    train, dev = Rows(DATA["train"], tokenizer), Rows(DATA["dev"], tokenizer)
    print(f"Entraînement : {len(train)} phrases ; validation : {len(dev)}")
    gpus = max(1, torch.cuda.device_count())
    total = args.max_steps if args.max_steps > 0 else math.ceil(len(train) / (args.batch * gpus)) * args.epochs
    warmup = int(0.1 * total)  # 10 % de montée progressive du pas d'apprentissage

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(torch_dir / "checkpoints"),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=64,
            warmup_steps=warmup,
            weight_decay=0.01,
            eval_strategy="epoch" if args.max_steps < 0 else "steps",
            eval_steps=None if args.max_steps < 0 else args.max_steps,
            save_strategy="no",
            logging_steps=50,
            fp16=torch.cuda.is_available(),
            seed=args.seed,
            report_to="none",
        ),
        train_dataset=train,
        eval_dataset=dev,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(torch_dir))
    tokenizer.save_pretrained(str(torch_dir))

    export(torch_dir, out)
    log = [e for e in trainer.state.log_history if "loss" in e or "eval_loss" in e]
    (out / "training.json").write_text(json.dumps({
        "base": [DAVLAN_REPO, DAVLAN_REVISION],
        "args": vars(args),
        "data": {f: sha256(Path(f)) for files in DATA.values() for f in files},
        "log": log,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # fmt: skip
    print(f"Modèle : {out} (mesurer : python -m masquener.evaluate --model {out} --name {args.name} ...)")


if __name__ == "__main__":
    main()
