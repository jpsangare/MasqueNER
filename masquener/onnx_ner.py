"""Inférence d'un modèle au format Xenova (config.json, tokenizer.json, onnx/model_quantized.onnx) sur processeur.

Le même code mesure l'ancien et le nouveau modèle, dans le format servi au navigateur (transformers.js).
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .labels import check_config
from .metrics import Span

MAX_TOKENS = 512


class OnnxNer:
    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        check_config(config)
        self.id2label = {int(k): v for k, v in config["id2label"].items()}
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(MAX_TOKENS)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 0
        self.session = ort.InferenceSession(
            str(model_dir / "onnx" / "model_quantized.onnx"), options, providers=["CPUExecutionProvider"]
        )

    def _probabilities(self, encodings) -> list[np.ndarray]:
        """Probabilités par jeton (softmax des logits) ; un lot est complété par des [PAD] masqués."""
        length = max(len(e.ids) for e in encodings)
        ids = np.zeros((len(encodings), length), dtype=np.int64)
        mask = np.zeros_like(ids)
        for i, e in enumerate(encodings):
            ids[i, : len(e.ids)] = e.ids
            mask[i, : len(e.ids)] = 1
        (logits,) = self.session.run(["logits"], {"input_ids": ids, "attention_mask": mask})
        logits = logits - logits.max(-1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(-1, keepdims=True)
        return [probs[i, : len(e.ids)] for i, e in enumerate(encodings)]

    def tag_words(self, sentences: list[list[str]], batch_size: int = 32) -> list[list[str]]:
        """Étiquette BIO de chaque mot d'un texte déjà découpé : celle de son premier sous-mot.

        Les mots au-delà de la limite de 512 jetons reçoivent « O ».
        """
        out = []
        for b in range(0, len(sentences), batch_size):
            batch = sentences[b : b + batch_size]
            encodings = self.tokenizer.encode_batch(batch, is_pretokenized=True)
            for words, e, probs in zip(batch, encodings, self._probabilities(encodings)):
                tags = ["O"] * len(words)
                seen = set()
                for t, w in enumerate(e.word_ids):
                    if w is not None and w not in seen:
                        seen.add(w)
                        tags[w] = self.id2label[int(probs[t].argmax())]
                out.append(tags)
        return out

    def spans(self, text: str) -> list[tuple[int, int, str, float]]:
        """Entités d'un texte brut, en positions de caractères : (début, fin, type, score).

        Le texte est traité ligne par ligne (une entité ne passe jamais à la ligne), et les sous-mots sont regroupés
        ainsi : un mot est une entité si l'un de ses
        sous-mots l'est, avec le type du sous-mot non « O » le plus sûr ; les mots consécutifs de même type
        fusionnent, sauf préfixe B-.
        """
        found = []
        offset = 0
        for line in text.split("\n"):
            if line.strip():
                found += [(s + offset, e + offset, t, p) for s, e, t, p in self._line_spans(line)]
            offset += len(line) + 1
        return found

    def _line_spans(self, line: str) -> list[tuple[int, int, str, float]]:
        encoding = self.tokenizer.encode(line)
        if len(encoding.overflowing):
            raise ValueError("ligne de plus de 512 jetons : à découper avant l'évaluation")
        probs = self._probabilities([encoding])[0]
        # Mots : un sous-mot « ## » collé au précédent le prolonge.
        words: list[list[tuple[int, int, str, float]]] = []
        for t, (token, (start, end)) in enumerate(zip(encoding.tokens, encoding.offsets)):
            if token in ("[CLS]", "[SEP]") or start == end:
                continue
            k = int(probs[t].argmax())
            piece = (start, end, self.id2label[k], float(probs[t, k]))
            if words and token.startswith("##") and start == words[-1][-1][1]:
                words[-1].append(piece)
            else:
                words.append([piece])

        spans: list[list] = []  # [début, fin, type, scores]
        open_ = False
        for word in words:
            tagged = [p for p in word if p[2] != "O"]
            if not tagged:
                open_ = False
                continue
            best = max(tagged, key=lambda p: p[3])
            prefix, _, kind = best[2].partition("-")
            if open_ and spans[-1][2] == kind and prefix != "B":
                spans[-1][1] = word[-1][1]
                spans[-1][3].append(best[3])
            else:
                spans.append([word[0][0], word[-1][1], kind, [best[3]]])
            open_ = True
        return [(s, e, t, sum(p) / len(p)) for s, e, t, p in spans]


def char_spans(spans: list[tuple[int, int, str, float]]) -> set[Span]:
    return {(t, s, e) for s, e, t, _ in spans}
