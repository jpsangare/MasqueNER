---
license: afl-3.0
base_model: Davlan/distilbert-base-multilingual-cased-ner-hrl
library_name: transformers.js
pipeline_tag: token-classification
language: [fr, en, es, nl, de, it, pt, ar, zh, lv]
tags: [ner, named-entity-recognition, privacy, anonymization, pii, cote-d-ivoire, onnx, distilbert]
metrics: [f1]
---

# distilbert-ner-masquener

**English summary.** A DistilBERT NER model (PER, ORG, LOC) fine-tuned to recognise Ivorian names of people, places and
organisations, so that personal data can be masked in the browser before a text is sent to an AI assistant. Trained
only on synthetic sentences (no real personal data). Same size, format and labels as the base model.

---

Modèle de reconnaissance de noms (personnes, lieux, organisations) adapté aux documents de **Côte d'Ivoire**. Il sert à
masquer les données personnelles **directement dans le navigateur**, avant qu'un texte soit envoyé à une IA.

C'est un affinage complet de [`Davlan/distilbert-base-multilingual-cased-ner-hrl`](https://huggingface.co/Davlan/distilbert-base-multilingual-cased-ner-hrl)
(révision `d421f57d`), livré au même format que sa version ONNX
[`Xenova/distilbert-base-multilingual-cased-ner-hrl`](https://huggingface.co/Xenova/distilbert-base-multilingual-cased-ner-hrl) :
mêmes étiquettes, même tokenizer, ONNX quantifié en entiers 8 bits (135 Mo).

## Résultats

F1 strict (type et limites exacts). « Référence » = le modèle de départ, mesuré avec le même code.

| Jeu | Référence | MasqueNER v1 |
|---|---|---|
| Documents administratifs ivoiriens réels, 25 documents · personnes | 0,40 | **0,72** |
| · lieux | 0,60 | **0,64** |
| · part des noms entièrement masqués | 68 % | **84 %** |
| Phrases ivoiriennes synthétiques (modèles et noms jamais vus à l'entraînement) | 0,607 | **0,872** |
| CoNLL 2002/2003 (en, es, nl), moyenne | 0,896 | 0,896 |
| MasakhaNER 2.0 (20 langues africaines), moyenne | 0,439 | 0,445 |
| WikiANN (10 langues, étiquettes automatiques, indicatif), moyenne | 0,560 | 0,564 |

Le modèle a été accepté par un contrôle automatique : aucune mesure bloquante (CoNLL, MasakhaNER, phrases synthétiques,
corpus réels : personnes, lieux, taux de masquage) ne recule de plus de 0,5 point.

## Données d'entraînement

40 000 phrases, à parts égales :

- **20 000 phrases ivoiriennes inventées.** 88 modèles de phrases (courriers administratifs, fiches, messages,
  demandes à un assistant) dont les cases sont remplies au hasard à partir de listes publiques : villes, quartiers,
  villages, écoles, établissements de santé, entreprises (data.gouv.ci, OpenStreetMap), institutions (listes écrites à la
  main). Les personnes sont composées de **prénoms et noms de famille isolés**, portés chacun par au moins deux
  personnalités (Wikidata), recombinés au hasard : aucun nom complet réel.
- **20 000 phrases de révision** (CoNLL 2002/2003 en, es, nl ; WikiANN, 10 langues), dont les étiquettes d'origine sont
  remplacées par les prédictions du modèle de départ (distillation), pour qu'il n'oublie pas ce qu'il savait.

Aucune donnée personnelle réelle n'a servi à l'entraînement. Les documents réels servent seulement d'examen ; ils ne
sont pas publiés.

Entraînement : 3 époques, pas d'apprentissage 3e-5, lots de 32, 2 GPU T4 (Kaggle), environ 10 minutes.

## Utilisation

JavaScript, dans le navigateur ([transformers.js](https://huggingface.co/docs/transformers.js)) :

```js
import { pipeline } from "@huggingface/transformers";

const ner = await pipeline("token-classification", "JPMSA1212/distilbert-ner-masquener", { dtype: "q8" });
// une entrée par sous-mot (B-PER, I-PER…) ; à regrouper en entités comme dans le dépôt
const tokens = await ner("Monsieur YAO Exemple, domicilié à Koumassi, est convoqué à Sinfra.");
```

Python, avec les poids PyTorch (`model.safetensors`) :

```python
from transformers import pipeline

ner = pipeline("token-classification", model="JPMSA1212/distilbert-ner-masquener", aggregation_strategy="simple")
ner("Le rapport de mission de Mme KONÉ Exemple à Bondoukou a été transmis à la Direction régionale de la Santé.")
```

Les mesures ci-dessus portent sur la version ONNX quantifiée, celle du navigateur. Sur 300 phrases de test, elle donne
la même étiquette que les poids PyTorch pour 98,8 % des sous-mots ; l'écart vient de la quantification.
Pour regrouper les sous-mots en entités (un mot est une entité si l'un de ses sous-mots l'est), voir `masquener/onnx_ner.py` dans le
[dépôt MasqueNER](https://github.com/jpsangare/MasqueNER).

Les noms des exemples sont fictifs.

Étiquettes : `O`, `B-/I-PER`, `B-/I-ORG`, `B-/I-LOC`, `B-/I-DATE`. Le modèle ne prédit pas les dates (le modèle de
départ non plus) : elles relèvent de règles à part.

## Limites

- **Lieux encore moyens** (F1 0,64 sur les documents réels) : quartiers et villages restent souvent manqués.
- **Erreurs de type** : un nom peut être masqué avec le mauvais type (une ville prise pour une personne), surtout dans
  les messages écrits sans majuscules. Pour le masquage, le nom est caché quand même.
- **Petit corpus réel** : 25 documents, 120 noms de personnes, 165 lieux. Les organisations y sont peu annotées
  (les ministères et directions ne le sont pas), d'où une note ORG non significative.
- **Langues africaines** : la moyenne MasakhaNER progresse (+0,6 pt), mais 9 langues sur 20 reculent un peu, jusqu'à
  −1,6 pt (chichewa), −1,4 pt (swahili), −1,3 pt (setswana) ; l'espagnol de CoNLL recule de 0,6 pt.
- **Français seulement** pour la Côte d'Ivoire : rien n'a été fait pour le dioula, le baoulé ou les autres langues
  locales.
- Comme tout modèle de détection, il peut laisser passer des noms : il réduit le risque de fuite, il ne le supprime
  pas.

## Licence et attribution

- Modèle : AFL-3.0, comme le modèle de départ de David Adelani (Davlan).
- Listes de lieux et d'organisations : Licence Ouverte 2.0 (data.gouv.ci), ODbL 1.0 (© contributeurs OpenStreetMap),
  CC0 (Wikidata). Détail dans `countries/LICENSE-DATA.md` du dépôt.
- Jeux d'évaluation : CoNLL 2002/2003, MasakhaNER 2.0, WikiANN, à révisions figées.
