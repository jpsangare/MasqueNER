# Licences des données des pays (`countries/`)

Le code du dépôt est sous licence MIT (`LICENSE`). Les listes et modèles de phrases de ce dossier suivent les
licences de leurs sources, détaillées pays par pays dans `countries/<code>/country.yaml` et en tête de chaque liste.

## Côte d'Ivoire (`countries/ci/`)

| Source | Licence | Listes concernées |
|---|---|---|
| data.gouv.ci (producteur data354-CI) | Licence Ouverte / Open Licence 2.0 | villes, régions, quartiers, villages, écoles, santé (CMU), entreprises |
| OpenStreetMap, via data.gouv.ci | ODbL 1.0, © contributeurs OpenStreetMap | santé, quartiers, entreprises (banques, hôtels, restaurants), sites |
| Wikidata | CC0 1.0 | prénoms, noms |
| Listes rédigées à la main (`manual/`) | MIT, comme le dépôt | prénoms, noms, institutions, entreprises |

Les listes qui contiennent des données OpenStreetMap (`sante.txt`, `quartiers.txt`, `entreprises.txt`, `sites.txt`)
forment une base de données dérivée : elles sont diffusées sous **ODbL 1.0**
(<https://opendatacommons.org/licenses/odbl/1-0/>), avec la mention « © contributeurs OpenStreetMap ». Toute
redistribution de ces listes, ou d'une base qui en dérive, doit garder cette licence et cette mention.

Les modèles de phrases (`templates/`) sont rédigés à la main et suivent la licence MIT.

## France (`countries/fr/`)

| Source | Licence | Listes concernées |
|---|---|---|
| Insee, base des prénoms 2024 | Licence Ouverte / Open Licence 2.0 | prénoms |
| Insee, fichier des personnes décédées 2024 (noms comptés seulement) | Licence Ouverte / Open Licence 2.0 | noms |
| geo.api.gouv.fr (Code officiel géographique de l'Insee) | Licence Ouverte / Open Licence 2.0 | villes, départements, régions |
| Wikidata | CC0 1.0 | entreprises, santé, établissements |
| Listes rédigées à la main (`manual/`) | MIT, comme le dépôt | institutions |

## Exemples négatifs communs (`countries/commun/`)

Modèles de phrases rédigés à la main, sans liste ni entité : licence MIT, comme le dépôt.

## Règle sur les personnes

Les listes `prenoms.txt` et `noms.txt` ne contiennent que des mots isolés, portés par au moins deux personnes, sans
aucun lien entre eux (ni nom complet, ni lieu, ni fonction). Aucune donnée personnelle réelle n'est utilisée pour
l'entraînement.
