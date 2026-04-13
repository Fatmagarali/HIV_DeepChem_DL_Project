# HIV DeepChem DL Project

Refactor du notebook d'exploration HIV/MolNet en projet Python testable avec:

- `src/models.py` pour les wrappers `RandomForestHIV`, `GraphConvHIV`, `AttentiveFPHIV`
- `src/featurizers.py` pour les vues `ECFP`, `ConvMol`, `MolGraphConv`
- `src/train.py` comme point d'entrée CLI d'entraînement
- `src/inference.py` pour la prédiction à partir de checkpoints sauvegardés
- `app/api.py` pour une API FastAPI et un frontend HTML minimal

## Structure

```text
src/
├── __init__.py
├── models.py
├── featurizers.py
├── train.py
├── inference.py
└── utils.py

app/
├── __init__.py
├── api.py
├── config.py
└── templates/
    └── index.html
```

Les checkpoints et métadonnées sont écrits dans `models/`. Les figures et rapports sont écrits dans `artifacts/`.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Le modèle AttentiveFP dépend de `dgl` et `dgllife`. Ils sont optionnels pour garder l'installation légère. Si vous voulez entraîner ce modèle, installez aussi:

```bash
pip install dgl dgllife
```

## Entraînement

Le script charge le dataset MolNet HIV, construit les trois vues de features et entraîne les modèles demandés.

```bash
python -m src.train --model all --epochs 30 --seed 42
```

Options utiles:

- `--model rf|graphconv|attentivefp|all`
- `--epochs 30`
- `--reload-data` pour forcer un rechargement du dataset
- `--output-dir artifacts` pour changer le dossier de sortie
- `--skip-plots` pour éviter la génération des figures

## API

Lancer l'API localement:

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `GET /models`
- `POST /predict`

Exemple de requête:

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"graphconv\",\"threshold\":0.5,\"smiles\":[\"CCO\",\"CC(=O)O\"]}"
```

## Docker

```bash
docker compose up --build
```

Le service expose l'API sur `http://localhost:8000`.

## Variables d'environnement

- `HIV_SEED`
- `HIV_DEFAULT_MODEL`
- `HIV_DEVICE`
- `HIV_API_HOST`
- `HIV_API_PORT`
- `HIV_MODELS_DIR`
- `HIV_ARTIFACTS_DIR`
- `HIV_EPOCHS`
- `HIV_BATCH_SIZE`
- `HIV_LEARNING_RATE`

## Notes de refactor

Le notebook original contient aussi des analyses exploratoires supplémentaires. La logique centrale a été déplacée dans les modules ci-dessus pour rendre le projet plus facile à tester, déployer et maintenir.
