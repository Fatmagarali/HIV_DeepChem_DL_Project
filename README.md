# HIV Activity Prediction — DeepChem Deep Learning Project

Predict anti-HIV activity of molecules using three deep-learning models
trained on the [MolNet HIV dataset](https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html):

| Model | Architecture | Representation |
|-------|-------------|----------------|
| **Random Forest** | sklearn ensemble | ECFP4 (1024-bit) |
| **GraphConv** | Spatial GCN | Molecular graph |
| **AttentiveFP** | GCN + attention | Molecular graph + edge features |

> Reference: Li et al. (2022) — *Deep learning methods for molecular representation and property prediction*

---

## Project structure

```
HIV_DeepChem_DL_Project/
├── HIV_DeepChem_DL_Project.ipynb   # Original exploration notebook
├── notebooks/
│   └── demo.ipynb                  # Lightweight inference demo
├── src/
│   ├── __init__.py
│   ├── models.py                   # Model classes (RF, GraphConv, AttentiveFP)
│   ├── train.py                    # Training CLI
│   └── inference.py                # Inference / prediction helpers
├── app/
│   ├── __init__.py
│   ├── config.py                   # All constants / env-var config
│   └── api.py                      # FastAPI REST API
├── models/                         # Saved checkpoints (git-ignored)
├── requirements.txt                # Core dependencies (no dgl)
├── requirements-attentivefp.txt    # Optional dgl deps — see security note below
└── README.md
```

---

## ⚠️ Security notice

| Dependency | Issue | Status |
|------------|-------|--------|
| **dgl ≤ 2.4.0** | RCE via pickle deserialization in `rpc.recv_request()` | **No patched version available** |
| **torch < 2.6.0** | Heap overflow, use-after-free, `torch.load` RCE | Patched in **2.6.0** (used here) |

**Consequences for this project:**

- `torch` is pinned to `2.6.0` in `requirements.txt`.
- `dgl` / `dgllife` are **removed from `requirements.txt`**.  
  The AttentiveFP model will gracefully report itself as unavailable.
- The REST API **blocks `attentivefp` requests** (`/predict` returns HTTP 422)
  to prevent the dgl RCE from being reachable over the network.
- A separate `requirements-attentivefp.txt` file documents the risk and
  provides the packages for users who explicitly accept it (e.g. isolated
  offline research environments).

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Fatmagarali/HIV_DeepChem_DL_Project.git
cd HIV_DeepChem_DL_Project

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install core dependencies (Random Forest + GraphConv + API)
pip install -r requirements.txt
```

**Optional — AttentiveFP** (read the security warning in that file first):

```bash
# ⚠️  dgl has an unpatched RCE vulnerability — only use in isolated environments
pip install -r requirements-attentivefp.txt
```

---

## Training

```bash
# Train all three models (default: 30 epochs)
python -m src.train --model all

# Train a single model with custom hyperparameters
python -m src.train --model random_forest --epochs 30
python -m src.train --model graphconv --epochs 50 --learning-rate 1e-3
python -m src.train --model attentivefp --epochs 50 --batch-size 64

# All options
python -m src.train --help
```

Checkpoints are saved to `./models/` (configurable with `--model-dir`).

---

## Inference

```bash
# Predict from the command line (SMILES as arguments)
python src/inference.py --smiles "CCO" "c1ccccc1" --model random_forest

# Predict from a file (one SMILES per line)
python src/inference.py --smiles-file molecules.txt --model graphconv

# Save results to JSON
python src/inference.py --smiles "CCO" --model random_forest --output results.json
```

**Import in Python:**

```python
from src.inference import predict_smiles

results = predict_smiles(
    smiles_list=["CCO", "c1ccccc1"],
    model_name="random_forest",
    threshold=0.5,
)
# {'smiles': [...], 'predictions': [...], 'labels': [...]}
```

---

## REST API

```bash
# Start the API server
python -m uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: **http://localhost:8000/docs**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + available models |
| `/models` | GET | List all models with metadata |
| `/predict` | POST | Predict HIV activity for SMILES |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": ["CCO", "c1ccccc1"], "model": "random_forest"}'
```

---

## Configuration

All constants live in `app/config.py` and can be overridden via a `.env` file:

```dotenv
SEED=42
BATCH_SIZE=128
LEARNING_RATE=0.001
EPOCHS=30
MODEL_DIR=./models
HOST=0.0.0.0
PORT=8000
THRESHOLD=0.5
```

---

## Demo notebook

Open `notebooks/demo.ipynb` for an interactive walk-through:
load data → predict → visualise — **no training code in the notebook**.

---

## Requirements

- Python 3.9+
- deepchem 2.7.1
- tensorflow 2.15.0
- torch 2.6.0
- dgl + dgllife (optional, for AttentiveFP — see security notice above)
- fastapi + uvicorn (for the API)

See `requirements.txt` for pinned versions.
