# Tropical-Attention
Official repository for the paper "Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms." 

It contains all code, data-set generators, and pretrained checkpoints.
---

### 1. Repository Contents

| Path | Purpose |
|------|---------|
| `dataloaders.py` | Defines eleven synthetic combinatorial data sets (Subset-Sum, Knapsack variants, Floyd–Warshall, etc.). |
| `models.py` | Three transformer variants: Vanilla, Adaptive-Softmax, and Tropical (ours). |
| `experiment.py` | Entry point for training and evaluation. Reads a parameter row from a CSV, sets up data, model, and training loop. |
| `jobs_to_do_train.csv` | Hyper-parameter grid used in the paper (one row per run). |
| `jobs_to_do_evaluate.csv` | Hyper-parameter grid used in the paper for testing models on in-distribution test datasets (one row per run). |
| `models/` | Pretrained model checkpoints (`*.pth`) used for the tables and figures. |

---

### 2. Software Requirements

```bash
pip install -r requirements.txt
```

---

### 3. Training a model from scratch
```bash
python experiment.py --job_file jobs_to_do_train --job_id 0 
```
`--job_id` selects the row in `jobs_to_do_train.csv`.
The script logs training progress to outputs/<timestamp>/.

---


### 4. Evaluating a pretrained checkpoint
```bash
python experiment.py --job_file jobs_to_do_evaluate --job_id 0 
```
If a matching checkpoint exists under `models/`, the script loads it,
runs the specified test configuration, and prints the metric summary. The logic in the experiment file is admittedly lacking and will take the first model found if there are multiple of the same configuration.

