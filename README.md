# Tropical-Attention
Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms.
---
We experiment on 11 combinatorial tasks:

1. **Floyd–Warshall**
2. **Quickselect**
3. **3SUM (Decision)**
4. **Balanced Partition**
5. **Convex Hull**
6. **Subset Sum (Decision)**
7. **0/1 Knapsack**
8. **Fractional Knapsack**
9. **Strongly Connected Components (SCC)**
10. **Bin Packing**
11. **Min Coin Change**

---

### 1. Training a model from scratch
```bash
python experiment.py --job_file jobs_to_do_train --job_id 0 
```
`--job_id` selects the row in `jobs_to_do_train.csv`.
The script logs training progress to outputs/<timestamp>/.

---


### 2. Evaluating a pretrained checkpoint
```bash
python experiment.py --job_file jobs_to_do_evaluate --job_id 0 
```


### 3. Minimal usage of the Tropical attention kernel

To minimally use the Tropical attention kernel, use `TropicalAttention.py` (a self-contained reference implementation you can copy or import into your project).

---

### 4. Citation

If you use this repository or Tropical Attention in your research, please cite:

```bibtex
@article{hashemi2025tropical,
  title={Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms},
  author={Hashemi, Baran and Pasque, Kurt and Teska, Chris and Yoshida, Ruriko},
  journal={arXiv e-prints},
  pages={arXiv--2505},
  year={2025}
}
```
