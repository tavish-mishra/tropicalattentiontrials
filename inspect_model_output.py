import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
import random

# ------- config -------
N = 8
P = 0.6
WEIGHT_LOW = 0.05
WEIGHT_HIGH = 1.0
SEED = 999
# ----------------------

random.seed(SEED)
np.random.seed(SEED)


def generate_er_graph(n, p, low, high, eps=0.05):
    adj_upper = np.random.binomial(1, p, size=(n, n))
    adj = np.triu(adj_upper, k=1)
    adj = adj + adj.T

    weights_upper = np.random.uniform(low, high, size=(n, n))
    weights = np.triu(weights_upper, k=1)
    weights = weights + weights.T

    M = np.random.uniform(1 - eps, 1 + eps)
    edge_weights = weights[adj == 1]
    if edge_weights.size > 0:
        weights = weights * (M / edge_weights.max())

    W = np.full((n, n), np.inf)
    W[adj == 1] = weights[adj == 1]
    np.fill_diagonal(W, 0.0)
    return W, adj


def print_matrix(name, M, fmt=".3f", inf_str="  inf"):
    n = M.shape[0]
    header = f"{'':>6}" + "".join(f"  {j:>6}" for j in range(n))
    print(f"\n{name}")
    print("-" * len(header))
    print(header)
    for i in range(n):
        row = f"  {i:>3} "
        for j in range(n):
            v = M[i, j]
            if np.isinf(v):
                row += f"  {inf_str:>6}"
            elif isinstance(v, (int, np.integer)):
                row += f"  {int(v):>6}"
            else:
                row += f"  {v:{fmt}}"
        print(row)


def print_pred_matrix(Pi, n):
    print("\nPredecessor matrix (Pi)  — -1 means 'no predecessor'")
    print("-" * (8 + 8 * n))
    header = f"{'':>6}" + "".join(f"  {j:>6}" for j in range(n))
    print(header)
    for i in range(n):
        row = f"  {i:>3} "
        for j in range(n):
            v = Pi[i, j]
            row += f"  {v:>6}"
        print(row)


def reconstruct_path(Pi, src, dst):
    if Pi[src, dst] == -9999 or src == dst:
        return [src] if src == dst else []
    path = [dst]
    while path[-1] != src:
        prev = Pi[src, path[-1]]
        if prev == -9999:
            return []
        path.append(prev)
    return list(reversed(path))


# --- generate ---
W, adj = generate_er_graph(N, P, WEIGHT_LOW, WEIGHT_HIGH)

# --- solve ---
graph = csr_matrix(W)
D, Pi = floyd_warshall(csgraph=graph, directed=False, return_predecessors=True)

# --- print ---
print(f"Graph: {N} nodes, edge probability={P}, seed={SEED}")
print(f"Edges: {int(adj.sum() // 2)}")

print_matrix("Adjacency / weight matrix (W)  — inf = no edge", W)
print_matrix("Shortest-path distance matrix (D)", D)
print_pred_matrix(Pi, N)

# --- show a few example paths ---
print("\nSample shortest paths:")
pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
for src, dst in pairs[:6]:
    path = reconstruct_path(Pi, src, dst)
    dist = D[src, dst]
    if path:
        path_str = " -> ".join(str(p) for p in path)
        print(f"  {src} -> {dst}:  [{path_str}]  (dist={dist:.3f})")
    else:
        print(f"  {src} -> {dst}:  unreachable")