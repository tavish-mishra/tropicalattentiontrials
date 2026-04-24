import numpy as np
import torch

from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset

class ShortestPathDataset(Dataset):
    def __init__(
        self, 
        num_graphs,
        num_nodes,
        edge_weight_low,
        edge_weight_high,
        graph_type='er',
        use_integer_weight=False,
    ):
        
        self.data = []
        
        if graph_type == 'er':
            curr_num_graphs = 0
            while curr_num_graphs < num_graphs:
                p = np.random.uniform(0.3, 0.9)
                X, Y = self._generate_er_graph_data(
                    n=num_nodes,
                    p=p,
                    edge_weight_low=edge_weight_low,
                    edge_weight_high=edge_weight_high,
                    use_integer_weight=use_integer_weight
                )
                if not torch.any(torch.isinf(Y)):
                    self.data.append((X, Y))
                    curr_num_graphs += 1
                    
        elif graph_type == 'tree':
            for _ in range(num_graphs):
                X, Y = self._generate_random_tree_data(
                    n=num_nodes,
                    edge_weight_low=edge_weight_low,
                    edge_weight_high=edge_weight_high,
                    use_integer_weight=use_integer_weight,
                )
                self.data.append((X, Y))

        elif graph_type == 'line':
            for _ in range(num_graphs):
                X, Y = self._generate_line_graph_data(
                    n=num_nodes,
                    edge_weight_low=edge_weight_low,
                    edge_weight_high=edge_weight_high,
                    use_integer_weight=use_integer_weight,
                )
                self.data.append((X, Y))

        elif graph_type == 'mixed':
            curr_num_graphs = 0
            while curr_num_graphs < num_graphs // 2:
                p = np.random.uniform(0.3, 0.9)
                X, Y = self._generate_er_graph_data(
                    n=num_nodes,
                    p=p,
                    edge_weight_low=edge_weight_low,
                    edge_weight_high=edge_weight_high,
                    use_integer_weight=use_integer_weight
                )
                if not torch.any(torch.isinf(Y)):
                    self.data.append((X, Y))
                    curr_num_graphs += 1

            for _ in range(num_graphs // 4):
                X, Y = self._generate_random_tree_data(
                    n=num_nodes,
                    edge_weight_low=edge_weight_low,
                    edge_weight_high=edge_weight_high,
                    use_integer_weight=use_integer_weight,
                )
                self.data.append((X, Y))

            for _ in range(num_graphs // 4):
                X, Y = self._generate_line_graph_data(
                    n=num_nodes,
                    edge_weight_low=edge_weight_low,
                    edge_weight_high=edge_weight_high,
                    use_integer_weight=use_integer_weight,
                )
                self.data.append((X, Y))
                     
        self.num_samples = len(self.data)

    def _generate_er_graph_data(
        self,
        n, 
        p, 
        edge_weight_low, 
        edge_weight_high, 
        use_integer_weight,
    ):
        
        m = int(n*(n-1)/2)
        edge_mask = np.random.binomial(1, p, size=m)
        if use_integer_weight:
            edge_weight = np.random.randint(
                low=edge_weight_low, high=edge_weight_high+1, size=m
            )
        else:
            edge_weight = np.random.uniform(
                low=edge_weight_low, high=edge_weight_high, size=m
            )
        edge_wieght = edge_weight * edge_mask
        X = np.zeros((n, n))
        X[np.triu_indices(n, 1)] = edge_wieght
        X = X + X.T
        graph = csr_matrix(X)
        Y = floyd_warshall(csgraph=graph, directed=False)
        return torch.tensor(X, dtype=torch.float), torch.tensor(Y, dtype=torch.float)

    def _generate_random_tree_data(
        self,
        n,
        edge_weight_low, 
        edge_weight_high, 
        use_integer_weight,
    ):
        
        prufer = np.random.randint(0, n, size=n-2)
        degree = np.ones(n, dtype=int)
        for x in prufer:
            degree[x] += 1
        adj = np.zeros((n, n), dtype=int)
        leaf_set = set(np.where(degree == 1)[0])
        for x in prufer:
            leaf = min(leaf_set)
            leaf_set.remove(leaf)
            adj[leaf, x] = 1
            adj[x, leaf] = 1
            degree[leaf] -= 1
            degree[x] -= 1
            if degree[x] == 1:
                leaf_set.add(x)
        remaining = list(leaf_set)
        a, b = remaining[0], remaining[1]
        adj[a, b] = 1
        edge_mask = adj[np.triu_indices(n, 1)]
        if use_integer_weight:
            edge_weight = np.random.randint(
                low=edge_weight_low, high=edge_weight_high+1, size=len(edge_mask)
            )
        else:
            edge_weight = np.random.uniform(
                low=edge_weight_low, high=edge_weight_high, size=len(edge_mask)
            )
        edge_wieght = edge_weight * edge_mask
        X = np.zeros((n, n))
        X[np.triu_indices(n, 1)] = edge_wieght
        X = X + X.T
        graph = csr_matrix(X)
        Y = floyd_warshall(csgraph=graph, directed=False)
        return torch.tensor(X, dtype=torch.float), torch.tensor(Y, dtype=torch.float)

    def _generate_line_graph_data(
        self,
        n, 
        edge_weight_low, 
        edge_weight_high, 
        use_integer_weight,
    ):
        def reverse_cumsum(x):
            return x +  x.sum() - torch.cumsum(x, dim=0)
            
        if use_integer_weight:
            edge_weight = torch.randint(
                low=edge_weight_low, high=edge_weight_high+1, size=(n-1,)
            ).float()
        else:
            edge_weight = low + (high - low)*torch.rand(n-1)
        perm = torch.randperm(n)
        X = torch.diag(edge_weight, 1) + torch.diag(edge_weight, -1)
        Y = torch.zeros(n,n)
        for i in range(n):
            Y[i,:i] = reverse_cumsum(edge_weight[:i])
            Y[i,(i+1):] = torch.cumsum(edge_weight[i:], dim=0)
        X = X[:,perm]
        X = X[perm,:]
        Y = Y[:,perm]
        Y = Y[perm,:]
        return X, Y

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

    