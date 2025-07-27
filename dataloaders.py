import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall, connected_components
import numpy as np
import bisect

from quickselect.hoare import nth_smallest, select

# -------------------------------------------------------------------
# Set-Based Datasets
# -------------------------------------------------------------------
#Given a set of numbers X and a target integer T, the problem is to decide if there exists a subset Y⊆X 
#such that the sum of the elements in Y equals T. An NP-complete problem
# ----- subset sum -----
def set_subset_sum_decision(x, T):
    possible_sums = {0}
    for val in x:
        possible_sums |= {s + val for s in possible_sums}
    return 1 if T in possible_sums else 0

class SubsetSumDecisionDataset(Dataset):
    """
    Dataset for subset-sum decision: given a set x and target T,
    label y = 1 if some subset of x sums to T, else 0.

    Each token is a 2-vector [value_i, T], so the model input_dim = 2.
    """
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: tuple[int,int] = (10,10),
        value_range: tuple[int,int] = (-2,2),
        target_range: tuple[int,int] = (1,10),
        noise_prob: float = 0.0,
        adversarial_range: tuple[int,int] = (10,30),
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        random.seed(seed)
        self.n_samples        = n_samples
        self.length_range     = length_range
        self.value_range      = value_range
        self.target_range     = target_range
        self.noise_prob       = noise_prob
        self.adversarial_range= adversarial_range
        self.data = []
        for _ in range(n_samples):
            n = random.randint(*self.length_range)
            x = [random.randint(*self.value_range) for _ in range(n)]
            
            T = random.randint(*self.target_range)
            x_sorted = sorted(x)
            
            y = set_subset_sum_decision(x_sorted, T)
            y_t = torch.tensor(y, dtype=torch.float)
            
            x_t_data_list = [[xi_val, T] for xi_val in x_sorted]
            x_t = torch.tensor(x_t_data_list, dtype=torch.float)
            
            if self.noise_prob > 0:
                for i in range(x_t.shape[0]):
                    if random.random() < self.noise_prob:
                        x_t[i, 0] += random.randint(*self.adversarial_range)
            
            self.data.append((x_t, y_t))
    def __len__(self): return self.n_samples
    def __getitem__(self, idx): return self.data[idx]

# Given a set of numbers X the goal is to find the maximum sum among all possible subsets Y⊆X. NP-hard
# ----- max subset -----
def set_max_subset_sum(x):
    possible_sums = {0}
    for val in x:
        possible_sums |= {s + val for s in possible_sums}
    return max(possible_sums)

class MaxSubsetSumDataset(Dataset):
    """
    Dataset for the maximum-subset-sum problem.

    Given a multiset x of integers, label y = max_{S⊆x} sum(S), 
    where the empty subset yields sum 0.

    Each example returns:
      x_t: Tensor of shape (n,1) — the sorted sequence of values
      y_t: Tensor of shape (1,)  — max s

    Args:
      n_samples         (int): number of examples to generate
      length_range ((int,int)): min/max sequence length
      value_range  ((int,int)): sampling range for each element in x
      noise_prob        (float): probability to perturb each x_i
      adversarial_range ((int,int)): magnitude of noise if applied
      seed               (int): RNG seed for reproducibility
      **kwargs                : catch-all for any extra params
    """
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: tuple[int,int] = (10,10),
        value_range: tuple[int,int] = (-2,2),
        noise_prob: float = 0.0,
        adversarial_range: tuple[int,int] = (10,30),
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        random.seed(seed)

        self.n_samples         = n_samples
        self.length_range      = length_range
        self.value_range       = value_range
        self.noise_prob        = noise_prob
        self.adversarial_range = adversarial_range

        self.data = []
        for _ in range(n_samples):
            n = random.randint(*self.length_range) # 1) sample sequence length and raw values
            x = [random.randint(*self.value_range) for _ in range(n)]
            if self.noise_prob > 0: # 2) optional adversarial noise
                x = [ xi + random.randint(*self.adversarial_range) if random.random() < self.noise_prob else xi for xi in x]
            x_sorted = sorted(x) # 3) sort for a consistent input order
            y_val = set_max_subset_sum(x_sorted) # 4) compute max subset sum via DP on possible sums
            x_t = torch.tensor(x_sorted, dtype=torch.float).unsqueeze(-1) # 5) build input tensor of shape (n,1)
            y_t = torch.tensor(y_val, dtype=torch.float)
            self.data.append((x_t, y_t))
    def __len__(self): return self.n_samples
    def __getitem__(self, idx): return self.data[idx]

#0/1 Knapsack Problem (Optimization Version). Given a set of items, each with a specific value and weight, 
#the goal is to select a subset of items whose total weight does not exceed a given capacity C, 
# such that the total value of the selected items is maximized. 
# Each item can either be included exactly once (1) or not at all (0). NP-hard
# ----- knapsack -----
def set_knapsack_01(items, capacity):
    dp = [0] * (capacity + 1)
    for val, wt in items:
        for w in range(capacity, wt - 1, -1):
            dp[w] = max(dp[w], dp[w - wt] + val)
    return max(dp)

class KnapsackDataset(Dataset):
    """
    Dataset for 0-1 knapsack decision/regression:
      Input:  sequence of items (v_i, w_i) plus a global capacity C
      Label:  maximum achievable value for that capacity
    
    We pack each token into a 3-vector [v_i, w_i, C], so your model's input_dim=3.
    """
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: tuple[int,int] = (10,10),
        value_range: tuple[int,int] = (1,10),
        weight_range: tuple[int,int] = (1,10),
        target_range: tuple[int,int] = (1,10),
        noise_prob: float = 0.0,
        adversarial_range: tuple[int,int] = (10,30),
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        random.seed(seed)

        self.n_samples         = n_samples
        self.length_range      = length_range
        self.value_range       = value_range
        self.weight_range      = weight_range
        self.target_range      = target_range
        self.noise_prob        = noise_prob
        self.adversarial_range = adversarial_range

        self.data = []
        for _ in range(n_samples):
            n = random.randint(*self.length_range)
            capacity = random.randint(*self.target_range)
            items = []
            for _ in range(n):
                v = random.randint(*self.value_range)
                w = random.randint(*self.weight_range)
                items.append((v, w))
            
            items_sorted = sorted(items, key=lambda x_item: x_item[0])
            y_val = set_knapsack_01(items_sorted, capacity)
            
            x_t_list_of_features = [[v_i, w_i, capacity] for v_i, w_i in items_sorted]
            x_t = torch.tensor(x_t_list_of_features, dtype=torch.float)
            
            y_t = torch.tensor(y_val, dtype=torch.float)
            
            if self.noise_prob > 0:
                for i in range(x_t.shape[0]):
                    if random.random() < self.noise_prob:
                        x_t[i, 0] += random.randint(*self.adversarial_range)
                        x_t[i, 1] += random.randint(*self.adversarial_range)
                        x_t[i, 1] = torch.clamp(x_t[i, 1], min=1.0)
            
            self.data.append((x_t, y_t))
    def __len__(self): return self.n_samples
    def __getitem__(self, idx): return self.data[idx]

#An optimization problem where you aim to maximize the total value of items placed into a knapsack with a limited capacity, 
#but unlike the 0/1 version, you are allowed to take fractions of items. polynomial time using a greedy approach.
# ----- fractional knapsack -----
def set_fractional_knapsack(items, capacity):
    sorted_items = sorted(items, key=lambda x: x[0] / x[1], reverse=True)
    remaining, total = capacity, 0.0
    for val, wt in sorted_items:
        if remaining <= 0:
            break
        if wt <= remaining:
            total += val
            remaining -= wt
        else:
            frac = remaining / wt
            total += val * frac
            break
    return total

class FractionalKnapsackDataset(Dataset):
    """
    Dataset for the Fractional Knapsack problem (maximize total value
    under a capacity constraint).

    Each token is a 3-vector [value_i, weight_i, capacity], so use
    input_dim=3 in your model.

    Returns:
      x_t: Tensor of shape (n, 3)
      y_t: Tensor of shape 1
    """
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: tuple[int,int] = (10,10),
        value_range: tuple[int,int] = (10,20),
        weight_range: tuple[int,int] = (1,5),
        target_range: tuple[int,int] = (1,5),
        noise_prob: float = 0.0,
        adversarial_range: tuple[int,int] = (10,30),
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        random.seed(seed)

        self.n_samples         = n_samples
        self.length_range      = length_range
        self.value_range       = value_range
        self.weight_range      = weight_range
        self.target_range      = target_range
        self.noise_prob        = noise_prob
        self.adversarial_range = adversarial_range

        self.data = []
        for _ in range(n_samples):
            n = random.randint(*self.length_range)
            capacity = random.randint(*self.target_range)
            items = []
            for _ in range(n):
                v = random.randint(*self.value_range)
                w = random.randint(*self.weight_range)
                items.append((v, w))
            
            items_sorted = sorted(items, key=lambda x_item: x_item[0])
            y_val = set_fractional_knapsack(items_sorted, capacity)
            
            x_t_list_of_features = [[v_i, w_i, capacity] for v_i, w_i in items_sorted]
            x_t = torch.tensor(x_t_list_of_features, dtype=torch.float)
            
            y_t = torch.tensor(y_val, dtype=torch.float)
            
            if self.noise_prob > 0:
                for i in range(x_t.shape[0]):
                    if random.random() < self.noise_prob:
                        x_t[i, 0] += random.randint(*self.adversarial_range)
                        x_t[i, 1] += random.randint(*self.adversarial_range)
                        x_t[i, 1] = torch.clamp(x_t[i, 1], min=1.0)
            
            self.data.append((x_t, y_t))
    def __len__(self): return self.n_samples
    def __getitem__(self, idx): return self.data[idx]

# 0/1 Minimum Coin Change Problem, NP-hard.
# ----- min coin -----
def set_min_coin_change(coins, T):
    dp = [math.inf] * (T + 1)
    dp[0] = 0
    for c in coins:
        for w in range(T, c - 1, -1):
            dp[w] = min(dp[w], dp[w - c] + 1)
    return 0 if dp[T] == math.inf else dp[T]

class MinCoinChangeDataset(Dataset):
    """
    Dataset for the Minimum Coin Change problem:
      Input:  sequence of coin values + global target T
      Label:  minimum number of coins summing to T (or 0 if impossible)

    Each token is a 2-vector [coin_i, T], so set your model's input_dim=2.
    """
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: tuple[int,int] = (10,10),
        value_range: tuple[int,int]   = (1,5),
        target_range: tuple[int,int] = (1,10),
        noise_prob: float = 0.0,
        adversarial_range: tuple[int,int] = (5,20),
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        random.seed(seed)

        self.n_samples         = n_samples
        self.length_range      = length_range
        self.value_range        = value_range
        self.target_range      = target_range
        self.noise_prob        = noise_prob
        self.adversarial_range = adversarial_range

        self.data = []
        for _ in range(n_samples):
            n = random.randint(*self.length_range)
            coins = [random.randint(*self.value_range) for _ in range(n)]
            
            T = random.randint(*self.target_range)
            coins_sorted = sorted(coins)
            
            y_val = set_min_coin_change(coins_sorted, T)
            
            x_t_list_of_pairs = [[c_val, T] for c_val in coins_sorted]
            x_t = torch.tensor(x_t_list_of_pairs, dtype=torch.float)
            
            y_t = torch.tensor(y_val, dtype=torch.float)
            
            if self.noise_prob > 0:
                for i in range(x_t.shape[0]):
                    if random.random() < self.noise_prob:
                        x_t[i, 0] += random.randint(*self.adversarial_range)
                        x_t[i, 0] = torch.clamp(x_t[i, 0], min=1.0)
            
            self.data.append((x_t, y_t))
    def __len__(self): return self.n_samples
    def __getitem__(self, idx): return self.data[idx]
    

def get_true_kth_smallest(unsorted_list, k_val):
    return sorted(unsorted_list)[k_val - 1]

class QuickselectDataset(Dataset):
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: tuple[int,int] = (10,10),     # n = number of elements in xs
        value_range: tuple[int,int]  = (1,10),      # Range for values in xs
        noise_prob: float = 0.0,                    # Probability of adding noise
        adversarial_range: tuple[int,int] = (1,5),  # Range of noise values
        classification: bool = False,               # True for classification, False for regression
        seed: int = 42,
        **kwargs  # Catches other arguments like k if passed from a config
    ):
        super().__init__()
        random.seed(seed)

        self.n_samples         = n_samples
        self.length_range      = length_range
        self.value_range       = value_range
        self.noise_prob        = noise_prob
        self.adversarial_range = adversarial_range
        self.classification    = classification
        self.k_default         = kwargs.get('k', 2) # Default k to 2 if not in kwargs

        self.data = [] # List to store (input_tensor, output_tensor) tuples
        for _ in range(n_samples):
            # 1) Draw sequence length n and initial values for xs
            n = random.randint(*self.length_range)
            if n == 0: # Cannot select k-th element from an empty list
                continue
            # This list will be the basis for the model's input (unsorted)
            xs_original_unsorted = [random.randint(*self.value_range) for _ in range(n)]


            # 3) Define k.
            k = self.k_default # Use k passed to init or its default
            # If k should be random per sample:
            #k = random.randint(1, k)

            # 4) Determine the TRUE k-th smallest value (y_val) from xs_original_unsorted.
            y_val = nth_smallest(xs_original_unsorted.copy(), k-1)

            # 5) Build the input tensor x_t using the UNSORTED xs_original_unsorted.
            # Each token is [value_from_unsorted_list, k].
            # Shape: (n, 2)
            x_t = torch.tensor([[xi_val, float(k)] for xi_val in xs_original_unsorted], dtype=torch.float)
            if self.noise_prob > 0:
                processed_rows = [
                    (row_tensor + random.randint(*self.adversarial_range)) 
                    if random.random() < self.noise_prob 
                    else row_tensor 
                    for row_tensor in x_t
                ]
                if processed_rows:
                    x_t = torch.stack(processed_rows)
                
            #In case of using index as position input_dim is 3.
            # input_tokens = []
            # for i, xi in enumerate(xs_original_unsorted):
            #    normalized_idx = i / (n - 1) if n > 1 else 0.0
            #    input_tokens.append([xi, float(k), normalized_idx])
            # x_t = torch.tensor(input_tokens, dtype=torch.float)

            # 6) Build the output tensor y_t.
            if not self.classification:
                # Regression target: the actual k-th smallest value.
                # Shape: scalar tensor ()
                y_t = torch.tensor(y_val, dtype=torch.float)
            else:
                # Classification mask: identify ALL positions of the k-th smallest value
                # in the *unsorted* input list `xs_original_unsorted`.
                # Shape: (n,)
                mask = [0.0] * n # Use float for consistency
                for i in range(n):
                    if xs_original_unsorted[i] == y_val:
                        mask[i] = 1.0 # Mark ALL occurrences
                y_t = torch.tensor(mask, dtype=torch.float)

            self.data.append((x_t, y_t))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]


#NP-hard
# ----- balanced partition -----
def set_balanced_partition(x):
    """
    Compute the minimum difference between two subset sums of x
    via DP on subset sums.
    """
    possible_sums = {0}
    for val in x:
        possible_sums |= {s + val for s in possible_sums}
    total = sum(x)
    # Find subset sum s minimizing |total - 2*s|
    return min(abs(total - 2 * s) for s in possible_sums)

class BalancedPartitionDataset(Dataset):
    """
    Dataset for the Balanced Partition problem:
      - Input: sequence of integers x_i
      - Label: minimum absolute difference between the sum of two subsets of x
    Each sample returns:
      x_t: Tensor of shape (n,1) holding each x_i
      y_t: Tensor the scalar partition-difference

    Sorting is used to canonicalize input order, but the problem is permutation-invariant.
    """
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: tuple[int,int] = (10,10),
        value_range: tuple[int,int] = (1,10),
        noise_prob: float = 0.0,
        adversarial_range: tuple[int,int] = (10,30),
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        random.seed(seed)

        self.n_samples = n_samples
        self.length_range = length_range
        self.value_range = value_range
        self.noise_prob = noise_prob
        self.adversarial_range = adversarial_range

        self.data = []
        for _ in range(n_samples):
            n = random.randint(*self.length_range)
            x = [random.randint(*self.value_range) for _ in range(n)]
            
            x_sorted = sorted(x)
            y_val = set_balanced_partition(x_sorted)
            
            x_t = torch.tensor(x_sorted, dtype=torch.float).unsqueeze(-1)
            y_t = torch.tensor(y_val, dtype=torch.float) 
            
            if self.noise_prob > 0:
                for i in range(x_t.shape[0]):
                    if random.random() < self.noise_prob:
                        x_t[i, 0] += random.randint(*self.adversarial_range)
            
            self.data.append((x_t, y_t))
            
    def __len__(self): return self.n_samples
    def __getitem__(self, idx): return self.data[idx]
    
# ----- bin packing -----
def first_fit_decreasing(sizes, capacity):
    """
    Greedy First-Fit Decreasing heuristic for bin packing.
    Returns approximate minimum number of bins.
    """
    bins = []
    for sz in sorted(sizes, reverse=True):
        placed = False
        for i in range(len(bins)):
            if bins[i] + sz <= capacity:
                bins[i] += sz
                placed = True
                break
        if not placed:
            bins.append(sz)
    return len(bins)

class BinPackingDataset(Dataset):
    """
    Dataset for the Bin Packing problem:
      - Input: sequence of item sizes and a global bin capacity
      - Label: approximate minimum number of bins (via First-Fit Decreasing)

    Each token is a 2-vector [size_i, capacity], so the model's input_dim=2.
    Outputs:
      x_t: Tensor of shape (n,2)
      y_t: Tensor of shape (n,) repeating the scalar bin-count
    """
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: tuple[int,int] = (10,10),
        value_range: tuple[int,int] = (1,10),
        target_range: tuple[int,int] = (10,30),
        noise_prob: float = 0.0,
        adversarial_range: tuple[int,int] = (10,30),
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        random.seed(seed)

        self.n_samples         = n_samples
        self.length_range      = length_range
        self.value_range       = value_range
        self.target_range      = target_range
        self.noise_prob        = noise_prob
        self.adversarial_range = adversarial_range

        self.data = []
        for _ in range(n_samples):
            n = random.randint(*self.length_range) # 1) sample number of items and their sizes
            capacity = random.randint(*self.target_range)
            sizes = [random.randint(*self.value_range) for _ in range(n)]
            if self.noise_prob > 0: # 2) optional adversarial noise
                sizes = [sz + random.randint(*self.adversarial_range) if random.random() < self.noise_prob else sz for sz in sizes]
            sizes = [min(max(1, sz), capacity) for sz in sizes]
            sizes_sorted = sorted(sizes) # 3) sort sizes for consistency (optional)
            y_val = first_fit_decreasing(sizes_sorted, capacity) # 4) compute approximate bins via First-Fit Decreasing
            x_t = torch.tensor([[sz, capacity] for sz in sizes_sorted], dtype=torch.float) # 5) build input tensor: each row [size, capacity]
            y_t = torch.tensor(y_val, dtype=torch.float) # 6) repeat the bin-count label per token
            self.data.append((x_t, y_t))
    def __len__(self): return self.n_samples
    def __getitem__(self, idx): return self.data[idx]



# ----- Longest Increasing Subsequence (LIS) Length ----- (regression)
def calculate_lis_length(nums):
    if not nums:
        return 0
    tails = []
    for num in nums:
        if not tails or num > tails[-1]:
            tails.append(num)
        else:
            tails[bisect.bisect_left(tails, num)] = num
    return len(tails)

class LISDataset(Dataset):
    """
    Dataset for the Longest Increasing Subsequence (LIS) Length problem.
      - Input: sequence of numbers x_i
      - Label: the length of the LIS of x

    Each token in x_t is a 2d-vector [value_i, positional index] by default.

    Outputs:
      x_t: Tensor of shape (n, input_dim) - input sequence
      y_t: Tensor of shape ()            - scalar LIS length (regression target)
    """
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: tuple[int,int] = (8,8),
        value_range: tuple[int,int] = (1,10),
        noise_prob: float = 0.0,
        adversarial_range: tuple[int,int] = (10,30),
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        random.seed(seed)

        self.n_samples         = n_samples
        self.length_range      = length_range
        self.value_range       = value_range
        self.noise_prob        = noise_prob
        self.adversarial_range = adversarial_range

        self.data = []
        for _ in range(n_samples):
            n = random.randint(*self.length_range)
            if n == 0:
                continue
            sequence = [random.randint(*self.value_range) for _ in range(n)]
            if self.noise_prob > 0:
                noisy_sequence_temp = []
                for val in sequence:
                    if random.random() < self.noise_prob:
                        noisy_sequence_temp.append(val + random.randint(*self.adversarial_range))
                    else:
                        noisy_sequence_temp.append(val)
                sequence = noisy_sequence_temp
            y_val = calculate_lis_length(sequence)
            # Shape: (n, 2)
            x_t_list = [[float(val)] for val in sequence]
            temp_xt_list = []
            for i, val in enumerate(sequence):
                normalized_idx = np.exp(i / (n - 1)) if n > 1 else 0.0
                temp_xt_list.append([float(val), normalized_idx])
            x_t_list = temp_xt_list

            x_t = torch.tensor(x_t_list, dtype=torch.float)
            y_t = torch.tensor(float(y_val), dtype=torch.float)

            self.data.append((x_t, y_t))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]
    
# ----- convex hull -----
def compute_convex_hull(points):
    pts = sorted(points)
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and ((lower[-1][0] - lower[-2][0]) * (p[1] - lower[-2][1]) -
                                     (lower[-1][1] - lower[-2][1]) * (p[0] - lower[-2][0])) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and ((upper[-1][0] - upper[-2][0]) * (p[1] - upper[-2][1]) -
                                     (upper[-1][1] - upper[-2][1]) * (p[0] - upper[-2][0])) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

class ConvexHullDataset(Dataset):
    """
    Dataset for the Convex Hull classification problem.

    Each example is a random set of 2D points; the label is 1 if the point
    lies on the set's convex hull, else 0.

    Args:
      n_samples        (int): number of examples
      length_range  ((int,int)): min/max number of points
      value_range((int,int)): min/max for x and y coordinates
      noise_prob      (float): chance to jitter each coordinate
      adversarial_range((int,int)): jitter range if noise applies
      seed             (int): RNG seed
      **kwargs
    """
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: tuple[int,int] = (10, 10),
        value_range: tuple[int,int] = (0, 10),
        noise_prob: float = 0.0,
        adversarial_range: tuple[int,int] = (1, 3),
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        random.seed(seed)

        self.n_samples        = n_samples
        self.length_range     = length_range
        self.value_range      = value_range
        self.noise_prob       = noise_prob
        self.adversarial_range= adversarial_range

        self.data = []
        for _ in range(n_samples):
            n = random.randint(*self.length_range)
            pts = []
            for _ in range(n):
                x = random.randint(*self.value_range)
                y = random.randint(*self.value_range)
                pts.append((x, y))
            
            hull_pts = set(compute_convex_hull(pts))
            mask = [1 if p in hull_pts else 0 for p in pts] #label each point: 1 if on hull, else 0
            
            x_t = torch.tensor(pts, dtype=torch.float)
            y_t = torch.tensor(mask, dtype=torch.float)

            if self.noise_prob > 0:
                for i in range(x_t.shape[0]):
                    if random.random() < self.noise_prob:
                        x_t[i, 0] += random.randint(*self.adversarial_range)
                        x_t[i, 1] += random.randint(*self.adversarial_range)
                            
            self.data.append((x_t, y_t))
    def __len__(self): return self.n_samples
    def __getitem__(self, idx): return self.data[idx]
    
# ----- three sum decision -----
def three_sum_decision(xs, target=0):
    """
    Checks if any three *distinct* elements (by index) in list 'xs' sum to 'target'.
    Args:
        xs (list): A list of numbers.
        target (int/float): The target sum.
    Returns:
        int: 1 if three distinct elements sum to target, 0 otherwise.
    """
    n = len(xs)
    if n < 3:
        return 0 # Cannot choose 3 distinct elements
    xs_sorted = sorted(xs) # O(n log n) - necessary for two-pointer approach
    # O(n^2) part
    for i in range(n - 2):
        if i > 0 and xs_sorted[i] == xs_sorted[i-1]:
             continue
        left, right = i + 1, n - 1 # Pointers for the remaining part
        needed_sum = target - xs_sorted[i]
        while left < right:
            current_two_sum = xs_sorted[left] + xs_sorted[right]
            if current_two_sum == needed_sum:
                # Found a triplet (xs_sorted[i], xs_sorted[left], xs_sorted[right])
                # Indices i, left, right are guaranteed distinct by construction
                return 1
            elif current_two_sum < needed_sum:
                left += 1 # Need a larger sum, move left pointer right
            else: # current_two_sum > needed_sum
                right -= 1 # Need a smaller sum, move right pointer left
    # If loops complete without finding a triplet
    return 0

class ThreeSumDecisionDataset(Dataset):
    """
    Dataset for the 3-SUM decision problem: given a list x and a target T,
    label = 1 if any three distinct elements of x sum to T, else 0.
    Uses the canonical T=0 by default.

    Each token is a 2-vector [value_i, T], so your model should use input_dim=2.

    Args:
      n_samples         (int): number of examples
      length_range ((int,int)): min/max list length n
      value_range  ((int,int)): sampling range for each x_i (can be negative)
      target_range ((int,int)): sampling range for T (default is fixed T=0)
      noise_prob        (float): probability to perturb each x_i
      adversarial_range ((int,int)): magnitude of noise if applied
      seed               (int): RNG seed
      **kwargs: catch-all for unused params
    """
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: tuple[int,int] = (8, 8), # List length n
        value_range: tuple[int,int]  = (-10, 10), # Value range for elements
        target_range: tuple[int,int] = (0, 0),   # Target T range (fixed T=0 default)
        noise_prob: float = 0.0,
        adversarial_range: tuple[int,int] = (100, 200), # Noise magnitude
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        random.seed(seed) # Reproducibility

        self.n_samples        = n_samples
        self.length_range     = length_range
        self.value_range      = value_range
        self.target_range     = target_range
        self.noise_prob       = noise_prob
        self.adversarial_range= adversarial_range

        self.data = []
        for _ in range(n_samples):
            n = random.randint(*self.length_range)
            xs = [random.randint(*self.value_range) for _ in range(n)]
            
            T = random.randint(*self.target_range)
            y_val = three_sum_decision(xs, T)
            
            xs_sorted = sorted(xs)
            
            x_t_list_of_pairs = [[val, T] for val in xs_sorted]
            x_t = torch.tensor(x_t_list_of_pairs, dtype=torch.float)
            
            y_t = torch.tensor(y_val, dtype=torch.long)
            
            if self.noise_prob > 0:
                for i in range(x_t.shape[0]):
                    if random.random() < self.noise_prob:
                        x_t[i, 0] += random.randint(*self.adversarial_range)
            
            self.data.append((x_t, y_t))

    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        return self.data[idx]

# ----- Floyd Warshall dataset -----
class FloydWarshallDataset(Dataset):
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: tuple[int,int] = (10, 10), # n = graph size
        value_range: tuple[int,int] = (1, 10),   # Edge weights W_ij
        noise_prob: float = 0.0,
        adversarial_range: tuple[int,int] = (10, 50), # Noise for W_ij
        classification: bool = False,
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        self.n_samples         = n_samples
        self.length_range      = length_range # n range
        self.value_range       = value_range
        self.noise_prob        = noise_prob
        self.adversarial_range = adversarial_range
        self.classification    = classification

        self.data = []
        for _ in range(n_samples):
            n = random.randint(*self.length_range)
            if n <= 0: continue

            W = np.random.randint(
                self.value_range[0],
                self.value_range[1] + 1,
                size=(n, n)
            ).astype(float)
            W = np.maximum(0, W)
            np.fill_diagonal(W, 0.0)

            graph = csr_matrix(W)
            D = floyd_warshall(csgraph=graph, directed=False, return_predecessors=False)

            large_val_for_inf = n * self.value_range[1] + 1 
            D[np.isinf(D)] = large_val_for_inf
            D_processed = D.astype(int) if self.classification else D
            
            flat_D = D_processed.flatten()

            if self.classification:
                y_t = torch.tensor(flat_D, dtype=torch.long)
            else:
                y_t = torch.tensor(flat_D, dtype=torch.float)

            flat_W = W.flatten()
            features = [torch.tensor(flat_W, dtype=torch.float).unsqueeze(-1)]
            indices = np.arange(n * n)
            norm_i = (indices // n) / (n - 1) if n > 1 else np.zeros(n * n)
            norm_j = (indices % n) / (n - 1) if n > 1 else np.zeros(n * n)
            features.append(torch.tensor(norm_i, dtype=torch.float).unsqueeze(-1))
            features.append(torch.tensor(norm_j, dtype=torch.float).unsqueeze(-1))
            x_t = torch.cat(features, dim=-1)

            if self.noise_prob > 0:
                for k_idx in range(n * n):
                    original_row = k_idx // n
                    original_col = k_idx % n
                    if original_row != original_col:
                        if random.random() < self.noise_prob:
                            jitter_val = random.randint(
                                self.adversarial_range[0],
                                self.adversarial_range[1] + 1
                            )
                            x_t[k_idx, 0] += jitter_val
                            x_t[k_idx, 0] = torch.clamp(x_t[k_idx, 0], min=0.0)
            
            self.data.append((x_t, y_t))

    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        return self.data[idx]

# ----- strongly connected components classification -----
class SCCDataset(Dataset):
    """
    Dataset for graph-connectivity classification.

    Each example is a random directed graph on n nodes (n ∈ size_range).
    We then (optionally) flip edges with probability noise_prob,
    treat the graph as undirected, and compute connected components.
    Every pair (i,j) gets a label 1.0 if they lie in the same component, else 0.0.

    Args:
      n_samples   (int): number of graphs to generate
      size_range (int,int): min/max number of nodes n
      edge_prob   (float): probability of including each directed edge
      noise_prob  (float): probability to flip any edge (A[i,j] ↔ 1−A[i,j])
      seed         (int): RNG seed for reproducibility
      **kwargs           : catch-all for unused params (e.g., target_range)
    """
    def __init__(
        self,
        n_samples: int = 1000,
        length_range: tuple[int,int] = (10, 10),
        value_range: float = 0.2,
        noise_prob: float = 0.0,
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        random.seed(seed)
        np.random.seed(seed)

        self.n_samples  = n_samples
        self.length_range = length_range
        self.edge_prob  = value_range
        self.noise_prob = noise_prob

        self.data = []
        for _ in range(n_samples):
            n = random.randint(*self.length_range)
            A = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(n):
                    if i != j and random.random() < self.edge_prob:
                        A[i, j] = 1.0
            
            graph = csr_matrix(A)
            _, labels = connected_components(csgraph=graph, directed=True, 
                                             return_labels=True, connection='strong')
            same_cc = (labels[:, None] == labels[None, :]).astype(float)
            
            flat_A = A.flatten()
            features = [torch.tensor(flat_A, dtype=torch.float).unsqueeze(-1)]
            
            indices = np.arange(n * n)
            norm_i = (indices // n) / (n - 1) if n > 1 else np.zeros(n * n)
            norm_j = (indices % n) / (n - 1) if n > 1 else np.zeros(n * n)
            
            features.append(torch.tensor(norm_i, dtype=torch.float).unsqueeze(-1))
            features.append(torch.tensor(norm_j, dtype=torch.float).unsqueeze(-1))
            x_t = torch.cat(features, dim=-1)

            if self.noise_prob > 0:
                for k_idx in range(n * n):
                    original_row_idx = k_idx // n
                    original_col_idx = k_idx % n
                    if original_row_idx != original_col_idx:
                        if random.random() < self.noise_prob:
                            x_t[k_idx, 0] = 1.0 - x_t[k_idx, 0]
                                
            #y_t = torch.tensor(flat_same_cc, dtype=torch.float) # Shape (n*n,)
            #x_t = torch.tensor(A.flatten(), dtype=torch.float).unsqueeze(-1)    # shape (n*n,1)
            y_t = torch.tensor(same_cc.flatten(), dtype=torch.float)
            self.data.append((x_t, y_t))
    def __len__(self): return self.n_samples
    def __getitem__(self, idx): return self.data[idx]
