import torch
import json
import os
import numpy as np
import torch.nn.functional as F
from methods.dsir import select_dsir
from methods.prismatic_cluster import ClusterManager

class DataSelector:
    def __init__(self, pool_grads_path, val_grads_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pool_grads = torch.load(pool_grads_path).to(self.device)
        self.val_grads = torch.load(val_grads_path).to(self.device)
        self.n_pool = self.pool_grads.shape[0]

    def select_random(self, k):
        """
        Baseline: randomly select k indices2
        """
        return np.random.choice(self.n_pool, k, replace=False)

    def select_less(self, k):
        """
        LESS: select training points based on influence function (Xia, 2024)
        """
        # TODO: We need to separate alignment from evaluation
        val_dir = torch.mean(self.val_grads, dim=0)
        scores = torch.matmul(self.pool_grads, val_dir)
        return torch.topk(scores, k).indices.cpu().numpy()

    def select_prismatic(self, k, sparsity=0.5, cluster_ratio=0.1, thres=0.5, num_iters=20, method="hard"):
        """
        Prismatic Synthesis:
        Selects k samples by identifying sparse regions in the gradient space.
        The original work selects few-shot synthetic generations from sparse clusters
        retrieved from k-means clustering on the training samples.
        We apply similar idea here to select from the training samples.
        # TODO: Clustering should also be done on a subset of training?
        Args:
            cluster_ratio: k-means 'k' as a percentage of total data (default 10%).
            sparse_threshold: The percentage of clusters to consider 'sparse' (e.g., bottom 50%).
        """
        cluster_labels, cluster_centroids = ClusterManager.cluster_kmeans(
            self.pool_grads, k=int(self.pool_grads.size(0) * cluster_ratio), num_iter=num_iters, use_tqdm=True)
        
        pool_grads_norm = F.normalize(self.pool_grads, dim=1)
        sim_with_centroids = torch.matmul(pool_grads_norm, cluster_centroids.T)  # (# samples, # centroids)
        cluster_labels = torch.argmax(sim_with_centroids, dim=-1)
        if method == "hard":
            small_clusters = ClusterManager.smallest_clusters(cluster_labels, sparsity)
            small_clusters = torch.tensor(small_clusters, device=cluster_labels.device)

            is_sparse = torch.isin(cluster_labels, small_clusters)
            sparse_ids = torch.where(is_sparse)[0]

            if len(sparse_ids) >= k:
                perm = torch.randperm(len(sparse_ids))[:k]
                return sparse_ids[perm].cpu().numpy()
            else:
                print(f"Warning: Only found {len(sparse_ids)} sparse samples. Returning all.")
                return sparse_ids.cpu().numpy()

        if method == "soft":
            # Sample with probability proportional to 1 / cluster_size
            labels_cpu = cluster_labels.detach().cpu().numpy()
            counts = np.bincount(labels_cpu, minlength=int(cluster_labels.max().item()) + 1)
            inv_counts = 1.0 / np.maximum(counts, 1)
            probs = inv_counts[labels_cpu]
            probs = probs / probs.sum()
            return np.random.choice(self.n_pool, k, replace=False, p=probs)

        raise ValueError(f"Unknown method: {method}. Use 'hard' or 'soft'.")

def run_selection_experiment(
    k=2000,
    pool_path="data/processed/candidate_pool.jsonl",
    val_path="data/processed/val_set.jsonl",
    pool_grads_path="data/features/pool_grads.pt",
    val_grads_path="data/features/val_grads.pt",
    out_dir="data/selected_indices",
):
    selector = DataSelector(pool_grads_path, val_grads_path)
    
    methods = {
        "random": selector.select_random,
        "less": selector.select_less,
        "prismatic": selector.select_prismatic,
        "prismatic_soft": lambda k: selector.select_prismatic(k, method="soft"),
        "dsir": lambda k: select_dsir(pool_path, val_path, k)
    }
    
    results_dir = out_dir
    os.makedirs(results_dir, exist_ok=True)
    
    for name, func in methods.items():
        print(f"Running {name}...")
        indices = func(k)
        np.save(f"{results_dir}/{name}_indices.npy", indices)
        
        # Evaluation helper: Check contamination rate
        # Load pool to check 'is_contaminated' flag
        with open(pool_path, "r") as f:
            pool = [json.loads(line) for line in f]
        
        selected_data = [pool[i] for i in indices]
        noise_count = sum(1 for d in selected_data if d['is_contaminated'])
        denom = max(len(indices), 1)
        print(f"Result: {name} selected {noise_count}/{len(indices)} contaminated samples ({(noise_count/denom)*100:.1f}%)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=2000)
    parser.add_argument("--pool-path", type=str, default="data/processed/candidate_pool.jsonl")
    parser.add_argument("--val-path", type=str, default="data/processed/val_set.jsonl")
    parser.add_argument("--pool-grads", type=str, default="data/features/pool_grads.pt")
    parser.add_argument("--val-grads", type=str, default="data/features/val_grads.pt")
    parser.add_argument("--out-dir", type=str, default="data/selected_indices")
    args = parser.parse_args()
    run_selection_experiment(
        k=args.k,
        pool_path=args.pool_path,
        val_path=args.val_path,
        pool_grads_path=args.pool_grads,
        val_grads_path=args.val_grads,
        out_dir=args.out_dir,
    )
