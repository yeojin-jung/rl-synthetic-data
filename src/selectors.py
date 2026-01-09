import numpy as np
import scipy
import torch
import torch.nn.functional as F

# k is our budget

def select_random(total_count, k):
    """
    Baseline: randomly select k indices
    """
    return np.random.choice(total_count, k, replace=False)

def select_less(train_grads, val_grads, k):
    """
    LESS: select training points based on influence function
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_grads = train_grads.to(device)
    val_grads = val_grads.to(device)
    
    target_dir = torch.mean(val_grads, dim=0) # average over validation gradients
    scores = torch.matmul(train_grads, target_dir)

    return torch.topk(scores, k).indices.cpu().numpy()


def select_cluster_grad(train_grads, k, cluster_ratio=0.1, thres=0.5, num_iters=10):
    """
    Selects k samples by identifying sparse regions in the gradient space.
    
    Args:
        train_grads: Tensor of shape [N, dim]
        k_samples: Number of samples to eventually select.
        cluster_ratio: k-means 'k' as a percentage of total data (default 10%).
        sparse_threshold: The percentage of clusters to consider 'sparse' (e.g., bottom 50%).
    """
    n = train_grads.shape[0]
    num_clusters = max(1, int(cluster_ratio*n))

    grads_norm = F.normalize(train_grads, dim=1)

    # K-means
    indices = torch.randperm(n)[:num_clusters]
    centroids = grads_norm[indices]

    for _ in range(num_iters):
        sim = torch.matmul(grads_norm, centroids.T)
        labels = torch.argmax(sim, dim=1)
        new_centroids = torch.zeros_like(centroids)
        for i in range(num_clusters):
            mask = (labels == i)
            if mask.any():
                new_centroids[i] = F.normalize(grads_norm[mask].mean(dim=0), dim=0)
        centroids = new_centroids
    
    # Identify sparse clusters
    cluster_counts = torch.bincount(labels, minlength=num_clusters)
    sorted_cluster = torch.argsort(cluster_counts)

    num_sparse = int(num_clusters*thres)
    selected_clusters = sorted_cluster[:num_sparse]

    is_sparse = torch.isin(labels, selected_clusters)
    sparse_ids = torch.where(is_sparse)[0]

    if len(sparse_ids) >= k:
        perm = torch.randperm(len(sparse_ids))[:k]
        return sparse_ids[perm].cpu().numpy()
    else:
        print(f"Warning: Only found {len(sparse_ids)} sparse samples. Returning all.")
        return sparse_ids.cpu().numpy()


def get_g_vendi(grads, n):
    grads_norm = F.normalize(grads, dim=1)
    sim = (torch.matmul(grads_norm, grads_norm.T)/n).cpu().numpy()

    lambdas = scipy.linalg.eigvalsh(sim)
    p_ = lambdas[lambdas > 1e-10]
    ent = -np.sum(p_ * np.log(p_ + 1e-12))
    
    return np.exp(ent)


def select_g_vendi(train_grads, k):
    """
    G-Vendi: Greedy Diversity Selection.
    Selects points that maximize the Vendi Score (eigenvalue-based diversity).
    ! in a large scale scenario, use a faster approximation
    """
    n = train_grads.shape[0]
    selected_indices = []
    remaining_indices = list(range(n))

    # we take the greedy approach to select one point at each time 
    # that increases the G-Vendi score the most
    for _ in range(k):
        best_vendi = -1
        best_idx = -1

        for idx in remaining_indices:
            s = selected_indices +[idx]
            grads = train_grads[s]
            gv = get_g_vendi(grads, len(s))

            if gv > best_vendi:
                best_vendi = gv
                best_idx = idx
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    return np.array(selected_indices)