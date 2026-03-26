import json
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F
import scipy
import numpy as np


def get_g_vendi(grads, n):
    """
    Eigenvalue-based diversity score.
    Note: eigvalsh is typically performed on CPU via SciPy for stability
    with small/singular matrices.
    """
    grads_norm = F.normalize(grads, dim=1)
    sim = (torch.matmul(grads_norm, grads_norm.T) / n).cpu().numpy()

    lambdas = scipy.linalg.eigvalsh(sim)

    p_ = lambdas[lambdas > 1e-10]
    ent = -np.sum(p_ * np.log(p_ + 1e-12))

    return np.exp(ent)


def get_influence(pool_grads, val_grads):
    val_dir = torch.mean(val_grads, dim=0)
    val_dir = F.normalize(val_dir, dim=0)
    pool_grads = F.normalize(pool_grads, dim=1)
    scores = torch.matmul(pool_grads, val_dir)
    return scores.detach().cpu().numpy()


def get_dist_to_centroid(embeds):
    embeds_t = torch.as_tensor(embeds, dtype=torch.float32)
    embeds_t = F.normalize(embeds_t, dim=1)
    centroid = F.normalize(torch.mean(embeds_t, dim=0, keepdim=True), dim=1)
    sim = torch.matmul(embeds_t, centroid.T).squeeze(1)
    return (1.0 - sim).cpu().numpy()


def get_length_bias(indices, pool_jsonl):
    with open(pool_jsonl, "r") as f:
        pool = [json.loads(line) for line in f]
    selected_lengths = [len(pool[i]["output"].split()) for i in indices]
    ranks = np.arange(len(indices))
    rho, _ = spearmanr(selected_lengths, ranks)
    return rho


# Backward compatibility
def compute_length_bias(indices, pool_jsonl):
    return get_length_bias(indices, pool_jsonl)
