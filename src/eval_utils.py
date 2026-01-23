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
    
    # Diversity Entropy
    p_ = lambdas[lambdas > 1e-10]
    ent = -np.sum(p_ * np.log(p_ + 1e-12))
    
    return np.exp(ent)
