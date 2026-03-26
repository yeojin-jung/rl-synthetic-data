import torch
import torch.nn.functional as F
import numpy as np

def get_clean_mapping(mapping, vocab_dict):
    new_mapping = {}
    for k, v in mapping.items():
        if k == v:
            continue
        if k in vocab_dict and v in vocab_dict:
            new_mapping[k] = v
    print(f"Mapping size: {len(new_mapping)}")
    return new_mapping


def get_base_target_probs(probs, indices0, indices1):
    probs0 = probs[:,indices0]
    probs1 = probs[:,indices1]
    return probs0.sum(dim = -1), probs1.sum(dim = -1)

def get_off_probs(probs, mapping, vocab_dict):
    off_probs = probs.clone()
    delete_indices = []
    for k, v in mapping.items():
        off_probs[:, vocab_dict[v]] += off_probs[:, vocab_dict[k]]
        off_probs[:, vocab_dict[k]] = 0.0
        delete_indices.append(vocab_dict[k])
    
    keep_indices = [i for i in range(off_probs.size(1)) if i not in delete_indices]
    off_probs = off_probs[:, keep_indices]
    
    return off_probs

def get_kls(probs, offset = 5e-3):
    q = probs[0]
    forward_kl = torch.sum(q * (torch.log(q + offset) - torch.log(probs + offset)), dim=-1)
    return forward_kl

def get_cos(probs, G, direction):
    duals = probs @ G
    dual_diff = duals[1:] - duals[:-1]
    dual_diff = torch.cat([dual_diff, dual_diff[-1].unsqueeze(0)], dim=0)
    normalized_dual_diff = dual_diff / (dual_diff.norm(dim=-1, keepdim=True) + 1e-16)
    normalized_direction = direction / (direction.norm() + 1e-16)
    return normalized_dual_diff @ normalized_direction


def get_rank_diff(probs, use_topp = True):
    if use_topp:
        seq = np.linspace(0, len(probs) - 1, 20, dtype=int).tolist()
        topp_indices = []
        for i in seq:
            q = probs[i]
            sorted_probs, sorted_indices = torch.sort(q, dim=-1, descending=True)
            cumsum_probs = sorted_probs.cumsum(dim=-1)
            cum_sort = cumsum_probs - sorted_probs
            topp_indices.extend(sorted_indices[cum_sort < 0.999].tolist())
        topp_indices = list(set(topp_indices))

        probs = probs[:, topp_indices]
        if probs.sum(dim =-1).min() < 0.99:
            print(probs.sum(dim =-1).min())
            print("Sum of selected probabilities is less than 0.99")

    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    ranks = torch.zeros_like(sorted_indices, dtype=torch.float)
    ranks.scatter_(-1, sorted_indices, 
                torch.arange(probs.size(-1),
                                dtype=torch.float).expand_as(sorted_indices).to(probs.device))
    ranks += 1 
    
    rank_diff = (1/ranks - 1/ranks[0]).abs()
    q = probs[0]
    q = q / q.sum()

    return rank_diff @ q