import torch
import torch.nn.functional as F

def get_mean_cov(primal, G, topk = None):
    logit = G @ primal

    if topk is not None:
        topk_vals, topk_idx = torch.topk(logit, topk)
        prob = F.softmax(topk_vals, dim=-1)
        G_top = G[topk_idx]
        mean = prob @ G_top
        cov = G_top.T @ (G_top * prob.unsqueeze(1)) - mean.unsqueeze(1) @ mean.unsqueeze(0)
    else:
        prob = F.softmax(logit, dim=-1)
        mean = prob @ G
        cov = G.T @ (G * prob.unsqueeze(1)) - mean.unsqueeze(1) @ mean.unsqueeze(0)

    return mean, cov

def primal_to_dual(primal, G):
    logit = G @ primal
    prob = F.softmax(logit, dim=-1)
    dual = prob @ G
    return dual

def primals_to_duals(primals, G, batch_size = 512):
    if primals.ndim == 1:
        primals = primals.unsqueeze(0)
    duals = []
    for i in range(0, primals.size(0), batch_size):
        batch = primals[i : i+batch_size]
        logit = batch @ G.T
        prob = F.softmax(logit, dim=-1)
        dual = prob @ G
        duals.append(dual)
    return torch.cat(duals, dim=0)