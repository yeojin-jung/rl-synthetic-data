import torch

def get_MD(group0, group1):
    mean0 = group0.mean(dim=0)
    mean1 = group1.mean(dim=0)
    md = mean1 - mean0
    return md

def get_LDA(group0, group1, alpha = 1e-6):
    n0 = group0.shape[0]
    n1 = group1.shape[0]

    mean0 = group0.mean(dim=0)
    mean1 = group1.mean(dim=0)
    
    S0 = torch.cov(group0.T)
    S1 = torch.cov(group1.T)
    
    Sw = ((n0 - 1) * S0 + (n1 - 1) * S1) / (n0 + n1 - 2)
    Sw += alpha * torch.eye(Sw.shape[0], device=Sw.device)
    md = mean1 - mean0
    direction = torch.linalg.solve(Sw, md)
    
    return direction