import torch
from tqdm import tqdm
from ..core.geometry import get_mean_cov


def e_steering(start_primal, direction, G,
               num_steps = 200, step_size = 1,
               use_tqdm = False, mapping = None, vocab_dict = None):
    current_primal = start_primal.clone()
    primals = [current_primal]
    direction = direction / direction.norm()

    if mapping is not None and vocab_dict is not None:
        indices0 = [vocab_dict[i] for i in list(set(mapping.keys()))]
        indices1 = [vocab_dict[i] for i in list(set(mapping.values()))]

    step_range = tqdm(range(num_steps)) if use_tqdm else range(num_steps)

    for _ in step_range:
        current_primal = current_primal + direction * step_size
        primals.append(current_primal)

        if mapping is not None and vocab_dict is not None:
            prob = current_primal @ G.T
            prob = torch.softmax(prob, dim=-1)
            prob0 = prob[indices0].sum()
            prob1 = prob[indices1].sum()
            ratio = prob1 / (prob0 + prob1 + 1e-10)
            if ratio > 0.9999:
                break

    primals = torch.stack(primals)
    return primals




def m_steering(start_primal, direction, G,
               num_steps=200, step_size=2, alpha=1e-3, topk=20000,
               use_tqdm=True, mapping=None, vocab_dict=None):
    if topk > G.shape[0]:
        topk = G.shape[0]

    if mapping is not None and vocab_dict is not None:
        indices0 = [vocab_dict[i] for i in list(set(mapping.keys()))]
        indices1 = [vocab_dict[i] for i in list(set(mapping.values()))]

    current_primal = start_primal.clone()
    primals = [current_primal]
    direction = direction / direction.norm()
    step_range = tqdm(range(num_steps)) if use_tqdm else range(num_steps)
    
    device = current_primal.device
    eye_alpha = torch.eye(current_primal.shape[0], device=device) * alpha
    direction_col = direction.unsqueeze(1)
    
    for _ in step_range:
        mean, cov = get_mean_cov(current_primal, G, topk=topk)
        reg_cov = cov + eye_alpha
        
        L = torch.linalg.cholesky(reg_cov)
        sol = torch.cholesky_solve(direction_col, L).squeeze(1)
        
        sol = sol / sol.norm()
        current_primal = current_primal + sol * step_size
        primals.append(current_primal.clone())

        if mapping is not None and vocab_dict is not None:
            prob = current_primal @ G.T
            prob = torch.softmax(prob, dim=-1)
            prob0 = prob[indices0].sum()
            prob1 = prob[indices1].sum()
            ratio = prob1 / (prob0 + prob1 + 1e-10)
            if ratio > 0.9999:
                break
    
    primals = torch.stack(primals)
    return primals