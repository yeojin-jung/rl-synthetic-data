import torch
import pandas as pd
from src.gradients import GradientExtractor
from src.selectors import select_random, select_less, select_cluster_grad, select_g_vendi

# 1. Extract gradients
extractor = GradientExtractor(batch_size=16)
pool_grads = extractor.process_file("data/processed/candidate_pool.jsonl")
target_grads = extractor.process_file("data/processed/target_set.jsonl")

# 2. Data selection
K = 50
n = len(pool_grads)
print("Selecting data...")
selections = {
    "random": select_random(n, K),
    "less": select_less(pool_grads, target_grads, K),
    "cluster": select_cluster_grad(pool_grads, K),
    "g_vendi": select_g_vendi(pool_grads, K)
}

# 3. Evaluation
results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    p_grads_gpu = pool_grads.to(device)
    t_grads_gpu = pool_grads.to(device)
    target_mean = torch.mean(t_grads_gpu, dim=0, keepdim=True)
    
    for method, indices in selections.items():
        # Calculate similarity
        selected_mean = torch.mean(p_grads_gpu[indices], dim=0, keepdim=True)
        sim_score = torch.nn.functional.cosine_similarity(
            selected_mean, 
            target_mean
        ).item()
        results.append({"method": method, "target_similarity": sim_score})

# 4. SUMMARY
df = pd.DataFrame(results)
print("\nFinal Benchmark Results:")
print(df)
df.to_csv("benchmark_results.csv", index=False)