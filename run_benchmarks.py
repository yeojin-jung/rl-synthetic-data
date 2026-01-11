import torch
import json
import pandas as pd
from src.gradients import GradientExtractor
from src.selectors import select_random, select_less, select_cluster_grad, get_g_vendi

# 1. Extract gradients
extractor = GradientExtractor(batch_size=16)

def load_texts(path):
    with open(path, 'r') as f:
        return [json.loads(line)["text"] for line in f]

pool_texts = load_texts("data/processed/candidate_pool.jsonl")
target_texts = load_texts("data/processed/target_set.jsonl")

pool_grads = extractor.process_file("data/processed/candidate_pool.jsonl")
target_grads = extractor.process_file("data/processed/target_set.jsonl")

# 2. Data selection
K = 50
n = len(pool_grads)
print("Selecting data...")
selections = {
    "random": select_random(n, K),
    "less": select_less(pool_grads, target_grads, K),
    "cluster": select_cluster_grad(pool_grads, K)
}

# 3. Fine-tuning Evaluation
results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for method, indices in selections.items():
    print(f"Evaluating: {method}")

    # 1) Cosine Similarity of gradients between target and selected
    selected_mean = torch.mean(pool_grads[indices].to(device), dim=0)
    target_mean = torch.mean(target_grads.to(device), dim=0)
    sim_score = torch.nn.functional.cosine_similarity(selected_mean.unsqueeze(0), target_mean.unsqueeze(0)).item()

    # 2) G-Vendi Score
    gvendi = get_g_vendi(pool_grads[indices], len(indices))

    # 3) Fine Tuning loss
    selected_texts = [pool_texts[i] for i in indices]
    target_loss = extractor.finetune_subset(selected_texts, target_texts)

    results.append({
        "method": method,
        "similarity": sim_score,
        "gvendi": gvendi,
        "target_loss": target_loss
    })

# 4. Results
df = pd.DataFrame(results)
df = df.sort_values("target_loss")
print("\nFinal Benchmark (Lower Loss = Better Selection):")
print(df)
df.to_csv("final_selection_results.csv", index=False)