import torch
import torch.nn.functional as F
import json
import re
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from scipy.stats import spearmanr
from scipy.linalg import eigh

def extract_answer(text):
    # Standard math extraction: find the last number or #### pattern
    match = re.search(r"####\s*([-+]?\d*\.?\d+)", text)
    if match: return match.group(1).strip()
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    return numbers[-1] if numbers else None

def compute_g_vendi(indices, pool_grads_path):
    """Structural Diversity"""
    n = len(indices)
    pool_grads = torch.load(pool_grads_path)
    selected_grads = pool_grads[indices].to(torch.float32)
    grads_norm = F.normalize(selected_grads, dim=1)
    K = torch.matmul(grads_norm, grads_norm.T / n).cpu().numpy()
    evals = eigh(K, eigvals_only=True)
    evals = np.maximum(evals, 1e-10)
    p = evals / np.sum(evals)
    return np.exp(-np.sum(p * np.log(p)))

def compute_length_bias(indices, pool_jsonl):
    """Is selection biased towards output token length?"""
    with open(pool_jsonl, "r") as f:
        pool = [json.loads(line) for line in f]
    selected_lengths = [len(pool[i]['output'].split()) for i in indices]

    ranks = np.arange(len(indices)) 
    rho, _ = spearmanr(selected_lengths, ranks)
    return rho

def run_benchmark(model, tokenizer, data_path, name):
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    correct = 0
    for item in tqdm(data, desc=f"Eval {name}"):
        prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=128, temperature=0.1, do_sample=False)
        response = tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        pred = extract_answer(response)
        gold = str(item.get('answer') or item.get('output'))
        if pred == gold: correct += 1
            
    return (correct / len(data)) * 100

def main():
    results = {}
    methods = ["random", "less", "dsir", "prismatic"]
    benchmarks = {
        "ID_Internal": "data/processed/target_set.jsonl",
        "ID_External": "data/eval/gsm8k_test.jsonl",
        "OOD_Symbolic": "data/eval/gsm_symbolic.jsonl"
    }

    for method in methods:
        print(f"\n>>> Final Evaluation: {method}")
        indices = np.load(f"data/selected_indices/{method}_indices.npy")
        
        # 1. Load Model
        base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(base, f"./models/final-{method}")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        
        # 2. Accuracy Metrics
        method_results = {}
        for b_name, b_path in benchmarks.items():
            method_results[b_name] = run_benchmark(model, tokenizer, b_path, b_name)
        
        # 3. Structural Metrics
        method_results["G_Vendi"] = compute_g_vendi(indices, "data/features/pool_grads.pt")
        method_results["Length_Bias"] = compute_length_bias(indices, "data/processed/candidate_pool.jsonl")
        
        results[method] = method_results
        
        # Cleanup to save VRAM
        del model, base
        torch.cuda.empty_cache()

    with open("results_summary.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nAll results saved to results_summary.json")

if __name__ == "__main__":
    main()