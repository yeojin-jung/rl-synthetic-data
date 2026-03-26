import os
import json
import re
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.eval_utils import get_g_vendi, get_influence, get_dist_to_centroid, get_length_bias

def save_barplot(metric_name, methods, values, out_dir):
    plt.figure(figsize=(8, 4), dpi=150)
    plt.bar(methods, values, color="#4C78A8")
    plt.title(metric_name)
    plt.ylabel(metric_name)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{metric_name.lower()}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

def extract_answer(text):
    match = re.search(r"####\s*([-+]?\d*\.?\d+)", text)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    return numbers[-1] if numbers else None

def run_benchmark(model, tokenizer, data_path, name, max_examples=None):
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]
    if max_examples is not None:
        data = data[:max_examples]

    correct = 0
    for item in tqdm(data, desc=f"Eval {name}"):
        prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=128, temperature=0.1, do_sample=False)
        response = tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        pred = extract_answer(response)
        gold = str(item.get("answer") or item.get("output"))
        if pred == gold:
            correct += 1

    return (correct / len(data)) * 100


def logprob_answer(model, tokenizer, instruction, answer):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\nThe answer is "
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(str(answer), add_special_tokens=False).input_ids
    if not answer_ids:
        return -math.inf, -math.inf, 0.0

    full_ids = prompt_ids + answer_ids
    input_ids = torch.tensor([full_ids], device=model.device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)

    start = len(prompt_ids)
    score = 0.0
    for i in range(start, len(full_ids)):
        token_id = full_ids[i]
        score += log_probs[i - 1, token_id].item()
    avg = score / max(len(answer_ids), 1)
    prob = math.exp(avg)
    return score, avg, prob

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--pool-name", type=str, default="candidate_pool")
    parser.add_argument("--indices-dir", type=str, default="data/selected_indices")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--pool-grads", type=str, default="data/features/pool_grads.pt")
    parser.add_argument("--val-grads", type=str, default="data/features/val_grads.pt")
    parser.add_argument("--pool-jsonl", type=str, default="data/processed/candidate_pool.jsonl")
    args = parser.parse_args()

    results = {}
    methods = ["base", "random", "less", "dsir", "prismatic_soft"]
    benchmarks = {
        "ID_External": "data/eval/gsm8k_test.jsonl",
        "OOD_Symbolic": "data/eval/gsm_symbolic.jsonl",
    }

    out_dir = os.path.join("outputs", args.pool_name, "metrics")
    prob_dir = os.path.join("outputs", args.pool_name, "benchmark_probs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(prob_dir, exist_ok=True)

    pool_grads = torch.load(args.pool_grads)
    val_grads = torch.load(args.val_grads)
    influence_scores = get_influence(pool_grads, val_grads)
    dist_scores = get_dist_to_centroid(pool_grads)

    base = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B", dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    for method in methods:
        if method == "base":
            model = base
            indices = None
        else:
            indices_path = os.path.join(args.indices_dir, f"{method}_indices.npy")
            if not os.path.exists(indices_path):
                print(f"Skipping {method}: missing {indices_path}")
                continue
            indices = np.load(indices_path)
            model = PeftModel.from_pretrained(base, os.path.join(args.models_dir, f"final-{method}"))

        # 2. Accuracy Metrics
        method_results = {}
        for b_name, b_path in benchmarks.items():
            method_results[b_name] = float(
                run_benchmark(model, tokenizer, b_path, b_name, max_examples=args.max_examples)
            )

            # Log-prob of correct answer (teacher forcing on "The answer is {ANS}")
            with open(b_path, "r") as f:
                data = [json.loads(line) for line in f]
            if args.max_examples is not None:
                data = data[: args.max_examples]
            prob_rows = []
            for i, item in enumerate(tqdm(data, desc=f"LogProb {b_name} | {method}")):
                ans = str(item.get("answer") or item.get("output"))
                lp_sum, lp_avg, p_avg = logprob_answer(
                    model, tokenizer, item["instruction"], ans
                )
                prob_rows.append(
                    {
                        "index": i,
                        "answer": ans,
                        "logprob_sum": lp_sum,
                        "logprob_avg": lp_avg,
                        "prob_avg": p_avg,
                    }
                )
            prob_path = os.path.join(prob_dir, f"{method}_{b_name}_probs.json")
            with open(prob_path, "w") as pf:
                json.dump(prob_rows, pf, indent=2)
            print(f"Saved {prob_path}")

        # 3. Structural Metrics
        if indices is not None:
            method_results["G_Vendi"] = float(get_g_vendi(pool_grads[indices].to(torch.float32), len(indices)))
            method_results["Influence_Mean"] = float(np.mean(influence_scores[indices]))
            method_results["Dist_To_Centroid_Mean"] = float(np.mean(dist_scores[indices]))
            method_results["Length_Bias"] = float(get_length_bias(indices, args.pool_jsonl))
        else:
            method_results["G_Vendi"] = None
            method_results["Influence_Mean"] = None
            method_results["Dist_To_Centroid_Mean"] = None
            method_results["Length_Bias"] = None

        results[method] = method_results

        # Cleanup to save VRAM
        if method != "base":
            del model
        torch.cuda.empty_cache()

    with open(os.path.join(out_dir, "results_summary.json"), "w") as f:
        json.dump(results, f, indent=4)
    print("\nAll results saved to results_summary.json")

    ordered_methods = [m for m in methods if m in results]
    if not ordered_methods:
        print("No methods found to plot.")
        return

    struct_methods = [m for m in ordered_methods if results[m]["G_Vendi"] is not None]
    if struct_methods:
        save_barplot(
            "G_Vendi",
            struct_methods,
            [results[m]["G_Vendi"] for m in struct_methods],
            out_dir,
        )
        save_barplot(
            "Influence_Mean",
            struct_methods,
            [results[m]["Influence_Mean"] for m in struct_methods],
            out_dir,
        )
        save_barplot(
            "Dist_To_Centroid_Mean",
            struct_methods,
            [results[m]["Dist_To_Centroid_Mean"] for m in struct_methods],
            out_dir,
        )
        save_barplot(
            "Length_Bias",
            struct_methods,
            [results[m]["Length_Bias"] for m in struct_methods],
            out_dir,
        )

    # Benchmark performance barplots (ID/OOD)
    for b_name in benchmarks.keys():
        save_barplot(
            b_name,
            ordered_methods,
            [results[m][b_name] for m in ordered_methods],
            out_dir,
        )

if __name__ == "__main__":
    main()
