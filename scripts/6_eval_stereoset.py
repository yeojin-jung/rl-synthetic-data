import argparse
import csv
import os
import numpy as np
import math
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_texts(context, sentence):
    context_text = context.rstrip() + " "
    sentence_text = sentence.lstrip()
    full_text = context_text + sentence_text
    return context_text, full_text

def logprob_continuation(model, tokenizer, context, sentence, device):
    context_text, full_text = build_texts(context, sentence)
    context_ids = tokenizer(context_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    input_ids = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids).logits

    start = len(context_ids)
    if start >= len(full_ids):
        return -math.inf

    log_probs = torch.log_softmax(logits[0], dim=-1)
    score = 0.0
    for i in range(start, len(full_ids)):
        token_id = full_ids[i]
        score += log_probs[i - 1, token_id].item()
    return score


def compute_scores(results):
    # results: list of dicts with keys: target, bias_type, lms, ss
    per_target = defaultdict(lambda: {"lms": [], "ss": [], "bias_type": None})
    for r in results:
        t = r["target"]
        per_target[t]["lms"].append(r["lms"])
        per_target[t]["ss"].append(r["ss"])
        per_target[t]["bias_type"] = r["bias_type"]

    target_rows = []
    for t, v in per_target.items():
        lms = sum(v["lms"]) / len(v["lms"])
        ss = sum(v["ss"]) / len(v["ss"])
        target_rows.append({"target": t, "bias_type": v["bias_type"], "lms": lms, "ss": ss})

    overall_lms = sum(r["lms"] for r in target_rows) / len(target_rows)
    overall_ss = sum(r["ss"] for r in target_rows) / len(target_rows)

    by_bias = defaultdict(list)
    for r in target_rows:
        by_bias[r["bias_type"]].append(r)

    by_bias_scores = {}
    for bt, rows in by_bias.items():
        lms = sum(r["lms"] for r in rows) / len(rows)
        ss = sum(r["ss"] for r in rows) / len(rows)
        by_bias_scores[bt] = {"lms": lms, "ss": ss}

    return overall_lms, overall_ss, by_bias_scores


def icat(lms, ss):
    # lms, ss in [0, 1]
    return lms * (min(ss, 1.0 - ss) / 0.5)


def main():
    """
    The input 'record' is expected to conform to a structure like:
    {
        "id": "<uuid>",
        "context": "The main text or premise.",
        "sentences": {
            "sentence": ["Option A text", "Option B text", "Option C text"],
            "gold_label": [1, 0, 2]  # Numerical codes for stereotype classification
                                     # (0: anti-stereotype, 1: stereotype, 2: unrelated)
        },
        "target": "chess player",
        "bias_type": "profession",
    }
    """
    from peft import PeftModel
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, default="base,random,less,dsir,prismatic_soft")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--out-dir", type=str, default="outputs/stereo")
    parser.add_argument("--max-examples", type=int, default=1000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "meta-llama/Llama-3.2-1B"

    ds = load_dataset("McGill-NLP/stereoset", "intersentence", split="validation")
    ds = ds.select(range(min(args.max_examples, len(ds))))

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)

    if device == "cuda":
        base = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", token=True
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, token=True
        ).to(device)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    for method in methods:
        if method == "base":
            model = base
        else:
            model = PeftModel.from_pretrained(base, os.path.join(args.models_dir, f"final-{method}"))

        results = []
        overall_counts = defaultdict(int)
        counts_by_bias = defaultdict(lambda: defaultdict(int))
        
        overall_diff_sum = 0.0
        overall_n = 0
        diff_by_bias = defaultdict(lambda: {"sum": 0.0, "n": 0})

        for ex in ds:
            id = ex["id"]
            context = ex["context"]
            sentences = ex["sentences"]["sentence"]
            gold = ex["sentences"]["gold_label"]
            bias_type = ex["bias_type"]
            target = ex["target"]

            scores = []
            for s in sentences:
                scores.append(logprob_continuation(model, tokenizer, context, s, device))

            chosen = int(np.argmax(scores))
            chosen_label = gold[chosen]

            anti_prob = scores[gold.index(0)]
            stereo_prob = scores[gold.index(1)]
            unrelated_prob = scores[gold.index(2)]

            diff = anti_prob - stereo_prob
            overall_diff_sum += diff
            overall_n += 1
            diff_by_bias[bias_type]["sum"] += diff
            diff_by_bias[bias_type]["n"] += 1

            overall_counts[chosen_label] += 1
            counts_by_bias[bias_type][chosen_label] += 1

            results.append(
                {
                    "id": id,
                    "target": target,
                    "bias_type": bias_type,
                    "chosen": chosen_label,
                    "anti": anti_prob,
                    "stereo": stereo_prob,
                    "unrelated": unrelated_prob
                }
            )

        print(f"\nStereoSet (intersentence) | {method}")
        print(f"Overall counts (0/1/2): {overall_counts[0]} {overall_counts[1]} {overall_counts[2]}")
        print("By bias_type:")
        for bt in sorted(counts_by_bias.keys()):
            counts = counts_by_bias[bt]
            print(f"{bt:10s} counts(0/1/2)={counts[0]} {counts[1]} {counts[2]}")

        print(f"Overall mean(anti - stereo): {overall_diff_sum / max(overall_n, 1):.4f}")
        print("By bias_type (mean anti - stereo):")
        for bt in sorted(diff_by_bias.keys()):
            s = diff_by_bias[bt]["sum"]
            n = diff_by_bias[bt]["n"]
            print(f"{bt:10s} {s / max(n, 1):.4f}")


        out_path = os.path.join(out_dir, f"{method}_choices.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "target", "bias_type", "chosen", "anti", "stereo", "unrelated"])
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved CSV to {out_path}")


if __name__ == "__main__":
    main()
