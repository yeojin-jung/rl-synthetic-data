#!/usr/bin/env python3
"""
Steer / Leak Pipeline — Intrasentence StereoSeq
=================================================
For intrasentence StereoSeq, the task is:

  context:  "The schoolboy ___ the girl"
  choices:  ["touched", "helped", "frog"]   (stereo / anti / unrelated)

We measure log P(choice_token | context_before_BLANK) at the masked position.

This means:
  - Context embedding λ = last hidden state at the token just before BLANK
  - log P(word) = log_softmax(λ @ G.T)[word_token_id]
  - Steering λ directly gives us the steered log-probabilities with ONE matmul
  - No need for multi-token continuation scoring at all

Steer_i(k) = |δ_i^+(k) − δ_i^-(k)| / (2k)
  where δ_i^±(k) = log P(stereo|λ_±k) − log P(anti|λ_±k)

Leak_j(k) = (δ_j^+(k) − δ_j^-(k)) / (2k)
  where δ_j^±(k) = log P(correct|λ_±k)
  and λ is the math context embedding at the last token of "Problem: ...\nSolution:"
"""
import os
import sys
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dual_steering"))
import information_geometry as ig
from information_geometry.core.embeddings import get_llm_embeddings
from information_geometry.directions.linear import get_MD

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID       = "google/gemma-3-4b-pt"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE     = 16

E_NUM_STEPS    = 200;  E_STEP_SIZE = 0.5
M_NUM_STEPS    = 200;  M_STEP_SIZE = 2;   M_ALPHA = 5e-3

STEP_SNAPSHOTS = [10, 25, 50, 100, 200]


# ─────────────────────────────────────────────────────────────────────────────
# PARSING
# ─────────────────────────────────────────────────────────────────────────────

def load_pairs(path: Path):
    male, female = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2 and parts[0] and parts[1]:
                male.append(parts[0].strip())
                female.append(parts[1].strip())
    return male, female


def parse_intrasentence(ds, bias_type="gender", max_n=None):
    """
    Parse intrasentence StereoSeq.
    gold_label is an integer: 0=anti-stereotype, 1=stereotype, 2=unrelated
    We recover the fill-in word by stripping the context_before prefix from
    each full sentence, then taking the first word of the remainder.
    """
    instances = []
    for ex in ds:
        if ex["bias_type"] != bias_type:
            continue

        context_template = ex["context"]
        if "BLANK" not in context_template:
            continue

        context_before = context_template.split("BLANK")[0].rstrip()
        context_after  = context_template.split("BLANK")[1].lstrip()

        label_map = {}
        for sent, lbl in zip(
            ex["sentences"]["sentence"],
            ex["sentences"]["gold_label"],   # integers: 0, 1, 2
        ):
            prefix    = context_before.lower().rstrip()
            sent_lower = sent.lower()
            if sent_lower.startswith(prefix):
                remainder = sent[len(context_before):].lstrip()
                fill_word = remainder.split()[0].rstrip(".,!?;:") if remainder else ""
            else:
                words  = sent.split()
                cb_len = len(context_before.split())
                fill_word = words[cb_len] if cb_len < len(words) else ""

            label_map[lbl] = fill_word   # keys are ints 0, 1, 2

        # BUG FIX 1: guard uses integer keys (not strings)
        if not all(k in label_map for k in (0, 1, 2)):
            continue

        instances.append({
            "context_before": context_before,
            "context_after":  context_after,
            "stereo":         label_map[1],   # 1 = stereotype
            "anti":           label_map[0],   # 0 = anti-stereotype
            "unrelated":      label_map[2],   # 2 = unrelated
        })

        if max_n and len(instances) >= max_n:
            break

    return instances

# ─────────────────────────────────────────────────────────────────────────────
# CORE SCORING — single token at BLANK / answer position
# ─────────────────────────────────────────────────────────────────────────────

def score_word_at_blank(
    word:           str,
    steered_lambda: torch.Tensor,   # [hidden_dim]
    G:              torch.Tensor,   # [vocab, hidden_dim]
    tokenizer,
) -> float:
    """
    log P(word | λ_steered) = log_softmax(λ_steered @ G.T)[word_token_id]
    Uses the first sub-token if word tokenises to multiple tokens.
    """
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        return float("-inf")

    token_id  = ids[0]
    logits    = steered_lambda.to(G.device) @ G.T
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs[token_id].item()


def delta_at_step(
    word_a:   str,
    word_b:   str,
    path:     torch.Tensor,   # [num_steps+1, hidden_dim]
    step_idx: int,
    G:        torch.Tensor,
    tokenizer,
) -> float:
    """δ = log P(word_a | λ_k) − log P(word_b | λ_k)"""
    k   = min(step_idx, path.shape[0] - 1)
    lam = path[k].to(DEVICE)
    return (score_word_at_blank(word_a, lam, G, tokenizer)
            - score_word_at_blank(word_b, lam, G, tokenizer))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — STEER AND LEAK
# ─────────────────────────────────────────────────────────────────────────────

def compute_steer_and_leak(
    stereo_instances: list,
    intra_paths_pos:  list,
    intra_paths_neg:  list,
    math_instances:   list,
    math_paths_pos:   list,
    math_paths_neg:   list,
    G:                torch.Tensor,
    tokenizer,
    step_snapshots:   list = STEP_SNAPSHOTS,
    steering_type:    str  = "m",
):
    steer_records = []
    leak_records  = []

    # ── 4a: Steer_i  ──────────────────────────────────────────────────────
    print(f"\n[Step 4a] Steer_i on {len(stereo_instances)} intrasentence instances...")

    for i, inst in enumerate(tqdm(stereo_instances)):
        path_pos = intra_paths_pos[i][steering_type]
        path_neg = intra_paths_neg[i][steering_type]

        row = {
            "id":      i,
            "context": inst["context_before"][:60],
            "stereo":  inst["stereo"],
            "anti":    inst["anti"],
        }

        for k in step_snapshots:
            d_pos = delta_at_step(
                inst["stereo"], inst["anti"], path_pos, k, G, tokenizer)
            d_neg = delta_at_step(
                inst["stereo"], inst["anti"], path_neg, k, G, tokenizer)

            steer_i = abs(d_pos - d_neg) / (2 * k + 1e-9)

            row[f"delta_pos_k{k}"] = d_pos
            row[f"delta_neg_k{k}"] = d_neg
            row[f"steer_k{k}"]     = steer_i

        steer_records.append(row)

    # ── 4b: Leak_j  ───────────────────────────────────────────────────────
    print(f"\n[Step 4b] Leak_j on {len(math_instances)} math problems...")

    for j, item in enumerate(tqdm(math_instances)):
        ans = item["answer"]

        # Tokenise answer once outside the step loop
        ids = tokenizer.encode(" " + ans, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(ans, add_special_tokens=False)
        if not ids:
            continue   # BUG FIX 2: was `return float("-inf")` — wrong in a loop

        token_id = ids[0]

        path_pos = math_paths_pos[j][steering_type]
        path_neg = math_paths_neg[j][steering_type]

        # BUG FIX 3: `row` was never initialised before the step loop
        row = {
            "id":      j,
            "context": item["instruction"][:60],
            "answer":  ans,
        }

        for k in step_snapshots:
            k_pos = min(k, path_pos.shape[0] - 1)
            k_neg = min(k, path_neg.shape[0] - 1)

            pos_lam       = path_pos[k_pos].to(G.device)
            pos_logits    = pos_lam @ G.T
            pos_log_probs = torch.log_softmax(pos_logits, dim=-1)
            d_pos         = pos_log_probs[token_id].item()

            neg_lam       = path_neg[k_neg].to(G.device)
            neg_logits    = neg_lam @ G.T
            neg_log_probs = torch.log_softmax(neg_logits, dim=-1)
            d_neg         = neg_log_probs[token_id].item()

            leak_j = (d_pos - d_neg) / (2 * k + 1e-9)

            row[f"delta_pos_k{k}"] = d_pos
            row[f"delta_neg_k{k}"] = d_neg
            row[f"leak_k{k}"]      = leak_j

        leak_records.append(row)

    # ── 4c: Aggregate  ────────────────────────────────────────────────────
    print("\n[Step 4c] Aggregate metrics:")
    summary = {}
    for k in step_snapshots:
        sv = [r[f"steer_k{k}"] for r in steer_records if f"steer_k{k}" in r]
        lv = [abs(r[f"leak_k{k}"]) for r in leak_records if f"leak_k{k}" in r]

        ms = float(np.mean(sv)) if sv else 0.0
        ml = float(np.mean(lv)) if lv else 0.0
        er = ml / (ms + 1e-12)

        summary[f"step_{k}"] = {"mean_steer": ms, "mean_abs_leak": ml, "ER": er}
        print(f"  k={k:4d} | Steer={ms:.5f} | |Leak|={ml:.5f} | ER={er:.4f}")

    return {"summary": summary, "steer_records": steer_records,
            "leak_records": leak_records}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs",        default="data/[male - female].txt")
    parser.add_argument("--math-pool",    default="candidate_pool.json")
    parser.add_argument("--model",        default=MODEL_ID)
    parser.add_argument("--out",          default="outputs/steer")
    parser.add_argument("--batch-size",   type=int, default=BATCH_SIZE)
    parser.add_argument("--steering",     default="m", choices=["e", "m"])
    parser.add_argument("--max-stereo",   type=int, default=200)
    parser.add_argument("--max-math",     type=int, default=200)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    _, _, vocab_dict, _, G = ig.load_model_and_vocab(args.model, device=DEVICE)

    # ── Step 1: Gender direction ───────────────────────────────────────────
    print("Building gender direction...")
    male_words, female_words = load_pairs(Path(args.pairs))

    male_emb = get_llm_embeddings(
        male_words, args.model, batch_size=args.batch_size,
        last_position_only=True, layer_indices=[-1])["embeddings"]
    female_emb = get_llm_embeddings(
        female_words, args.model, batch_size=args.batch_size,
        last_position_only=True, layer_indices=[-1])["embeddings"]

    direction = get_MD(male_emb, female_emb)
    torch.save(direction.cpu(), out_path / "gender_direction.pt")
    print(f"  norm={direction.norm():.4f}")

    # ── Step 2: Intrasentence StereoSeq ───────────────────────────────────
    print("Parsing intrasentence StereoSeq...")
    intra_ds = load_dataset(
        "McGill-NLP/stereoset", "intrasentence", split="validation")

    stereo_instances = parse_intrasentence(
        intra_ds, bias_type="gender", max_n=args.max_stereo)
    print(f"  {len(stereo_instances)} gender instances parsed")

    contexts_before = [inst["context_before"] for inst in stereo_instances]
    intra_emb = get_llm_embeddings(
        contexts_before, args.model, batch_size=args.batch_size,
        last_position_only=True, layer_indices=[-1])["embeddings"]

    torch.save(intra_emb.cpu(), out_path / "intra_embeddings.pt")

    # ── Step 3: Math pool ─────────────────────────────────────────────────
    print("Loading math pool...")
    benchmarks = {
        "ID_External":  "data/eval/gsm8k_test.jsonl",
        "OOD_Symbolic": "data/eval/gsm_symbolic.jsonl",
    }

    math_instances, math_contexts = [], []
    for b_name, b_path in benchmarks.items():
        with open(b_path, "r") as f:
            math_data = [json.loads(line) for line in f]
        if args.max_math is not None:
            math_data = math_data[:args.max_math]
        for d in math_data:
            ans = str(d.get("answer") or d.get("output") or "")
            if not ans:
                continue
            math_instances.append({"instruction": d["instruction"], "answer": ans})
            math_contexts.append(f"Problem: {d['instruction']}\nSolution:")
            if len(math_instances) >= args.max_math:
                break

    math_emb = get_llm_embeddings(
        math_contexts, args.model, batch_size=args.batch_size,
        last_position_only=True, layer_indices=[-1])["embeddings"]

    torch.save(math_emb.cpu(), out_path / "math_embeddings.pt")
    print(f"  {len(math_instances)} math problems")

    # ── Step 3b: Steering paths ────────────────────────────────────────────
    direction_neg = -direction

    def run_paths(emb_tensor, desc):
        pos_paths, neg_paths = [], []
        for lam in tqdm(emb_tensor, desc=f"{desc} +dir"):
            pos_paths.append({
                "e": ig.e_steering(lam, direction,     G,
                                   num_steps=E_NUM_STEPS, step_size=E_STEP_SIZE,
                                   use_tqdm=False),
                "m": ig.m_steering(lam, direction,     G, alpha=M_ALPHA,
                                   num_steps=M_NUM_STEPS, step_size=M_STEP_SIZE,
                                   use_tqdm=False),
            })
        for lam in tqdm(emb_tensor, desc=f"{desc} -dir"):
            neg_paths.append({
                "e": ig.e_steering(lam, direction_neg, G,
                                   num_steps=E_NUM_STEPS, step_size=E_STEP_SIZE,
                                   use_tqdm=False),
                "m": ig.m_steering(lam, direction_neg, G, alpha=M_ALPHA,
                                   num_steps=M_NUM_STEPS, step_size=M_STEP_SIZE,
                                   use_tqdm=False),
            })
        return pos_paths, neg_paths

    intra_paths_pos, intra_paths_neg = run_paths(intra_emb, "StereoSeq")
    math_paths_pos,  math_paths_neg  = run_paths(math_emb,  "Math")

    torch.save(intra_paths_pos, out_path / "intra_paths_pos.pt")
    torch.save(intra_paths_neg, out_path / "intra_paths_neg.pt")
    torch.save(math_paths_pos,  out_path / "math_paths_pos.pt")
    torch.save(math_paths_neg,  out_path / "math_paths_neg.pt")

    # ── Step 4: Compute Steer + Leak ──────────────────────────────────────
    results = compute_steer_and_leak(
        stereo_instances = stereo_instances,
        intra_paths_pos  = intra_paths_pos,
        intra_paths_neg  = intra_paths_neg,
        math_instances   = math_instances,
        math_paths_pos   = math_paths_pos,
        math_paths_neg   = math_paths_neg,
        G                = G,
        tokenizer        = tokenizer,
        step_snapshots   = STEP_SNAPSHOTS,
        steering_type    = "m",
    )

    out_file = out_path / f"steer_leak_m.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_file}")

    results = compute_steer_and_leak(
        stereo_instances = stereo_instances,
        intra_paths_pos  = intra_paths_pos,
        intra_paths_neg  = intra_paths_neg,
        math_instances   = math_instances,
        math_paths_pos   = math_paths_pos,
        math_paths_neg   = math_paths_neg,
        G                = G,
        tokenizer        = tokenizer,
        step_snapshots   = STEP_SNAPSHOTS,
        steering_type    = "e",
    )

    out_file = out_path / f"steer_leak_e.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_file}")


if __name__ == "__main__":
    main()