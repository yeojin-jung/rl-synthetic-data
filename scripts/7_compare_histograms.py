#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def list_pools(outputs_dir: Path) -> List[str]:
    pools = []
    for p in outputs_dir.iterdir():
        if p.is_dir() and p.name not in {"umap"}:
            pools.append(p.name)
    return sorted(pools)


def load_probs(path: Path) -> Dict[int, float]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(row["index"]): float(row["logprob_avg"]) for row in data}


def load_stereo_csv(path: Path) -> Dict[str, Dict[str, float]]:
    import csv

    rows = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows[r["id"]] = {
                "anti": float(r["anti"]),
                "stereo": float(r["stereo"]),
                "unrelated": float(r["unrelated"]),
            }
    return rows


def save_hist(values: List[float], out_path: Path, title: str, bins: int = 30):
    if not values:
        return
    plt.figure(figsize=(6, 4), dpi=150)
    plt.hist(values, bins=bins, color="#4C78A8", alpha=0.85)
    plt.title(title)
    plt.xlabel("Delta")
    plt.ylabel("Count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument("--pools", type=str, default="candidate_pool,candidate_pool_0.6,candidate_pool_0.99")
    parser.add_argument("--selectors", type=str, default="random,less,dsir,prismatic_soft")
    parser.add_argument("--bins", type=int, default=30)
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    pools = [p.strip() for p in args.pools.split(",") if p.strip()] if args.pools else list_pools(outputs_dir)
    selectors = [s.strip() for s in args.selectors.split(",") if s.strip()]

    benchmarks = ["ID_External", "OOD_Symbolic"]

    # Build 4x4 grid: rows = ID/OOD/Stereo/Leak, cols = selectors
    fig, axes = plt.subplots(4, len(selectors), figsize=(4 * len(selectors), 10), dpi=150)
    if len(selectors) == 1:
        axes = np.expand_dims(axes, axis=1)

    row_titles = ["ID_External", "OOD_Symbolic", "Stereotype (gender)", "Leak (gender)"]
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    for p_idx, pool in enumerate(pools):
        pool_dir = outputs_dir / pool
        prob_dir = pool_dir / "benchmark_probs"
        stereo_dir = pool_dir / "stereo"
        if not prob_dir.exists():
            print(f"Skipping {pool}: missing {prob_dir}")
            continue

        # Preload base probs for benchmarks
        base_probs = {}
        for b in benchmarks:
            base_path = prob_dir / f"base_{b}_probs.json"
            if base_path.exists():
                base_probs[b] = load_probs(base_path)

        # Preload base stereo rows
        base_stereo_path = stereo_dir / "base_choices.csv"
        base_rows = load_stereo_csv(base_stereo_path) if base_stereo_path.exists() else {}

        # Map id->bias_type for gender filtering
        bias_type_by_id = {}
        if base_stereo_path.exists():
            import csv
            with base_stereo_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    bias_type_by_id[r["id"]] = r.get("bias_type", "")

        for c, method in enumerate(selectors):
            # Row 0: ID utility delta
            b = "ID_External"
            ax = axes[0, c]
            if b in base_probs:
                method_path = prob_dir / f"{method}_{b}_probs.json"
                if method_path.exists():
                    method_probs = load_probs(method_path)
                    keys = sorted(set(base_probs[b].keys()) & set(method_probs.keys()))
                    deltas = [method_probs[k] - base_probs[b][k] for k in keys]
                    ax.hist(deltas, bins=args.bins, color=colors[p_idx % len(colors)], alpha=0.35, label=pool)
            ax.set_title(method, fontsize=9)
            ax.set_ylabel(row_titles[0], fontsize=9)

            # Row 1: OOD utility delta
            b = "OOD_Symbolic"
            ax = axes[1, c]
            if b in base_probs:
                method_path = prob_dir / f"{method}_{b}_probs.json"
                if method_path.exists():
                    method_probs = load_probs(method_path)
                    keys = sorted(set(base_probs[b].keys()) & set(method_probs.keys()))
                    deltas = [method_probs[k] - base_probs[b][k] for k in keys]
                    ax.hist(deltas, bins=args.bins, color=colors[p_idx % len(colors)], alpha=0.35, label=pool)
            ax.set_ylabel(row_titles[1], fontsize=9)

            # Row 2: stereotype delta (gender only)
            ax = axes[2, c]
            method_path = stereo_dir / f"{method}_choices.csv"
            if base_rows and method_path.exists():
                method_rows = load_stereo_csv(method_path)
                shared = sorted(set(base_rows.keys()) & set(method_rows.keys()))
                deltas = []
                for k in shared:
                    if bias_type_by_id.get(k) != "gender":
                        continue
                    b = base_rows[k]
                    m = method_rows[k]
                    deltas.append((m["stereo"] - m["anti"]) - (b["stereo"] - b["anti"]))
                ax.hist(deltas, bins=args.bins, color=colors[p_idx % len(colors)], alpha=0.35, label=pool)
            ax.set_ylabel(row_titles[2], fontsize=9)

            # Row 3: leak delta (gender only)
            ax = axes[3, c]
            if base_rows and method_path.exists():
                method_rows = load_stereo_csv(method_path)
                shared = sorted(set(base_rows.keys()) & set(method_rows.keys()))
                deltas = []
                for k in shared:
                    if bias_type_by_id.get(k) != "gender":
                        continue
                    b = base_rows[k]
                    m = method_rows[k]
                    b_leak = max(b["stereo"], b["anti"]) - b["unrelated"]
                    m_leak = max(m["stereo"], m["anti"]) - m["unrelated"]
                    deltas.append(m_leak - b_leak)
                ax.hist(deltas, bins=args.bins, color=colors[p_idx % len(colors)], alpha=0.35, label=pool)
            ax.set_ylabel(row_titles[3], fontsize=9)

    # Add legend to the top-right subplot
    axes[0, -1].legend(fontsize=8)

    fig.suptitle("Deltas vs Base (Overlayed Pools)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir = outputs_dir / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "delta_grid_overlay.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
