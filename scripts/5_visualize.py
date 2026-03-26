import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import umap.umap_ as umap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.eval_utils import get_g_vendi, get_influence, get_dist_to_centroid


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def maybe_load_text_embeds(pool_path, val_path, pool_out, val_out, text_model, batch_size):
    if os.path.exists(pool_out) and os.path.exists(val_out):
        pool_embeds = torch.load(pool_out)
        val_embeds = torch.load(val_out)
        return pool_embeds, val_embeds

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers is required to compute text embeddings. "
            "Install it or precompute embeddings and save to the expected paths."
        ) from e

    pool = load_jsonl(pool_path)
    val = load_jsonl(val_path)
    pool_texts = [f"{data['instruction']}\n{data['output']}" for data in pool]
    val_texts = [f"{data['instruction']}\n{data['output']}" for data in val]

    model = SentenceTransformer(text_model)
    pool_embeds = model.encode(pool_texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    val_embeds = model.encode(val_texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)

    os.makedirs(os.path.dirname(pool_out), exist_ok=True)
    torch.save(torch.tensor(pool_embeds), pool_out)
    torch.save(torch.tensor(val_embeds), val_out)
    return torch.tensor(pool_embeds), torch.tensor(val_embeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument("--embedding", choices=["gradient", "text"], default="text")
    parser.add_argument("--pool-path", type=str, default="data/processed/candidate_pool.jsonl")
    parser.add_argument("--val-path", type=str, default="data/processed/val_set.jsonl")
    parser.add_argument("--pool-grads", type=str, default="data/features/pool_grads.pt")
    parser.add_argument("--val-grads", type=str, default="data/features/val_grads.pt")
    parser.add_argument("--pool-text-embeds", type=str, default="data/features/pool_text_embeds.pt")
    parser.add_argument("--val-text-embeds", type=str, default="data/features/val_text_embeds.pt")
    parser.add_argument("--text-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--compute-text-embeds", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-metric", type=str, default="cosine")
    parser.add_argument("--methods", type=str, default="random,less,prismatic,prismatic_soft, dsir")
    parser.add_argument("--indices-dir", type=str, default="data/selected_indices")
    parser.add_argument("--out-dir", type=str, default="outputs/umap")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pool_grads = torch.load(args.pool_grads)
    val_grads = torch.load(args.val_grads)
    pool = load_jsonl(args.pool_path)
    is_contaminated = np.array([bool(d.get("is_contaminated")) for d in pool])

    if args.embedding == "gradient":
        pool_embeds = pool_grads
        val_embeds = val_grads
    else:
        pool_embeds, val_embeds = maybe_load_text_embeds(
            args.pool_path,
            args.val_path,
            args.pool_text_embeds,
            args.val_text_embeds,
            args.text_model,
            args.batch_size
        )

    all_embeds = torch.cat([pool_embeds, val_embeds], dim=0).detach().cpu().numpy()
    reducer = umap.UMAP(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=42,
    )
    all_2d = reducer.fit_transform(all_embeds)
    n_pool = pool_embeds.shape[0]
    pool_2d = all_2d[:n_pool]
    val_2d = all_2d[n_pool:]

    influence_scores = get_influence(pool_grads, val_grads)
    diversity_scores = get_dist_to_centroid(pool_embeds.detach().cpu().numpy())

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for method in methods:
        indices_path = os.path.join(args.indices_dir, f"{method}_indices.npy")
        if not os.path.exists(indices_path):
            print(f"Skipping {method}: missing {indices_path}")
            continue

        indices = np.load(indices_path)
        if len(indices) > args.k:
            indices = indices[: args.k]

        g_vendi = get_g_vendi(pool_grads[indices].to(torch.float32), len(indices))

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
        fig.suptitle(
            f"{method} | embedding={args.embedding} | k={len(indices)} | G-Vendi={g_vendi:.3f}",
            fontsize=11,
        )

        # Influence heatmap
        ax = axes[0]
        sc = ax.scatter(
            pool_2d[:, 0],
            pool_2d[:, 1],
            c=influence_scores,
            s=6,
            cmap="viridis",
            alpha=0.7,
            linewidths=0,
        )
        selected_contam = indices[is_contaminated[indices]]
        selected_clean = indices[~is_contaminated[indices]]
        if len(selected_clean) > 0:
            ax.scatter(
                pool_2d[selected_clean, 0],
                pool_2d[selected_clean, 1],
                s=30,
                facecolors="none",
                edgecolors="red",
                linewidths=1.0,
                label="selected (clean)",
            )
        if len(selected_contam) > 0:
            ax.scatter(
                pool_2d[selected_contam, 0],
                pool_2d[selected_contam, 1],
                s=30,
                facecolors="none",
                edgecolors="blue",
                linewidths=1.0,
                label="selected (contaminated)",
            )
        ax.set_title("Influence (pool)")
        ax.legend(loc="best", fontsize=8)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        # Diversity heatmap (proxy)
        ax = axes[1]
        sc = ax.scatter(
            pool_2d[:, 0],
            pool_2d[:, 1],
            c=diversity_scores,
            s=6,
            cmap="magma",
            alpha=0.7,
            linewidths=0,
        )
        if len(selected_clean) > 0:
            ax.scatter(
                pool_2d[selected_clean, 0],
                pool_2d[selected_clean, 1],
                s=30,
                facecolors="none",
                edgecolors="red",
                linewidths=1.0,
                label="selected (clean)",
            )
        if len(selected_contam) > 0:
            ax.scatter(
                pool_2d[selected_contam, 0],
                pool_2d[selected_contam, 1],
                s=30,
                facecolors="none",
                edgecolors="blue",
                linewidths=1.0,
                label="selected (contaminated)",
            )
        ax.set_title("Diversity (1 - cos to centroid)")
        ax.legend(loc="best", fontsize=8)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")

        out_path = os.path.join(args.out_dir, f"umap_{args.embedding}_{method}.png")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved {out_path}")

    # save a clean heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.suptitle(
        f"Heatmap | embedding={args.embedding}",
        fontsize=11,
    )

    ax = axes[0]
    sc = ax.scatter(
        pool_2d[:, 0],
        pool_2d[:, 1],
        c=influence_scores,
        s=6,
        cmap="viridis",
        alpha=0.7,
        linewidths=0,
    )
    ax.set_title("Influence (pool)")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    sc = ax.scatter(
        pool_2d[:, 0],
        pool_2d[:, 1],
        c=diversity_scores,
        s=6,
        cmap="magma",
        alpha=0.7,
        linewidths=0,
    )
    ax.set_title("Diversity (1 - cos to centroid)")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

    out_path = os.path.join(args.out_dir, f"umap_{args.embedding}_clean.png")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")

    # save a clean heatmap + target only
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.suptitle(
        f"Heatmap + Target | embedding={args.embedding}",
        fontsize=11,
    )

    ax = axes[0]
    sc = ax.scatter(
        pool_2d[:, 0],
        pool_2d[:, 1],
        c=influence_scores,
        s=6,
        cmap="viridis",
        alpha=0.7,
        linewidths=0,
    )
    ax.scatter(
        val_2d[:, 0],
        val_2d[:, 1],
        s=20,
        marker="x",
        c="black",
        linewidths=0.8,
        label="target",
    )
    ax.set_title("Influence (pool)")
    ax.legend(loc="best", fontsize=8)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    sc = ax.scatter(
        pool_2d[:, 0],
        pool_2d[:, 1],
        c=diversity_scores,
        s=6,
        cmap="magma",
        alpha=0.7,
        linewidths=0,
    )
    ax.scatter(
        val_2d[:, 0],
        val_2d[:, 1],
        s=20,
        marker="x",
        c="black",
        linewidths=0.8,
        label="target",
    )
    ax.set_title("Diversity (1 - cos to centroid)")
    ax.legend(loc="best", fontsize=8)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")

    out_path = os.path.join(args.out_dir, f"umap_{args.embedding}_target.png")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
