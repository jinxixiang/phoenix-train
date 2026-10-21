"""
Evaluate a finetuned Phoenix model on the held-out slides of its cohort.

Sampling integrates the learned velocity field from t=0 to t=1. The released
pipeline uses zuko's adaptive DOPRI5 at rtol=atol=1e-1; zuko is not installed in
this environment, so we integrate with fixed-step Heun instead, which is finer
than that tolerance and deterministic.

Reported metrics follow the preprint: micro-bulk Spearman (predicted vs measured
mean profile per slide, all genes and top-50 by expression) and per-gene Spearman
across cells. A "predict the training mean" baseline is included because that is
the failure mode the paper reports for competing methods.

Usage:
    python evaluate.py --cohort lung_1
"""

import argparse
import json

import h5py
import numpy as np
import torch
from scipy.stats import spearmanr

from common import OUTPUTS, cell_mask, feature_file, load_panel, load_splits, stats_file
from model import build_model, load_stats

# -------------------------------------------------------------------------------


@torch.inference_mode()
def sample(model, cond: torch.Tensor, n_genes: int, steps: int) -> torch.Tensor:
    """Integrate dx/dt = v(x, t, cond) from t=0 to t=1 with Heun's method."""
    x = torch.randn(cond.shape[0], n_genes, 1, device=cond.device)
    dt = 1.0 / steps
    for i in range(steps):
        t0 = torch.full((cond.shape[0],), i * dt, device=cond.device)
        t1 = torch.full((cond.shape[0],), (i + 1) * dt, device=cond.device)
        v0 = model(x, t0, cond)
        v1 = model(x + dt * v0, t1, cond)
        x = x + 0.5 * dt * (v0 + v1)
    return x


def per_gene_spearman(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Spearman correlation across cells, one value per gene (NaN if constant)."""
    out = np.full(pred.shape[1], np.nan)
    for g in range(pred.shape[1]):
        if true[:, g].std() > 0 and pred[:, g].std() > 0:
            out[g] = spearmanr(pred[:, g], true[:, g]).statistic
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    run = args.run_name or args.cohort
    run_dir = f"{OUTPUTS}/runs/{run}"
    device = "cuda"

    panel = load_panel(args.cohort)
    genes = panel["genes"]
    mean, std = load_stats(stats_file(args.cohort), device=device)
    model, _ = build_model(args.cohort, checkpoint=f"{run_dir}/flow_model_finetuned.pth")
    model = model.eval().to(device)

    # "Collapse to the mean" reference, from the training statistics.
    train_mean = mean.view(-1).cpu().numpy()

    results = {}
    for slide in load_splits(args.cohort)["val"]:
        with h5py.File(feature_file(args.cohort, slide), "r") as fh:
            # For a SPATCH cohort only the held-out spatial fold of the slide is
            # scored; `rows` is increasing, which is what h5py fancy indexing needs.
            mask = cell_mask(args.cohort, slide, "val")
            rows = np.arange(fh["counts"].shape[0]) if mask is None else np.flatnonzero(mask)
            counts = fh["counts"][:][rows]
            cell_id = fh["cell_id"][:][rows]
            n = len(rows)
            preds = np.empty((n, len(genes)), dtype=np.float32)
            for i in range(0, n, args.batch_size):
                batch = rows[i : i + args.batch_size]
                cond = torch.from_numpy(fh["feat"][batch]).float().to(device)
                x1 = sample(model, cond, len(genes), args.steps)
                preds[i : i + args.batch_size] = (x1 * std + mean).squeeze(-1).cpu().numpy()

        true = np.log1p(counts.astype(np.float32))
        gene_rho = per_gene_spearman(preds, true)

        # Micro-bulk: the slide's mean expression profile over genes.
        top50 = np.argsort(true.mean(0))[::-1][:50]
        bulk_pred, bulk_true = preds.mean(0), true.mean(0)
        results[slide] = {
            "n_cells": int(n),
            "microbulk_spearman_all": float(spearmanr(bulk_pred, bulk_true).statistic),
            "microbulk_spearman_top50": float(spearmanr(bulk_pred[top50], bulk_true[top50]).statistic),
            "microbulk_spearman_all_meanbaseline": float(spearmanr(train_mean, bulk_true).statistic),
            "gene_spearman_median": float(np.nanmedian(gene_rho)),
            "gene_spearman_frac_gt_0.5": float(np.nanmean(gene_rho > 0.5)),
            "gene_spearman_n_valid": int(np.sum(~np.isnan(gene_rho))),
        }
        np.savez(
            f"{run_dir}/pred_{slide}.npz",
            pred=preds.astype(np.float16),
            true=true.astype(np.float16),
            gene_spearman=gene_rho,
            genes=np.array(genes),
            # Row identity, so the benchmark aggregation can assert cell alignment
            # without re-deriving the fold mask. Absent from the first breast_1 /
            # lung_1 runs, which predate this.
            cell_id=np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cell_id]),
        )
        print(slide, json.dumps(results[slide]))

    summary = {
        "cohort": args.cohort,
        "run": run,
        "solver": f"heun-{args.steps}",
        "per_slide": results,
        "median_microbulk_all": float(np.median([r["microbulk_spearman_all"] for r in results.values()])),
        "median_microbulk_top50": float(np.median([r["microbulk_spearman_top50"] for r in results.values()])),
        "median_gene_spearman": float(np.median([r["gene_spearman_median"] for r in results.values()])),
    }
    with open(f"{run_dir}/eval.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_slide"}, indent=1))


if __name__ == "__main__":
    main()
