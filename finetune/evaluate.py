"""
Evaluate a finetuned Phoenix model on the held-out slides of its cohort.

Sampling is the released `phoenix.helpers.inference.FlowPipeline`, unchanged:
zuko's adaptive DOPRI5 at rtol=atol=1e-1 from t=0 to t=1, the result clipped at
0 and de-normalized with the cohort stats. The cached image features go in as
the pipeline's "images" (`vision_forward` passes them through when no encoder
is attached). The noise is seeded, so a rerun reproduces the predictions.

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
from torch.utils.data import DataLoader, Dataset

from common import (
    OUTPUTS,
    cell_mask,
    feature_file,
    load_panel,
    load_splits,
    load_target,
    stats_file,
)
from model import build_model

from phoenix.helpers.inference import FlowPipeline

# -------------------------------------------------------------------------------

SEED = 0


class CachedFeatures(Dataset):
    """`(features, row)` pairs of one slide's feature cache, for `FlowPipeline`."""

    def __init__(self, path: str, rows: np.ndarray):
        self.path, self.rows, self.handle = path, rows, None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        if self.handle is None:
            self.handle = h5py.File(self.path, "r")
        return torch.from_numpy(self.handle["feat"][self.rows[i]]).float(), int(self.rows[i])


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
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    run = args.run_name or args.cohort
    run_dir = f"{OUTPUTS}/runs/{run}"
    device = "cuda"

    panel = load_panel(args.cohort)
    genes = panel["genes"]
    stats = dict(np.load(stats_file(args.cohort)))
    model, _ = build_model(args.cohort, checkpoint=f"{run_dir}/flow_model_finetuned.pth")
    pipeline = FlowPipeline(model.eval().to(device), stats=stats)

    # "Collapse to the mean" reference, from the training statistics.
    train_mean = stats["mean"]

    results = {}
    for slide in load_splits(args.cohort)["val"]:
        path = feature_file(args.cohort, slide)
        with h5py.File(path, "r") as fh:
            cell_id = fh["cell_id"][:]
        # For a SPATCH cohort only the held-out spatial fold of the slide is scored.
        mask = cell_mask(args.cohort, slide, "val")
        rows = np.arange(len(cell_id)) if mask is None else np.flatnonzero(mask)
        true = load_target(args.cohort, slide, genes, cell_id)[rows]
        cell_id = cell_id[rows]
        n = len(rows)

        torch.manual_seed(SEED)
        # FlowPipeline squeezes each batch, so a last batch of one cell would lose its axis.
        batch_size = args.batch_size + int(n % args.batch_size == 1)
        loader = DataLoader(CachedFeatures(path, rows), batch_size=batch_size, shuffle=False)
        preds, _ = pipeline(genes, loader)
        preds = preds.reshape(n, len(genes)).astype(np.float32)
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
        "solver": "FlowPipeline dopri5 atol=rtol=1e-1, clip 0",
        "seed": SEED,
        "target": "log1p(CP100)",
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
