"""
Pearson metrics for a finished run, from the saved `pred_<slide>.npz` files.

`evaluate.py` reports Spearman only. This adds the Pearson counterparts on the
same quantities and the same log1p scale, so the two are directly comparable,
and needs no GPU: `pred_<slide>.npz` already stores `pred` and `true`.

Definitions (all on log1p(counts)):
  per-gene    Pearson across cells, one value per gene -> median / mean.
              This is the "PCC" that ST-prediction papers usually quote.
  per-cell    Pearson across genes, one value per cell -> median.
  micro-bulk  Pearson between the slide-mean predicted and true profiles.
  baseline    the same micro-bulk figure for a constant "training mean"
              predictor, which is the number the model has to beat.

    python compute_pcc.py --cohort breast_1
"""

import argparse
import glob
import json
import os

import numpy as np

from common import OUTPUTS, stats_file

# -------------------------------------------------------------------------------


def columnwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson per column, NaN where either column is constant."""
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    denom = np.sqrt((a**2).sum(0) * (b**2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (a * b).sum(0) / denom, np.nan)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    run_dir = f"{OUTPUTS}/runs/{args.run_name or args.cohort}"
    train_mean = np.load(stats_file(args.cohort))["mean"].astype(np.float64)

    out = {}
    for path in sorted(glob.glob(f"{run_dir}/pred_*.npz")):
        slide = os.path.basename(path)[len("pred_") : -len(".npz")]
        with np.load(path) as z:
            pred = z["pred"].astype(np.float64)
            true = z["true"].astype(np.float64)

        gene_r = columnwise_pearson(pred, true)
        cell_r = columnwise_pearson(pred.T, true.T)
        top50 = np.argsort(true.mean(0))[::-1][:50]
        bulk_p, bulk_t = pred.mean(0), true.mean(0)

        out[slide] = {
            "n_cells": int(pred.shape[0]),
            "n_genes": int(pred.shape[1]),
            "gene_pcc_median": float(np.nanmedian(gene_r)),
            "gene_pcc_mean": float(np.nanmean(gene_r)),
            "gene_pcc_top50_mean": float(np.nanmean(gene_r[top50])),
            "gene_pcc_frac_gt_0.5": float(np.nanmean(gene_r > 0.5)),
            "gene_pcc_frac_gt_0.3": float(np.nanmean(gene_r > 0.3)),
            "cell_pcc_median": float(np.nanmedian(cell_r)),
            "microbulk_pcc_all": float(columnwise_pearson(bulk_p[:, None], bulk_t[:, None])[0]),
            "microbulk_pcc_top50": float(
                columnwise_pearson(bulk_p[top50][:, None], bulk_t[top50][:, None])[0]
            ),
            "microbulk_pcc_meanbaseline": float(
                columnwise_pearson(train_mean[:, None], bulk_t[:, None])[0]
            ),
        }
        print(slide, json.dumps(out[slide], indent=1))

    keys = [
        "gene_pcc_median", "gene_pcc_mean", "gene_pcc_top50_mean",
        "cell_pcc_median", "microbulk_pcc_all", "microbulk_pcc_top50",
        "microbulk_pcc_meanbaseline",
    ]
    summary = {"cohort": args.cohort, "per_slide": out}
    summary.update({f"median_{k}": float(np.median([s[k] for s in out.values()])) for k in keys})
    with open(f"{run_dir}/eval_pcc.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_slide"}, indent=1))


if __name__ == "__main__":
    main()
