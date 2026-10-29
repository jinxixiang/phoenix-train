"""
Score the finetuned Phoenix as one more model of the ST-Bench single-cell
HVG-200 arm, with the convention of the ST-Bench single-cell aggregation script.

Everything that defines the comparison is taken from the benchmark, not from
this pipeline:

  * test slides come from `ST-Bench/input/storm_sc_hvg200/<bench>/ids.csv`
    (fold 0), never from a directory listing,
  * ground truth is `stpbench/sc_hvg200/<bench>/st/<slide>.h5ad`, and the
    scored target is `log1p(GT.X)` -- CP100-normalized expression, the unit
    Phoenix is finetuned on here (`common.load_target`), so predictions are
    scored as they come out of `evaluate.py`,
  * per-gene PCC / Spearman / SSIM over cells, pooled per output cohort with
    `np.concatenate` and reduced with `np.nanmean`; the three metric functions
    are copied verbatim from the ST-Bench aggregation script.

    python aggregate_stbench_sc.py --cohort breast_1 --cohort lung_1
    python aggregate_stbench_sc.py --all --write            # write .npy + csv
"""

import argparse
import os

import anndata
import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common import OUTPUTS, feature_file

# -------------------------------------------------------------------------------

# ST-Bench locations (environment-configurable); see README "Scoring against ST-Bench".
GT_ROOT = os.environ.get("STBENCH_SC_GT_ROOT", "data/stbench/sc_hvg200")
IDS = os.environ.get("STBENCH_SC_IDS", "data/stbench/input/storm_sc_hvg200/{bench}/ids.csv")
HE2ST_SC = os.environ.get("STBENCH_SC_OUT", "data/stbench/he2st_sc")
PACK = os.environ.get("PHOENIX_COHORT_DIR", "data/finetune_sc") + "/slides"

MODEL_KEY = "phoenix"
METRICS = ("pcc", "spearman", "ssim")

# Output cohort -> benchmark cohorts, as in the ST-Bench aggregation script.
COHORTS = {
    "breast_1": ["breast_1"],
    "lung_1": ["lung_1"],
    "spatch-xenium": ["spatch_xenium_coad", "spatch_xenium_hcc", "spatch_xenium_ov"],
    "spatch-cosmx": ["spatch_cosmx_coad", "spatch_cosmx_hcc", "spatch_cosmx_ov"],
    "spatch-visium-hd": [
        "spatch_visiumhd_ff_coad", "spatch_visiumhd_ff_hcc", "spatch_visiumhd_ff_ov",
        "spatch_visiumhd_ffpe_coad", "spatch_visiumhd_ffpe_hcc", "spatch_visiumhd_ffpe_ov",
    ],
}

def bench_to_run(bench: str) -> str:
    """Phoenix run name (= this pipeline's cohort) for a benchmark cohort."""
    return bench


def pack_slide(slide: str) -> str:
    """Pack / feature-file slide id for a benchmark slide id."""
    return slide[: -len("_val")] if slide.endswith("_val") else slide


# ---- metrics: copied verbatim from the ST-Bench aggregation script -----------


def gene_pcc(labels, preds):
    L = labels - labels.mean(axis=0)
    P = preds - preds.mean(axis=0)
    num = (L * P).sum(axis=0)
    den = np.sqrt((L ** 2).sum(axis=0) * (P ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def gene_ssim(labels, preds):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    l_rng = labels.max(axis=0) - labels.min(axis=0)
    p_rng = preds.max(axis=0) - preds.min(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        X = (labels - labels.min(axis=0)) / l_rng
        Y = (preds - preds.min(axis=0)) / p_rng
        mx, my = X.mean(axis=0), Y.mean(axis=0)
        vx, vy = X.var(axis=0), Y.var(axis=0)
        cov = ((X - mx) * (Y - my)).mean(axis=0)
        ssim = ((2 * mx * my + C1) * (2 * cov + C2)) / \
               ((mx ** 2 + my ** 2 + C1) * (vx + vy + C2))
    ssim[(l_rng == 0) | (p_rng == 0)] = np.nan
    return ssim


def gene_metrics(labels, preds):
    return {"pcc": gene_pcc(labels, preds),
            "spearman": gene_pcc(rankdata(labels, axis=0), rankdata(preds, axis=0)),
            "ssim": gene_ssim(labels, preds)}


# ------------------------------------------------------------------------------


def dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def test_slides(bench: str) -> list[str]:
    ids = pd.read_csv(IDS.format(bench=bench))
    return sorted(ids.loc[ids["fold_0"] == "test", "sample_id"].astype(str))


def load_prediction(run: str, bench: str, slide: str, var_names,
                    run_suffix: str = ""):
    """Phoenix prediction for `slide`, in benchmark gene order and units."""
    path = f"{OUTPUTS}/runs/{run}{run_suffix}/pred_{pack_slide(slide)}.npz"
    if not os.path.exists(path):
        return None, None
    with np.load(path) as z:
        pred = z["pred"].astype(np.float64)
        genes = [str(g) for g in z["genes"]]
        cells = [str(c) for c in z["cell_id"]] if "cell_id" in z else None
    pos = {g: i for i, g in enumerate(genes)}
    missing = [g for g in var_names if g not in pos]
    assert not missing, f"{bench}/{slide}: {len(missing)} HVG-200 genes not predicted: {missing[:5]}"
    return pred[:, [pos[g] for g in var_names]], cells


def check_cell_order(run: str, bench: str, slide: str, obs_names, cells) -> None:
    """
    Assert that prediction row i is GT cell i.

    Runs since 2026-09-08 store `cell_id` in the prediction file; the first
    breast_1 / lung_1 runs do not, and for those the rows are the feature
    file's cell order.
    """
    if cells is None:
        with h5py.File(feature_file(run, pack_slide(slide)), "r") as fh:
            cells = [c.decode() if isinstance(c, bytes) else str(c) for c in fh["cell_id"][:]]
    assert cells == list(obs_names), f"{bench}/{slide}: prediction cell order != GT obs order"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", action="append", default=[],
                        help="output cohort key (repeatable); default: all")
    parser.add_argument("--all", action="store_true", help="every output cohort (the default)")
    parser.add_argument("--write", action="store_true",
                        help="write phoenix_{metric}.npy into STBENCH_SC_OUT")
    parser.add_argument("--allow-partial", action="store_true",
                        help="write a cohort even if some of its test slides are missing")
    parser.add_argument("--suffix", default="",
                        help="appended to the model key, e.g. '_interim'")
    parser.add_argument("--run-suffix", default="",
                        help="read predictions from runs/<cohort><suffix>, e.g. '_interim'")
    args = parser.parse_args()

    cohorts = list(COHORTS) if args.all or not args.cohort else args.cohort
    key = MODEL_KEY + args.suffix
    summary, stats = [], []

    for cohort in cohorts:
        pooled = {m: [] for m in METRICS}
        n_slides = n_cells = n_expected = 0
        for bench in COHORTS[cohort]:
            run = bench_to_run(bench)
            for slide in test_slides(bench):
                n_expected += 1
                gt = anndata.read_h5ad(f"{GT_ROOT}/{bench}/st/{slide}.h5ad")
                assert gt.n_vars == 200, f"{bench}/{slide}: GT has {gt.n_vars} genes"
                labels = np.log1p(dense(gt.X).astype(np.float64))

                pred, cells = load_prediction(run, bench, slide, gt.var_names,
                                              args.run_suffix)
                if pred is None:
                    print(f"[skip] {bench}/{slide}: no Phoenix prediction yet")
                    continue
                assert pred.shape[0] == gt.n_obs, \
                    f"{bench}/{slide}: {pred.shape[0]} predicted cells vs {gt.n_obs} in GT"
                check_cell_order(run, bench, slide, gt.obs_names, cells)

                mets = gene_metrics(labels, pred)
                for m in METRICS:
                    pooled[m].append(mets[m])
                n_slides += 1
                n_cells += gt.n_obs
                summary.append({"cohort": cohort, "bench_cohort": bench, "slide": slide,
                                "model": key,
                                **{f"mean_{m}": np.nanmean(mets[m]) for m in METRICS},
                                "n_genes_scored": int(np.sum(~np.isnan(mets["pcc"])))})
                print(f"  {bench}/{slide}: " + ", ".join(
                    f"{m}={np.nanmean(mets[m]):.4f}" for m in METRICS)
                      + f" ({gt.n_obs} cells)")

        if not pooled["pcc"]:
            print(f"[skip] {cohort}: nothing scored")
            continue
        vals = {m: np.concatenate(pooled[m]) for m in METRICS}
        stats.append({"cohort": cohort, "n_test_slides": n_slides, "n_test_cells": n_cells})
        print(f"[cohort] {cohort}: {n_slides} slides, {n_cells} cells, "
              + ", ".join(f"{m} mean={np.nanmean(vals[m]):.4f}" for m in METRICS))

        # A cohort pools its test slides, so writing it half-finished would
        # silently publish a number computed on a subset of the benchmark.
        complete = n_slides == n_expected
        if not complete:
            print(f"    [partial] {n_slides}/{n_expected} test slides scored"
                  + ("" if args.allow_partial else " -- not writing (use --allow-partial)"))

        out_dir = f"{HE2ST_SC}/{cohort}_hvg200"
        for m in METRICS:
            path = f"{out_dir}/{key}_{m}.npy"
            if args.write and (complete or args.allow_partial):
                os.makedirs(out_dir, exist_ok=True)
                np.save(path, vals[m])
            elif not args.write:
                print(f"    [dry-run] would write {path}")

        # Side by side with the benchmark's own models, same pooling.
        existing = sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []
        for m in METRICS:
            row = {"phoenix": np.nanmean(vals[m])}
            for f in existing:
                if f.endswith(f"_{m}.npy") and not f.startswith(MODEL_KEY):
                    row[f[: -len(f"_{m}.npy")]] = float(np.nanmean(np.load(f"{out_dir}/{f}")))
            order = sorted(row, key=row.get, reverse=True)
            print(f"    {m:9s} " + "  ".join(
                f"{'*' if k == 'phoenix' else ''}{k}={row[k]:.4f}" for k in order))

    if args.write and summary:
        df = pd.DataFrame(summary)
        path = f"{HE2ST_SC}/summary_hvg200_{key}.csv"
        df.to_csv(path, index=False)
        pd.DataFrame(stats).to_csv(f"{HE2ST_SC}/cohort_stats_hvg200_{key}.csv", index=False)
        print(f"\nWrote {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
