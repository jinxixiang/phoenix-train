"""
Per-gene mean/std of the Phoenix target, computed on the training slides only.

The target is `log1p(CP100)` on the HVG-200 panel (see `common.load_target`),
z-scored per gene. Same layout as the `stats_table.npz` that ships with the
released checkpoints.

Writes `outputs/panels/<cohort>_stats.npz`.
"""

import numpy as np

from common import (
    COHORTS,
    cell_mask,
    ensure_dir,
    load_panel,
    load_splits,
    load_target,
    stats_file,
)

# -------------------------------------------------------------------------------


def main(cohorts: list[str]) -> None:
    for cohort in cohorts:
        genes = load_panel(cohort)["genes"]
        train = load_splits(cohort)["train"]

        total = np.zeros(len(genes), dtype=np.float64)
        total_sq = np.zeros(len(genes), dtype=np.float64)
        n = 0
        for slide in train:
            expr = load_target(cohort, slide, genes).astype(np.float64)
            # SPATCH cohorts are one slide: only its training folds count.
            mask = cell_mask(cohort, slide, "train")
            if mask is not None:
                expr = expr[mask]
            total += expr.sum(0)
            total_sq += (expr**2).sum(0)
            n += len(expr)

        mean = total / n
        std = np.sqrt(np.maximum(total_sq / n - mean**2, 0.0))
        assert np.all(std > 0), "constant gene in the training split"

        np.savez(
            ensure_dir(stats_file(cohort)),
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
        )
        print(
            f"{cohort}: {len(genes)} genes over {n} cells from {len(train)} train slides | "
            f"mean {mean.min():.4f}-{mean.max():.4f}  std {std.min():.4f}-{std.max():.4f}"
        )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:] or COHORTS)
