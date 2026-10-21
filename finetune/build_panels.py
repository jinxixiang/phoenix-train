"""
Define the target gene panel of each finetuning cohort.

A cohort panel is the intersection of the per-slide panels (one lung slide,
NCBI859, is missing a gene), kept in the order of the first training slide.

Phoenix encodes gene identity purely positionally: token `i` is embedded with
row `i` of `px_embedding`. Finetuning on a new panel therefore means deciding,
for each slot of the new panel, whether a pretrained row can be reused. We reuse
the row of the pretrained multi-tissue panel whenever the gene names match, and
mark the remaining slots for re-initialization.

Writes `outputs/panels/<cohort>.json`.
"""

import json

from common import (
    COHORTS,
    ensure_dir,
    load_base_panel,
    load_splits,
    panel_file,
    read_genes,
    slide_ids,
)

# -------------------------------------------------------------------------------


def build(cohort: str) -> dict:
    """Intersect the per-slide panels and map each slot onto the pretrained panel."""
    ids = slide_ids(cohort)
    per_slide = {s: read_genes(cohort, s) for s in ids}

    first = per_slide[load_splits(cohort)["train"][0]]
    shared = set.intersection(*(set(g) for g in per_slide.values()))
    genes = [g for g in first if g in shared]

    base = load_base_panel()
    base_pos = {g: i for i, g in enumerate(base)}
    warm = [base_pos.get(g, -1) for g in genes]

    dropped = {s: sorted(set(g) - shared) for s, g in per_slide.items()}
    return {
        "cohort": cohort,
        "genes": genes,
        "n_genes": len(genes),
        "warm_start_rows": warm,
        "n_warm_start": sum(w >= 0 for w in warm),
        "base_panel": "xenium_human_multi",
        "n_base_panel": len(base),
        "slides": ids,
        "dropped_per_slide": {s: d for s, d in dropped.items() if d},
    }


def main(cohorts: list[str]) -> None:
    for cohort in cohorts:
        panel = build(cohort)
        with open(ensure_dir(panel_file(cohort)), "w") as fh:
            json.dump(panel, fh, indent=1)
        print(
            f"{cohort}: {panel['n_genes']} genes "
            f"({panel['n_warm_start']} warm-started from {panel['base_panel']}, "
            f"{panel['n_genes'] - panel['n_warm_start']} re-initialized), "
            f"{len(panel['slides'])} slides"
        )
        for slide, genes in panel["dropped_per_slide"].items():
            print(f"   dropped from {slide}: {genes}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:] or COHORTS)
