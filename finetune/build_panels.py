"""
Define the target gene panel of each finetuning cohort.

A cohort panel is the ST-Bench HVG-200 list of that cohort (`HVG_PANEL`), in
list order. Every slide must measure all 200 genes, and the panel must have
exactly 200: a model on any other panel is not comparable to the benchmark.

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
    HVG_PANEL,
    N_HVG,
    ensure_dir,
    load_base_panel,
    panel_file,
    read_genes,
    slide_ids,
)

# -------------------------------------------------------------------------------


def build(cohort: str) -> dict:
    """Take the cohort's HVG-200 list and map each slot onto the pretrained panel."""
    with open(HVG_PANEL.format(cohort=cohort)) as fh:
        genes = [str(g) for g in json.load(fh)["genes"]]
    assert len(genes) == N_HVG == len(set(genes)), \
        f"{cohort}: HVG panel has {len(genes)} genes ({len(set(genes))} unique), expected {N_HVG}"

    ids = slide_ids(cohort)
    for s in ids:
        missing = sorted(set(genes) - set(read_genes(cohort, s)))
        assert not missing, f"{cohort}/{s}: {len(missing)} HVG genes not measured: {missing[:5]}"

    base = load_base_panel()
    base_pos = {g: i for i, g in enumerate(base)}
    warm = [base_pos.get(g, -1) for g in genes]
    return {
        "cohort": cohort,
        "genes": genes,
        "n_genes": len(genes),
        "warm_start_rows": warm,
        "n_warm_start": sum(w >= 0 for w in warm),
        "base_panel": "xenium_human_multi",
        "n_base_panel": len(base),
        "slides": ids,
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


if __name__ == "__main__":
    import sys

    main(sys.argv[1:] or COHORTS)
