"""
Shared paths and constants for the Phoenix finetuning pipeline.

Phoenix conventions taken from the preprint (Methods: Image/Gene preprocessing)
and the released `nest/multi` checkpoint:

- patches are 224 x 224 px at 20x (0.5 um/px), i.e. a 112 um field of view,
  centered on the cell,
- expression is shifted-log transformed with NO library-size normalization,
  then z-scored per gene with the cohort `stats_table`,
- the image encoder (H-Optimus-1 style ViT-g/14 with 4 registers) is frozen;
  its weights ship inside `flow_model.pth`.
"""

import json
import os
import sys

import h5py
import numpy as np

# -------------------------------------------------------------------------------

# All locations are configurable through the environment; the defaults put
# data, assets and outputs under the repository root.
PHOENIX_DIR = os.environ.get(
    "PHOENIX_TRAIN_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

# The upstream `phoenix` package is vendored under src/ (see src/phoenix/VENDORED.md)
# so no separate install is needed. Set PHOENIX_SRC to point at a different
# checkout (the directory containing `phoenix/`).
sys.path.insert(0, os.environ.get("PHOENIX_SRC", f"{PHOENIX_DIR}/src"))

# Packed cohorts (`<cohort>/<slide>.h5`, `splits.json`); see README "Data format".
COHORT_DIR = os.environ.get("PHOENIX_COHORT_DIR", f"{PHOENIX_DIR}/data/finetune_sc")
# H&E whole-slide images of the slide-split cohorts (`<slide>.tif`).
WSI_DIR = os.environ.get("PHOENIX_WSI_DIR", f"{PHOENIX_DIR}/data/wsis")
# SPATCH root (`<Platform>/<TUMOUR>/HE.tif`).
SPATCH_DIR = os.environ.get("PHOENIX_SPATCH_DIR", f"{PHOENIX_DIR}/data/SPATCH")

ASSETS = os.environ.get("PHOENIX_ASSETS", f"{PHOENIX_DIR}/assets")
OUTPUTS = os.environ.get("PHOENIX_OUTPUTS", f"{PHOENIX_DIR}/outputs")
BASE_CKPT = f"{ASSETS}/weights/nest/multi/flow_model.pth"
BASE_PANEL = f"{ASSETS}/panels/xenium_human_multi.npy"

# The two HEST cohorts are split by slide; the SPATCH cohorts are one slide each,
# split by spatial fold (fold `VAL_FOLD` is held out). Cohort keys are the
# ST-Bench benchmark cohort names, so a cohort maps 1:1 onto a row of the
# single-cell HVG-200 arm -- see `aggregate_stbench_sc.py`.
SLIDE_SPLIT_COHORTS = ["breast_1", "lung_1"]

# cohort -> (pack sub-directory under slides/spatch_pku, slide id, WSI sub-path)
SPATCH_COHORTS = {
    f"spatch_{plat_key}_{tum.lower()}": (plat_key, f"{plat_slide}_{tum}", f"{plat_wsi}/{tum}")
    for plat_key, plat_slide, plat_wsi in [
        ("xenium", "Xenium", "Xenium"),
        ("cosmx", "CosMx", "CosMx"),
        ("visiumhd_ff", "Visium_HD_FF", "Visium_HD_FF"),
        ("visiumhd_ffpe", "Visium_HD_FFPE", "Visium_HD_FFPE"),
    ]
    for tum in ("COAD", "HCC", "OV")
}

COHORTS = SLIDE_SPLIT_COHORTS + list(SPATCH_COHORTS)

VAL_FOLD = 4

# Phoenix 20x single-cell patch geometry.
PATCH_UM = 112.0
PATCH_PX = 224

# H-Optimus-1 image normalization (from the Phoenix README usage example).
IMAGE_MEAN = (0.707223, 0.578729, 0.703617)
IMAGE_STD = (0.211883, 0.230117, 0.177517)

# -------------------------------------------------------------------------------


def is_spatch(cohort: str) -> bool:
    """True for the one-slide, spatially-folded SPATCH cohorts."""
    return cohort in SPATCH_COHORTS


def slide_path(cohort: str, slide_id: str) -> str:
    """Path of a packed slide in the STORM finetune_sc cohort."""
    if is_spatch(cohort):
        platform, _, _ = SPATCH_COHORTS[cohort]
        return f"{COHORT_DIR}/slides/spatch_pku/{platform}/{slide_id}.h5"
    return f"{COHORT_DIR}/slides/{cohort}/{slide_id}.h5"


def wsi_path(cohort: str, slide_id: str) -> str:
    """Path of the H&E whole-slide image a cohort's crops are cut from."""
    if is_spatch(cohort):
        _, _, wsi_dir = SPATCH_COHORTS[cohort]
        return f"{SPATCH_DIR}/{wsi_dir}/HE.tif"
    return f"{WSI_DIR}/{slide_id}.tif"


def load_splits(cohort: str) -> dict:
    """
    Return the ``{"train": [...], "val": [...]}`` slide split for a cohort.

    A SPATCH cohort is a single slide that appears in both halves; which of its
    cells belong to which half is `cell_mask`.
    """
    if is_spatch(cohort):
        _, slide_id, _ = SPATCH_COHORTS[cohort]
        return {"train": [slide_id], "val": [slide_id]}
    with open(f"{COHORT_DIR}/splits.json") as fh:
        return json.load(fh)["cohorts"][cohort]


def cell_mask(cohort: str, slide_id: str, part: str) -> np.ndarray | None:
    """
    Boolean mask over a slide's cells for the "train" or "val" half.

    `None` means "every cell", which is the case for the slide-split cohorts.
    For SPATCH the mask comes from the pack's `fold`: fold `VAL_FOLD` is the
    benchmark's `<slide>_val` test set, the rest is training.
    """
    if not is_spatch(cohort):
        return None
    with h5py.File(slide_path(cohort, slide_id), "r") as fh:
        fold = fh["fold"][:]
    return fold == VAL_FOLD if part == "val" else fold != VAL_FOLD


def slide_ids(cohort: str) -> list[str]:
    """All slide ids of a cohort, train split first."""
    split = load_splits(cohort)
    if is_spatch(cohort):
        return list(split["train"])
    return list(split["train"]) + list(split["val"])


def read_genes(cohort: str, slide_id: str) -> list[str]:
    """Gene panel of one packed slide, in that slide's column order."""
    with h5py.File(slide_path(cohort, slide_id), "r") as fh:
        return [g.decode() if isinstance(g, bytes) else str(g) for g in fh["genes"][:]]


def panel_file(cohort: str) -> str:
    """Path of the cohort panel definition written by `build_panels.py`."""
    return f"{OUTPUTS}/panels/{cohort}.json"


def load_panel(cohort: str) -> dict:
    """Load the cohort panel definition (gene list plus warm-start map)."""
    with open(panel_file(cohort)) as fh:
        return json.load(fh)


def stats_file(cohort: str) -> str:
    """Path of the cohort per-gene mean/std table written by `build_stats.py`."""
    return f"{OUTPUTS}/panels/{cohort}_stats.npz"


def feature_file(cohort: str, slide_id: str) -> str:
    """Path of the cached image features + counts for one slide."""
    return f"{OUTPUTS}/features/{cohort}/{slide_id}.h5"


def features_complete(cohort: str, slide_id: str) -> bool:
    """
    True if a cached feature file is finished, not just present.

    `extract_features.py` pre-allocates `feat` and only writes `counts` and the
    attributes after the last row, so a killed or preempted extraction leaves a
    large but useless file behind (18 GB of zeros, in one observed case). Size
    is therefore not a usable test; this checks the same invariants as
    `verify_features.py`.
    """
    path = feature_file(cohort, slide_id)
    if not os.path.exists(path):
        return False
    try:
        with h5py.File(path, "r") as fh:
            n = int(fh.attrs["n_cells"])
            if fh["feat"].shape[0] != n or fh["counts"].shape[0] != n:
                return False
            last = fh["feat"][n - 1].astype(np.float32)
        return bool(np.isfinite(last).all() and np.abs(last).sum() > 0)
    except (OSError, KeyError):
        return False


def load_base_panel() -> list[str]:
    """Gene list of the pretrained Phoenix multi-tissue panel (377 genes)."""
    return [str(g) for g in np.load(BASE_PANEL, allow_pickle=True)]


def gene_reindex(src_genes: list[str], dst_genes: list[str]) -> np.ndarray:
    """
    Column indices that reorder `src_genes` into `dst_genes` order.

    Every gene in `dst_genes` must be present in `src_genes`.
    """
    pos = {g: i for i, g in enumerate(src_genes)}
    return np.asarray([pos[g] for g in dst_genes], dtype=np.int64)


def ensure_dir(path: str) -> str:
    """Create the directory holding `path` and return `path`."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
