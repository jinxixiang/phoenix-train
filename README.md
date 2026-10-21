# phoenix-train

Fine-tuning and evaluation pipeline for **Phoenix** (Tran et al., *Pan-cancer
virtual spatial transcriptomics from routine histology with Phoenix*, bioRxiv
2026) on new single-cell spatial transcriptomics cohorts and gene panels.

This is the code used to produce the Phoenix baseline of the STORM single-cell
benchmark. The released Phoenix repository ships a trainer only for its optional
MLP-Mixer autoencoder, which no released checkpoint uses; this repository adds
the missing piece, a conditional flow-matching trainer for the flow transformer,
plus feature caching, new-panel adaptation, ODE sampling and benchmark scoring.
Nothing in Phoenix is modified: the model class, the frozen image encoder, the
checkpoint, the preprocessing conventions and the scheduler are imported from
the `phoenix` package.

## What is trained, and how

| Component | Treatment |
|---|---|
| Image encoder (ViT-g/14 with registers, 1.13 B params, inside `flow_model.pth`) | Frozen. Features of a 224 px patch at 0.5 µm/px (112 µm field of view) centred on each cell are cached once (`extract_features.py`). |
| Flow transformer (68.8 M params) | Trained in full with conditional flow matching: `x_t = (1-t) x_0 + t x_1`, `x_0 ~ N(0, I)`, L2 regression of the velocity `x_1 - x_0`. Warm-started from the released `nest/multi` checkpoint. AdamW, lr 1e-4, warm-up + cosine, batch 256, 30 epochs, grad clip 1.0, bf16. |
| Autoencoder / decoder | Identity, as in every released (`discrete`, `d_genes = 1`) checkpoint. |
| Gene panel | Phoenix identifies genes by position. For a new panel the pretrained embedding row of every gene shared with the multi-tissue panel is kept and the remaining rows are resampled from the distribution of the pretrained rows (`model.py`). |
| Targets | `log1p(raw counts)`, z-scored per gene with training-split statistics. No library-size normalisation, per the Phoenix preprint. |
| Inference | Integrate the learned flow from t = 0 to 1 with a fixed 50-step Heun solver, one sample per cell (`evaluate.py`). The release uses adaptive DOPRI5 at atol = rtol = 0.1. |

## Installation

```bash
pip install git+https://github.com/peng-lab/phoenix.git
pip install -r requirements.txt
```

Download the base checkpoint and panel from the Phoenix authors' Hugging Face
repository (`peng-lab/phoenix`, PolyForm Noncommercial licence) into `assets/`:

```
assets/weights/nest/multi/flow_model.pth      # weights/flow/nest/multi/cell/20x/discrete/flow_model.pth
assets/panels/xenium_human_multi.npy
```

## Data format

A cohort is a directory of packed slides plus a split file, under
`$PHOENIX_COHORT_DIR` (default `data/finetune_sc`):

```
finetune_sc/
├── splits.json                   {"cohorts": {"<cohort>": {"train": [slide, ...], "val": [slide, ...]}}}
└── slides/<cohort>/<slide>.h5
```

Each slide HDF5 file contains, for N cells and G panel genes:

| key | shape / type | meaning |
|---|---|---|
| `coords` | (N, 2) float | cell centroid (x, y) in level-0 pixels of the H&E WSI |
| `counts` | (N, G) int | raw transcript counts |
| `cell_id` | (N,) str | cell identifier |
| `genes` | (G,) str | gene symbols, column order of `counts` |
| `fold` | (N,) int8 | optional, spatial fold; fold 4 is held out for single-slide cohorts |
| attrs `native_mpp`, `organ` | float, str | WSI resolution and tissue |

Slide-split cohorts read the WSI from `$PHOENIX_WSI_DIR/<slide>.tif`;
single-slide SPATCH cohorts read `$PHOENIX_SPATCH_DIR/<Platform>/<TUMOUR>/HE.tif`
and split cells by `fold`. Cohort names and their split scheme are registered
in `finetune/common.py` (`SLIDE_SPLIT_COHORTS`, `SPATCH_COHORTS`); add new
cohorts there.

## Pipeline

```bash
cd finetune
python build_panels.py <cohort>                      # panel + warm-start map  -> outputs/panels/<cohort>.json
python build_stats.py  <cohort>                      # per-gene mean/std (train only)
python extract_features.py --cohort <cohort> --slide <slide>   # once per slide, GPU, ~20 GB per 25k cells
python train.py    --cohort <cohort> --epochs 30     # conditional flow matching
python evaluate.py --cohort <cohort>                 # ODE sampling on held-out cells -> outputs/runs/<cohort>/
python compute_pcc.py --cohort <cohort>              # Pearson metrics from the saved predictions
```

`slurm/` holds the SLURM launchers used for the paper (`extract.sbatch` array
per slide, `train.sbatch` train + evaluate, `spatch.sbatch` extract → train →
evaluate for several single-slide cohorts, resumable).

Utilities: `export_ckpt.py` scores a run that is still training from its latest
checkpoint into a separate directory; `verify_features.py` checks cached
feature files for completeness; `check_grad_checkpoint.py` verifies that
activation checkpointing is gradient-equivalent; `probe_memory.py` measures the
peak memory of a training step per batch size.

### Scoring against ST-Bench

`aggregate_stbench_sc.py` scores the predictions with the ST-Bench single-cell
metric (per-gene Pearson, Spearman and SSIM across cells on `log1p` of the
benchmark ground truth, pooled over the test slides of each cohort) and asserts
that prediction rows align with the ground-truth cells. Point it at an ST-Bench
checkout with `STBENCH_SC_GT_ROOT`, `STBENCH_SC_IDS` and `STBENCH_SC_OUT`.

Note that the ST-Bench ground truth is library-size normalised while Phoenix
predicts `log1p(raw counts)`; the predictions are scored as they come out of the
model (`--pred-scale raw`). Renormalising them (`--pred-scale cp100`) is
available but scores lower.

## Environment variables

| variable | default | meaning |
|---|---|---|
| `PHOENIX_TRAIN_ROOT` | repository root | base for the defaults below |
| `PHOENIX_COHORT_DIR` | `$ROOT/data/finetune_sc` | packed cohorts and `splits.json` |
| `PHOENIX_WSI_DIR` | `$ROOT/data/wsis` | WSIs of slide-split cohorts |
| `PHOENIX_SPATCH_DIR` | `$ROOT/data/SPATCH` | SPATCH WSIs |
| `PHOENIX_ASSETS` | `$ROOT/assets` | base checkpoint and panel |
| `PHOENIX_OUTPUTS` | `$ROOT/outputs` | panels, cached features, runs |
| `PHOENIX_SRC` | unset | use a Phoenix source checkout instead of the installed package |

## Reproducing the benchmark

For the STORM single-cell benchmark one model was fine-tuned per cohort (breast
and lung Xenium from HEST-1k, and the twelve SPATCH tumour × platform cohorts),
30 epochs each, final-epoch weights evaluated with no checkpoint selection on
held-out data. A 25k-cell cohort takes about two hours on one 48 GB GPU
including feature extraction and sampling.

## Licence

The code in this repository is released under the MIT licence. It depends on
the `phoenix` package and weights, which are distributed by their authors under
the PolyForm Noncommercial License 1.0.0; those terms apply to any use of the
Phoenix model.
