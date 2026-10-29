# phoenix-train

Fine-tuning and evaluation for **Phoenix** (Tran et al., bioRxiv 2026) on new
single-cell spatial transcriptomics cohorts. This produced the Phoenix baseline
of the STORM single-cell benchmark.

The Phoenix release has no trainer for its flow transformer. This repository
adds one (conditional flow matching). Phoenix itself is vendored unmodified in
`src/phoenix`.

## Method

- **Image encoder** (ViT-g/14, frozen): features of a 224 px patch at 0.5 µm/px
  centred on each cell, cached once.
- **Flow transformer** (68.8 M params): fully trained by conditional flow
  matching, warm-started from `nest/multi`. AdamW, lr 1e-4, cosine, batch 256,
  30 epochs, bf16.
- **New gene panels**: embeddings of genes shared with the pretrained panel are
  kept; the rest are resampled from the pretrained distribution.
- **Panel**: the cohort's ST-Bench HVG-200 list (`STBENCH_SC_HVG`); `build_panels.py`
  asserts exactly 200 genes, all measured on every slide.
- **Targets**: `log1p(CP100)` (counts scaled to 100 per cell over the full pack
  panel, as the ST-Bench ground truth), z-scored per gene on the training split.
- **Inference**: the released `FlowPipeline` (DOPRI5, atol=rtol=0.1, clip at 0),
  fixed seed.

## Setup

```bash
pip install -r requirements.txt
```

Download from Hugging Face (`peng-lab/phoenix`) into `assets/`:

```
assets/weights/nest/multi/flow_model.pth   # weights/flow/nest/multi/cell/20x/discrete/flow_model.pth
assets/panels/xenium_human_multi.npy
```

## Data

```
data/finetune_sc/
├── splits.json              {"cohorts": {"<cohort>": {"train": [...], "val": [...]}}}
└── slides/<cohort>/<slide>.h5
```

Each slide `.h5` holds `coords` (N×2, level-0 px), `counts` (N×G raw),
`cell_id`, `genes`, optional `fold`, and attrs `native_mpp`, `organ`.
Register new cohorts in `finetune/common.py`. Paths are set by `PHOENIX_*`
environment variables (see `finetune/common.py`).

## Run

```bash
cd finetune
python build_panels.py <cohort>
python build_stats.py  <cohort>
python extract_features.py --cohort <cohort> --slide <slide>   # per slide, GPU
python train.py    --cohort <cohort> --epochs 30               # resumes from last.ckpt
python evaluate.py --cohort <cohort>
python compute_pcc.py --cohort <cohort>
```

`aggregate_stbench_sc.py` scores predictions with the ST-Bench metrics
(Pearson, Spearman, SSIM). Set `STBENCH_SC_GT_ROOT`, `STBENCH_SC_IDS`,
`STBENCH_SC_HVG` and `STBENCH_SC_OUT`.

A 25k-cell cohort takes about two hours on one 48 GB GPU.

## Licence

`finetune/` is MIT. The vendored `src/phoenix` and the Phoenix weights are
under the PolyForm Noncommercial License 1.0.0.
