"""
Build a Phoenix flow transformer for a cohort panel, warm-started from `nest/multi`.

Gene identity in Phoenix is positional: token `i` is embedded with row `i` of
`px_embedding` and decoded by a head that is shared across tokens. Everything
except those rows is therefore panel-agnostic and transfers unchanged. For a new
panel we keep the pretrained row of every gene that also appears in the
multi-tissue panel and resample the remaining rows from the distribution of the
pretrained rows.

The image encoder is not attached: conditioning features are precomputed by
`extract_features.py`, so the trainable model is the 68.8M-parameter flow
transformer alone.
"""

import numpy as np
import torch

from common import BASE_CKPT, load_panel

from phoenix.models.flow_simple import FlowTransformerConfig, FlowTransformerModel

# -------------------------------------------------------------------------------

# Matches the released `nest/multi/cell/20x/discrete` checkpoint.
FLOW_CONFIG = dict(
    d_genes=1,
    d_image=1536,
    d_model=512,
    d_cross=512,
    n_heads=8,
    n_layers=8,
    qkv_bias=False,
    ffn_bias=False,
    ffn_mult=4,
    attn_drop=0.0,
    proj_drop=0.0,
    n_classes=0,
    cls_drop=0.1,
)

# -------------------------------------------------------------------------------


def remap_gene_embedding(pretrained: torch.Tensor, warm_rows: list[int], n_base: int) -> torch.Tensor:
    """
    Reorder `px_embedding` rows onto a new panel.

    `warm_rows[i]` is the pretrained row holding gene `i` of the new panel, or -1
    when the gene is absent from the pretrained panel. Fresh rows are drawn from
    a Gaussian matched to the mean and standard deviation of the pretrained rows
    that were actually trained (the first `n_base`).
    """
    source = pretrained.clone()
    trained = source[:n_base]
    generator = torch.Generator().manual_seed(0)
    new = source.clone()
    n_fresh = 0
    for i, row in enumerate(warm_rows):
        if row >= 0:
            new[i] = source[row]
        else:
            new[i] = torch.normal(
                trained.mean(), trained.std(), size=(source.shape[1],), generator=generator
            )
            n_fresh += 1
    return new, n_fresh


def build_model(cohort: str, checkpoint: str | None = None, grad_checkpoint: bool = False):
    """
    Instantiate the flow transformer for `cohort`.

    With `checkpoint=None` the model is warm-started from the pretrained
    `nest/multi` weights; otherwise the given finetuned checkpoint is loaded and
    no remapping happens.
    """
    panel = load_panel(cohort)
    config = FlowTransformerConfig(**FLOW_CONFIG, checkpoint=grad_checkpoint)
    model = FlowTransformerModel(config, vision_model=None)

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
        return model, panel

    state = torch.load(BASE_CKPT, map_location="cpu", mmap=True)
    state = {k: v.clone() for k, v in state.items() if not k.startswith("vision_model.")}
    state["px_embedding.weight"], n_fresh = remap_gene_embedding(
        state["px_embedding.weight"], panel["warm_start_rows"], panel["n_base_panel"]
    )
    model.load_state_dict(state, strict=True)
    print(
        f"warm-started {cohort}: {panel['n_genes']} genes, "
        f"{panel['n_genes'] - n_fresh} pretrained rows reused, {n_fresh} resampled"
    )
    return model, panel


def load_stats(path: str, device: str = "cpu"):
    """Load the per-gene mean/std table as `(1, n_genes, 1)` tensors."""
    table = np.load(path)
    mean = torch.tensor(table["mean"], dtype=torch.float32, device=device).view(1, -1, 1)
    std = torch.tensor(table["std"], dtype=torch.float32, device=device).view(1, -1, 1)
    return mean, std
