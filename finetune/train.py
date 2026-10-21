"""
Finetune Phoenix on a cohort with conditional flow matching.

The training objective is the one described in the preprint (Methods:
Generative modeling): a linear interpolant between noise and data,

    x_t = (1 - t) * x_0 + t * x_1,        x_0 ~ N(0, I),  t ~ U[0, 1]

whose conditional vector field is the constant `x_1 - x_0`, regressed with an
L2 loss. The image encoder stays frozen (its features are precomputed), so the
68.8M-parameter flow transformer is trained in full.

Usage:
    python train.py --cohort lung_1 --epochs 30
"""

import argparse
import json
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from common import OUTPUTS, load_splits, stats_file
from data import CellDataset
from model import build_model, load_stats

from phoenix.trainers.mixer_trainer import WarmupCosineAnnealingLR

# -------------------------------------------------------------------------------


class FlowMatchingTrainer(pl.LightningModule):
    """Conditional flow matching objective around the Phoenix flow transformer."""

    def __init__(self, model, max_lr, warmup_steps, total_steps, weight_decay=0.1):
        super().__init__()
        self.model = model
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.weight_decay = weight_decay

    def shared_step(self, batch):
        """Regress the conditional velocity field at a random point on the path."""
        cond, x_1 = batch
        cond = cond.float()
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.shape[0], device=x_1.device)
        x_t = (1.0 - t[:, None, None]) * x_0 + t[:, None, None] * x_1
        return F.mse_loss(self.model(x_t, t, cond), x_1 - x_0)

    def training_step(self, batch, _):
        loss = self.shared_step(batch)
        self.log("loss/train", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        loss = self.shared_step(batch)
        self.log("loss/val", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.max_lr,
            betas=(0.9, 0.95),
            eps=1e-08,
            weight_decay=self.weight_decay,
        )
        scheduler = WarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            max_lr=self.max_lr,
            start_lr=0.0,
            final_lr=self.max_lr / 10,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# -------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run-name", default=None)
    # Batch 256 without checkpointing needs ~50 GiB and OOMs on a 46 GB
    # L40S/A40 (measured: 38.1 GiB at batch 192, OOM at 256). Checkpointing the
    # transformer blocks brings batch 256 down to 11.3 GiB, so the intended
    # effective batch and LR schedule are kept intact. Verified to be
    # gradient-equivalent by `check_grad_checkpoint.py`.
    parser.add_argument(
        "--grad-checkpoint", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    pl.seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("high")
    run = args.run_name or args.cohort
    run_dir = f"{OUTPUTS}/runs/{run}"
    os.makedirs(run_dir, exist_ok=True)

    mean, std = load_stats(stats_file(args.cohort))
    mean, std = mean.numpy().reshape(-1), std.numpy().reshape(-1)
    split = load_splits(args.cohort)
    train_set = CellDataset(args.cohort, split["train"], mean, std, part="train")
    val_set = CellDataset(args.cohort, split["val"], mean, std, part="val")
    print(f"{args.cohort}: {len(train_set)} train cells, {len(val_set)} val cells")

    common = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **common)

    model, panel = build_model(args.cohort, grad_checkpoint=args.grad_checkpoint)
    steps = len(train_loader) * args.epochs
    module = FlowMatchingTrainer(
        model, max_lr=args.lr, warmup_steps=min(2000, steps // 10), total_steps=steps
    )

    with open(f"{run_dir}/config.json", "w") as fh:
        json.dump(
            {**vars(args), "n_genes": panel["n_genes"], "total_steps": steps,
             "train_cells": len(train_set), "val_cells": len(val_set)},
            fh,
            indent=1,
        )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        default_root_dir=run_dir,
        logger=CSVLogger(run_dir, name="", version=""),
        callbacks=[
            ModelCheckpoint(dirpath=run_dir, filename="best", monitor="loss/val", save_last=True),
            LearningRateMonitor(logging_interval="step"),
        ],
        log_every_n_steps=50,
        gradient_clip_val=1.0,
    )
    # Owner-partition jobs are preemptable, so always resume if a run is present.
    resume = f"{run_dir}/last.ckpt"
    resume = resume if os.path.exists(resume) else None
    if resume:
        print(f"resuming from {resume}")
    trainer.fit(module, train_loader, val_loader, ckpt_path=resume)

    torch.save(module.model.state_dict(), f"{run_dir}/flow_model_finetuned.pth")
    print(f"saved {run_dir}/flow_model_finetuned.pth")


if __name__ == "__main__":
    main()
