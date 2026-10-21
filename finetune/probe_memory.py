"""
Measure peak GPU memory of a full training step for several micro-batch sizes.

Run 42396054 OOM'd at batch 256 on a 46 GB L40S, on the step where AdamW first
allocates its state. This probe reports what actually fits, so the micro-batch
and gradient-accumulation factor are chosen from evidence rather than guessed.

    python probe_memory.py --cohort breast_1 --batches 256 128 64
"""

import argparse

import torch
import torch.nn.functional as F

from model import build_model

# -------------------------------------------------------------------------------


def one_step(model, optimizer, batch, n_genes, device):
    """A single fwd + bwd + optimizer step with the training shapes."""
    cond = torch.randn(batch, 261, 1536, device=device)
    x_1 = torch.randn(batch, n_genes, 1, device=device)
    x_0 = torch.randn_like(x_1)
    t = torch.rand(batch, device=device)
    x_t = (1.0 - t[:, None, None]) * x_0 + t[:, None, None] * x_1

    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = F.mse_loss(model(x_t, t, cond), x_1 - x_0)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--grad-checkpoint", action="store_true")
    args = parser.parse_args()

    device = "cuda"
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"{name}, {total:.1f} GiB total, grad_checkpoint={args.grad_checkpoint}")

    for batch in args.batches:
        # Rebuild per batch so a failure cannot contaminate the next measurement.
        model, panel = build_model(args.cohort, grad_checkpoint=args.grad_checkpoint)
        model = model.to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            # Three steps: step 1 allocates activations, step 2 adds AdamW state.
            for _ in range(3):
                one_step(model, optimizer, batch, panel["n_genes"], device)
            peak = torch.cuda.max_memory_allocated() / 2**30
            reserved = torch.cuda.max_memory_reserved() / 2**30
            print(f"  batch {batch:4d}: OK   peak_alloc {peak:6.2f} GiB  peak_reserved {reserved:6.2f} GiB")
        except torch.cuda.OutOfMemoryError:
            print(f"  batch {batch:4d}: OOM  (reserved {torch.cuda.max_memory_reserved()/2**30:.2f} GiB)")
        del model, optimizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
