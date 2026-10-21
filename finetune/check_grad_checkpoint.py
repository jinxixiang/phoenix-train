"""
Verify that gradient checkpointing does not change the gradients.

`flow_simple.py` checkpoints with `use_reentrant=True`, which silently produces
no parameter gradients when no input to the checkpointed segment requires grad.
Here the segments are fed by the trainable `x_projection` / `c_projection`, so
it should be safe -- this asserts that rather than assuming it.

    python check_grad_checkpoint.py --cohort lung_1
"""

import argparse

import torch
import torch.nn.functional as F

from model import build_model

# -------------------------------------------------------------------------------


def grads_for(cohort: str, grad_checkpoint: bool, batch: int, n_genes: int):
    """Loss and parameter gradients for one identical step."""
    model, panel = build_model(cohort, grad_checkpoint=grad_checkpoint)
    model.train()

    # Identical inputs regardless of the checkpointing flag.
    generator = torch.Generator().manual_seed(1234)
    cond = torch.randn(batch, 261, 1536, generator=generator)
    x_1 = torch.randn(batch, panel["n_genes"], 1, generator=generator)
    x_0 = torch.randn(batch, panel["n_genes"], 1, generator=generator)
    t = torch.rand(batch, generator=generator)
    x_t = (1.0 - t[:, None, None]) * x_0 + t[:, None, None] * x_1

    loss = F.mse_loss(model(x_t, t, cond), x_1 - x_0)
    loss.backward()
    grads = {n: (p.grad.clone() if p.grad is not None else None) for n, p in model.named_parameters()}
    return loss.item(), grads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(0)
    loss_off, grads_off = grads_for(args.cohort, False, args.batch, None)
    torch.manual_seed(0)
    loss_on, grads_on = grads_for(args.cohort, True, args.batch, None)

    print(f"loss  no-ckpt {loss_off:.8f}   ckpt {loss_on:.8f}   diff {abs(loss_off-loss_on):.2e}")

    missing = [n for n, g in grads_on.items() if g is None]
    zero = [n for n, g in grads_on.items() if g is not None and g.abs().sum() == 0]
    worst, worst_name = 0.0, ""
    for name, g_off in grads_off.items():
        g_on = grads_on[name]
        if g_off is None or g_on is None:
            continue
        denom = g_off.abs().max().clamp(min=1e-12)
        rel = ((g_on - g_off).abs().max() / denom).item()
        if rel > worst:
            worst, worst_name = rel, name

    n_params = len(grads_on)
    print(f"params={n_params}  grads None under ckpt: {len(missing)}  all-zero: {len(zero)}")
    if missing:
        print("  MISSING:", missing[:10])
    if zero:
        print("  ZERO:", zero[:10])
    print(f"worst relative grad diff: {worst:.3e} ({worst_name})")
    ok = not missing and len(zero) == 0 and worst < 1e-3
    print("VERDICT:", "checkpointing is gradient-equivalent" if ok else "MISMATCH -- do not use")


if __name__ == "__main__":
    main()
