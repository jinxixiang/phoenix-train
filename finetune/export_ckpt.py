"""
Snapshot a mid-training Lightning checkpoint as a plain flow-model state dict.

`evaluate.py` loads `<run_dir>/flow_model_finetuned.pth`, which `train.py` only
writes when the schedule finishes. To score a run that is still training, copy
its `last.ckpt` weights into a *separate* run directory and evaluate that:

    python export_ckpt.py --cohort spatch_xenium_coad          # -> <cohort>_interim
    python evaluate.py --cohort spatch_xenium_coad --run-name spatch_xenium_coad_interim

The separate directory matters. Writing interim predictions into the real run
directory would make `slurm/spatch.sbatch` believe the cohort was already
evaluated and skip the final, full-schedule evaluation.

`source.json` records which epoch the snapshot came from, so an interim number
can never be mistaken for the final one.
"""

import argparse
import json
import os

import torch

from common import OUTPUTS

# -------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--suffix", default="_interim")
    parser.add_argument("--ckpt", default=None, help="default: <cohort>/last.ckpt")
    args = parser.parse_args()

    src = args.ckpt or f"{OUTPUTS}/runs/{args.cohort}/last.ckpt"
    out_dir = f"{OUTPUTS}/runs/{args.cohort}{args.suffix}"
    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(src, map_location="cpu")
    prefix = "model."
    state = {k[len(prefix):]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)}
    assert state, f"{src}: no '{prefix}*' keys in state_dict"
    assert len(state) == len(ckpt["state_dict"]), \
        "unexpected non-model keys in the checkpoint state_dict"

    torch.save(state, f"{out_dir}/flow_model_finetuned.pth")
    meta = {"cohort": args.cohort, "source": src, "epoch": int(ckpt["epoch"]),
            "global_step": int(ckpt["global_step"]), "n_tensors": len(state),
            "interim": True}
    with open(f"{out_dir}/source.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"{args.cohort}: epoch {meta['epoch']}, step {meta['global_step']}, "
          f"{len(state)} tensors -> {out_dir}/flow_model_finetuned.pth")


if __name__ == "__main__":
    main()
