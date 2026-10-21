"""
Cache Phoenix image conditions for one slide.

For every cell centroid we crop a 112 um window from the WSI at native
resolution, resample it to 224 x 224 (Phoenix's 20x setting) and run it through
the frozen ViT-g/14 image encoder that ships inside `flow_model.pth`. The
resulting token sequence is the conditioning signal `c`; caching it once avoids
a 1.1B-parameter forward pass on every training epoch.

Usage:
    python extract_features.py --cohort lung_1 --slide NCBI856
"""

import argparse
import time

import h5py
import numpy as np
import timm
import tifffile
import torch
import torch.nn.functional as F
import zarr
from torch.utils.data import DataLoader, Dataset

from common import (
    BASE_CKPT,
    IMAGE_MEAN,
    IMAGE_STD,
    PATCH_PX,
    PATCH_UM,
    ensure_dir,
    feature_file,
    gene_reindex,
    load_panel,
    read_genes,
    slide_path,
    wsi_path,
)

# -------------------------------------------------------------------------------


def is_tiled(wsi_path: str) -> bool:
    """True if level 0 is stored as tiles, so windows can be read lazily."""
    with tifffile.TiffFile(wsi_path) as handle:
        return bool(handle.series[0].keyframe.is_tiled)


class PatchDataset(Dataset):
    """Cell-centered WSI crops, resampled and normalized for the image encoder."""

    def __init__(self, wsi_path: str, coords: np.ndarray, window: int,
                 in_memory: bool = False, workers: int = 4):
        self.wsi_path = wsi_path
        self.coords = coords
        self.window = window
        self.store = None
        self.mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGE_STD).view(3, 1, 1)

        # The SPATCH H&E images are single-level, LZW, *striped with one row per
        # strip*, so a lazy 507 px window costs 507 full-width row decodes --
        # ~75 MB of decode per cell, which does not finish. For those the whole
        # level 0 is decoded once here instead (2-6 GB); dataloader workers are
        # forked after this and share the pages read-only.
        self.image = None
        if in_memory:
            image = tifffile.imread(wsi_path, maxworkers=workers)
            if image.ndim != 3 or image.shape[2] < 3:
                raise ValueError(f"{wsi_path}: expected an RGB image, got {image.shape}")
            self.image = image[:, :, :3]

    def _level0(self):
        if self.image is not None:
            return self.image
        # Opened lazily so each dataloader worker gets its own file handle.
        if self.store is None:
            handle = tifffile.TiffFile(self.wsi_path)
            self.store = zarr.open(handle.series[0].aszarr(level=0), mode="r")
        return self.store

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, i: int):
        level0 = self._level0()
        height, width = level0.shape[:2]
        win = self.window
        x, y = int(self.coords[i, 0]), int(self.coords[i, 1])
        y0, x0 = y - win // 2, x - win // 2

        # Clip to the slide and pad the overhang with white, as for glass background.
        cy0, cy1 = max(0, y0), min(height, y0 + win)
        cx0, cx1 = max(0, x0), min(width, x0 + win)
        patch = np.full((win, win, 3), 255, dtype=np.uint8)
        if cy1 > cy0 and cx1 > cx0:
            patch[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] = level0[cy0:cy1, cx0:cx1, :3]

        img = torch.from_numpy(patch).permute(2, 0, 1).float().div_(255.0)
        img = F.interpolate(
            img.unsqueeze(0),
            size=(PATCH_PX, PATCH_PX),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        return img.clamp_(0, 1).sub_(self.mean).div_(self.std), i


def build_encoder(device: str) -> torch.nn.Module:
    """Instantiate the ViT-g/14 encoder and load its weights from the checkpoint."""
    model = timm.create_model(
        "vit_giant_patch14_reg4_dinov2",
        pretrained=False,
        img_size=PATCH_PX,
        num_classes=0,
        global_pool="token",
        init_values=1e-5,
        dynamic_img_size=False,
    )
    state = torch.load(BASE_CKPT, map_location="cpu", mmap=True)
    vision = {k[len("vision_model.") :]: v for k, v in state.items() if k.startswith("vision_model.")}
    missing, unexpected = model.load_state_dict(vision, strict=False)
    assert not unexpected, f"unexpected vision keys: {unexpected[:5]}"
    assert not missing, f"missing vision keys: {missing[:5]}"
    return model.eval().to(device)


# -------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--slide", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="only the first N cells (smoke test)")
    # Default: decide from the WSI's layout (striped level 0 -> decode it once).
    parser.add_argument("--in-memory", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    panel = load_panel(args.cohort)
    genes = panel["genes"]
    out_path = feature_file(args.cohort, args.slide)

    with h5py.File(slide_path(args.cohort, args.slide), "r") as fh:
        coords = fh["coords"][:]
        counts = fh["counts"][:]
        cell_id = fh["cell_id"][:]
        mpp = float(fh.attrs["native_mpp"])
        organ = str(fh.attrs["organ"])
    counts = counts[:, gene_reindex(read_genes(args.cohort, args.slide), genes)]
    assert counts.max() < 65535, "counts overflow uint16"

    if args.limit:
        coords, counts, cell_id = coords[: args.limit], counts[: args.limit], cell_id[: args.limit]
        out_path = out_path.replace(".h5", ".smoke.h5")

    window = int(round(PATCH_UM / mpp))
    device = "cuda"
    encoder = build_encoder(device)

    # Visit cells in raster order so neighbouring crops hit the same WSI tiles.
    order = np.lexsort((coords[:, 0], coords[:, 1]))
    wsi = wsi_path(args.cohort, args.slide)
    in_memory = (not is_tiled(wsi)) if args.in_memory is None else args.in_memory
    load_start = time.time()
    dataset = PatchDataset(wsi, coords[order], window,
                           in_memory=in_memory, workers=max(args.workers, 1))
    if in_memory:
        print(f"decoded {wsi} into memory "
              f"({dataset.image.nbytes / 1e9:.2f} GB, {time.time() - load_start:.0f} s)",
              flush=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False,
    )

    n_cells = len(coords)
    start = time.time()
    feat_shape = None
    with h5py.File(ensure_dir(out_path), "w") as out:
        feat_ds = None
        with torch.inference_mode():
            for images, sorted_idx in loader:
                with torch.autocast("cuda", dtype=torch.float16):
                    tokens = encoder.forward_features(images.to(device, non_blocking=True))
                tokens = tokens.float().cpu().numpy().astype(np.float16)
                if feat_ds is None:
                    n_tokens, dim = tokens.shape[1], tokens.shape[2]
                    feat_ds = out.create_dataset(
                        "feat",
                        shape=(n_cells, n_tokens, dim),
                        dtype=np.float16,
                        chunks=(1, n_tokens, dim),
                    )
                    feat_shape = (n_tokens, dim)
                # `sorted_idx` indexes the raster order; write back to slide order.
                for row, sidx in zip(tokens, sorted_idx.numpy()):
                    feat_ds[order[sidx]] = row

        out.create_dataset("counts", data=counts.astype(np.uint16))
        out.create_dataset("coords", data=coords)
        out.create_dataset("cell_id", data=cell_id)
        out.attrs.update(
            slide_id=args.slide,
            cohort=args.cohort,
            organ=organ,
            native_mpp=mpp,
            window_px=window,
            patch_px=PATCH_PX,
            patch_um=PATCH_UM,
            n_cells=n_cells,
            n_genes=len(genes),
        )

    elapsed = time.time() - start
    print(
        f"{args.slide}: {n_cells} cells, window {window}px @ {mpp:.4f} um/px, "
        f"{feat_shape[0]} tokens x {feat_shape[1]}d, {elapsed / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
