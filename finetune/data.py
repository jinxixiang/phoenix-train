"""
Dataset over the cached image conditions and their expression targets.

Image features are read lazily from the per-slide HDF5 caches (one chunk per
cell); counts are small enough to keep in memory and are turned into Phoenix
targets on the fly: ``x1 = (log1p(counts) - mean) / std``.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from common import cell_mask, feature_file

# -------------------------------------------------------------------------------


class CellDataset(Dataset):
    """
    Cells of a set of slides, as (image tokens, normalized expression) pairs.

    `part` selects the half of each slide to use ("train" or "val"). It only
    matters for the SPATCH cohorts, where a slide contributes its training folds
    to one dataset and its held-out fold to the other; for the slide-split
    cohorts every cell of the listed slides is used.
    """

    def __init__(self, cohort: str, slides: list[str], mean: np.ndarray, std: np.ndarray,
                 part: str = "train"):
        self.paths = [feature_file(cohort, s) for s in slides]
        self.handles: list[h5py.File | None] = [None] * len(self.paths)

        counts, index = [], []
        for slide, (sid, path) in enumerate(zip(slides, self.paths)):
            with h5py.File(path, "r") as fh:
                slide_counts = fh["counts"][:]
            mask = cell_mask(cohort, sid, part)
            rows = np.arange(len(slide_counts)) if mask is None else np.flatnonzero(mask)
            counts.append(slide_counts[rows])
            index += [(slide, int(i)) for i in rows]
        self.index = np.asarray(index, dtype=np.int64)

        expr = np.log1p(np.concatenate(counts).astype(np.float32))
        self.target = ((expr - mean[None]) / std[None]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        slide, local = self.index[i]
        if self.handles[slide] is None:
            self.handles[slide] = h5py.File(self.paths[slide], "r")
        feat = torch.from_numpy(self.handles[slide]["feat"][local])
        return feat, torch.from_numpy(self.target[i]).unsqueeze(-1)
