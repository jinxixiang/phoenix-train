"""
Dataset over the cached image conditions and their expression targets.

Image features are read lazily from the per-slide HDF5 caches (one chunk per
cell); targets are small enough to keep in memory:
``x1 = (log1p(CP100) - mean) / std`` on the cohort panel (`common.load_target`).
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from common import cell_mask, feature_file, load_panel, load_target

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

        genes = load_panel(cohort)["genes"]
        targets, index = [], []
        for slide, (sid, path) in enumerate(zip(slides, self.paths)):
            with h5py.File(path, "r") as fh:
                cell_id = fh["cell_id"][:]
            slide_target = load_target(cohort, sid, genes, cell_id)
            mask = cell_mask(cohort, sid, part)
            rows = np.arange(len(slide_target)) if mask is None else np.flatnonzero(mask)
            targets.append(slide_target[rows])
            index += [(slide, int(i)) for i in rows]
        self.index = np.asarray(index, dtype=np.int64)

        expr = np.concatenate(targets)
        self.target = ((expr - mean[None]) / std[None]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        slide, local = self.index[i]
        if self.handles[slide] is None:
            self.handles[slide] = h5py.File(self.paths[slide], "r")
        feat = torch.from_numpy(self.handles[slide]["feat"][local])
        return feat, torch.from_numpy(self.target[i]).unsqueeze(-1)
