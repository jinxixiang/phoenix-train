"""Check that cached feature files are complete (see common.features_complete).

    python verify_features.py breast_1 lung_1
"""
import sys, glob, os, h5py, numpy as np
from common import OUTPUTS
bad = []
for cohort in sys.argv[1:]:
    files = sorted(glob.glob(f"{OUTPUTS}/features/{cohort}/*.h5"))
    print(f"--- {cohort}: {len(files)} files")
    for p in files:
        b = os.path.basename(p)
        try:
            with h5py.File(p, "r") as f:
                n = int(f.attrs["n_cells"]); sh = f["feat"].shape
                ok = sh[0] == n and f["counts"].shape[0] == n
                last = f["feat"][n - 1].astype(np.float32)
                fin = bool(np.isfinite(last).all() and np.abs(last).sum() > 0)
                print(f"{b:12s} feat={sh} n={n} rows_ok={ok} lastrow_ok={fin}")
                if not (ok and fin):
                    bad.append(b)
        except Exception as e:
            print(f"{b:12s} ERROR {type(e).__name__}: {e}")
            bad.append(b)
print("BAD:", bad if bad else "none")
