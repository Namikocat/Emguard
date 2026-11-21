"""Convert EMBER2024 PE feature files into model input arrays.

Download the PE parts of EMBER2024 from https://huggingface.co/datasets/joyce8/EMBER2024
(Win32_train.zip, Win64_train.zip, Dot_Net_train.zip, the matching *_test.zip files
and challenge.zip) into one folder, then run:

    python prepare_data.py path/to/zip/folder

Each weekly .jsonl inside the zips is streamed (not unpacked to disk), labeled
samples are vectorized with features.vectorize, and the results are written to
data/train.npz, data/test.npz and data/challenge.npz.

Win32 has 3x more files per week than the other types, so only every third
Win32 record is kept (WIN32_KEEP_EVERY) to keep the arrays a manageable size.
"""

import os
import subprocess
import sys
import zipfile
from multiprocessing import Pool

import numpy as np

try:
    import orjson as json_lib
    loads = json_lib.loads
except ImportError:  # pragma: no cover
    import json as json_lib
    loads = json_lib.loads

from features import vectorize

WIN32_KEEP_EVERY = 3
SPLITS = {
    "train": ["Win32_train.zip", "Win64_train.zip", "Dot_Net_train.zip"],
    "test": ["Win32_test.zip", "Win64_test.zip", "Dot_Net_test.zip"],
    "challenge": ["challenge.zip"],
}


def convert_member(job):
    zip_path, member, keep_every = job
    X, y = [], []
    proc = subprocess.Popen(["unzip", "-p", zip_path, member], stdout=subprocess.PIPE, bufsize=1 << 20)
    for i, line in enumerate(proc.stdout):
        if i % keep_every:
            continue
        d = loads(line)
        if d.get("label") not in (0, 1):
            continue
        X.append(vectorize(d))
        y.append(d["label"])
    proc.wait()
    print(f"{os.path.basename(zip_path)}:{member}: {len(y)} samples", flush=True)
    if not y:
        return None
    return np.stack(X), np.array(y, dtype=np.int8)


def main(folder):
    os.makedirs("data", exist_ok=True)
    for split, zips in SPLITS.items():
        jobs = []
        for name in zips:
            path = os.path.join(folder, name)
            if not os.path.exists(path):
                sys.exit(f"Missing {path}")
            keep = WIN32_KEEP_EVERY if name.startswith("Win32") else 1
            with zipfile.ZipFile(path) as z:
                jobs += [(path, m, keep) for m in z.namelist() if m.endswith(".jsonl")]
        with Pool() as pool:
            results = [r for r in pool.imap_unordered(convert_member, jobs) if r is not None]
        X = np.concatenate([r[0] for r in results])
        y = np.concatenate([r[1] for r in results])
        np.savez(f"data/{split}.npz", X=X, y=y)
        print(f"Saved data/{split}.npz: {len(y)} samples ({int(y.sum())} malicious)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
