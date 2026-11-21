"""Convert EMBER 2018 feature files into model input arrays.

Reads ember2018/train_features_*.jsonl and ember2018/test_features.jsonl,
keeps labeled samples (0 = benign, 1 = malicious), turns each record into a
vector with features.vectorize, and writes data/train.npz and data/test.npz.

Usage:
    python prepare_data.py path/to/ember2018
The data comes from https://ember.elastic.co/ember_dataset_2018_2.tar.bz2
"""

import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

from features import vectorize


def convert(path):
    X, y = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["label"] in (0, 1):
                X.append(vectorize(d))
                y.append(d["label"])
    print(f"{os.path.basename(path)}: {len(y)} labeled samples", flush=True)
    return np.stack(X), np.array(y, dtype=np.int8)


def main(ember_dir):
    train_files = sorted(glob.glob(os.path.join(ember_dir, "train_features_*.jsonl")))
    test_file = os.path.join(ember_dir, "test_features.jsonl")
    if not train_files or not os.path.exists(test_file):
        sys.exit(f"EMBER feature files not found in {ember_dir}")

    with Pool() as pool:
        results = pool.map(convert, train_files + [test_file])

    os.makedirs("data", exist_ok=True)
    X_train = np.concatenate([r[0] for r in results[:-1]])
    y_train = np.concatenate([r[1] for r in results[:-1]])
    np.savez("data/train.npz", X=X_train, y=y_train)
    np.savez("data/test.npz", X=results[-1][0], y=results[-1][1])
    print(f"Saved data/train.npz ({len(y_train)}) and data/test.npz ({len(results[-1][1])})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
