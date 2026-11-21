"""Feature extraction for Emguard.

The model (``Virus_detect.keras``, trained by ``train_model.py``) uses summary
features computed from the EMBER 2018 *raw* features. This module computes the
same features from the bytes of an uploaded PE file, following the EMBER
feature definitions (https://github.com/elastic/ember).
"""

import re

import lief
import numpy as np

lief.logging.disable()

# Model input columns, in order.
FEATURE_NAMES = [
    "histogram_mean",
    "histogram_std",
    "byteentropy_mean",
    "byteentropy_std",
    "numstrings",
    "avg_length",
    "printables",
    "strings_entropy",
    "import_count",
    "size",
    "vsize",
    "has_debug",
    "exports_count",
    "imports_count",
]

_ALL_STRINGS = re.compile(rb"[\x20-\x7f]{5,}")


def byte_histogram(bytez):
    """EMBER ByteHistogram raw features: count of each byte value."""
    return np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)


def byte_entropy_histogram(bytez, step=1024, window=2048):
    """EMBER ByteEntropyHistogram raw features (16x16 counts, flattened)."""

    def entropy_bin_counts(block):
        c = np.bincount(block >> 4, minlength=16)
        p = c.astype(np.float32) / window
        wh = np.where(c)[0]
        h = np.sum(-p[wh] * np.log2(p[wh])) * 2
        hbin = int(h * 2)
        if hbin == 16:
            hbin = 15
        return hbin, c

    output = np.zeros((16, 16), dtype=np.int64)
    a = np.frombuffer(bytez, dtype=np.uint8)
    if a.shape[0] < window:
        hbin, c = entropy_bin_counts(a)
        output[hbin, :] += c
    else:
        blocks = np.lib.stride_tricks.sliding_window_view(a, window)[::step, :]
        for block in blocks:
            hbin, c = entropy_bin_counts(block)
            output[hbin, :] += c
    return output.flatten()


def string_features(bytez):
    """EMBER StringExtractor: numstrings, avlength, printables, entropy."""
    allstrings = _ALL_STRINGS.findall(bytez)
    if not allstrings:
        return 0, 0.0, 0, 0.0
    lengths = [len(s) for s in allstrings]
    shifted = np.frombuffer(b"".join(allstrings), dtype=np.uint8) - 0x20
    c = np.bincount(shifted, minlength=96)
    csum = c.sum()
    p = c.astype(np.float32) / csum
    wh = np.where(c)[0]
    entropy = float(np.sum(-p[wh] * np.log2(p[wh])))
    return len(allstrings), sum(lengths) / len(lengths), int(csum), entropy


def parse_pe(bytez):
    """Parse bytes as a PE file. Returns a lief PE Binary, or None if invalid."""
    try:
        # Pass a list of ints: lief.parse(bytes) treats bytes as a file path.
        return lief.PE.parse(list(bytez))
    except Exception:
        return None


def extract_features(bytez, binary=None):
    """Return a dict of the model features (FEATURE_NAMES) for the file bytes."""
    if binary is None:
        binary = parse_pe(bytez)

    hist = byte_histogram(bytez)
    entropy_hist = byte_entropy_histogram(bytez)
    numstrings, avlength, printables, strings_entropy = string_features(bytez)

    if binary is not None:
        vsize = binary.virtual_size
        has_debug = int(binary.has_debug)
        exports = len(binary.exported_functions)
        imports = len(binary.imported_functions)
        import_libs = len({lib.name for lib in binary.imports})
    else:
        vsize = has_debug = exports = imports = import_libs = 0

    return {
        "histogram_mean": float(np.mean(hist)),
        "histogram_std": float(np.std(hist)),
        "byteentropy_mean": float(np.mean(entropy_hist)),
        "byteentropy_std": float(np.std(entropy_hist)),
        "numstrings": numstrings,
        "avg_length": avlength,
        "printables": printables,
        "strings_entropy": strings_entropy,
        "import_count": import_libs,
        "size": len(bytez),
        "vsize": vsize,
        "has_debug": has_debug,
        "exports_count": exports,
        "imports_count": imports,
    }


def transform(x):
    """log1p transform applied before scaling (counts and sizes are very skewed)."""
    return np.log1p(np.clip(np.asarray(x, dtype=np.float64), 0, None))
