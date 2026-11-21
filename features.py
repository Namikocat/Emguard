"""Feature extraction for Emguard.

Two steps, shared by training and the app so both see identical features:

1. ``raw_features(bytez)`` computes the subset of EMBER 2018 *raw* features the
   model uses (byte histogram, byte-entropy histogram, strings, general info),
   in the same format as the records in EMBER's ``*_features.jsonl`` files,
   following https://github.com/elastic/ember.
2. ``vectorize(raw)`` turns such a record (from a file or from the dataset)
   into the model's input vector.

Only features computed from the file bytes or parsed PE structure are used,
never dataset metadata such as the first-seen date or AV family label.
"""

import re

import lief
import numpy as np

lief.logging.disable()

_ALL_STRINGS = re.compile(rb"[\x20-\x7f]{5,}")
_PATHS = re.compile(rb"c:\\", re.IGNORECASE)
_URLS = re.compile(rb"https?://", re.IGNORECASE)
_REGISTRY = re.compile(rb"HKEY_")
_MZ = re.compile(rb"MZ")

STRING_COUNTS = ["numstrings", "avlength", "printables", "entropy", "paths", "urls", "registry", "MZ"]
GENERAL = ["size", "vsize", "has_debug", "exports", "imports", "has_relocations",
           "has_resources", "has_signature", "has_tls", "symbols"]

FEATURE_NAMES = (
    [f"histogram_{i}" for i in range(256)]
    + [f"byteentropy_{i}" for i in range(256)]
    + [f"strings_{k}" for k in STRING_COUNTS]
    + [f"printabledist_{i}" for i in range(96)]
    + [f"general_{k}" for k in GENERAL]
)

# Count/size-like columns that get log1p before scaling (they are very skewed).
LOG_FEATURES = [f"strings_{k}" for k in STRING_COUNTS if k != "entropy"] + [
    f"general_{k}" for k in ("size", "vsize", "exports", "imports", "symbols")
]


def parse_pe(bytez):
    """Parse bytes as a PE file. Returns a lief PE Binary, or None if invalid."""
    try:
        # Pass a list of ints: lief.parse(bytes) treats bytes as a file path.
        return lief.PE.parse(list(bytez))
    except Exception:
        return None


def _byte_histogram(a):
    return np.bincount(a, minlength=256)


def _byte_entropy_histogram(a, step=1024, window=2048):
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
    if a.shape[0] < window:
        hbin, c = entropy_bin_counts(a)
        output[hbin, :] += c
    else:
        blocks = np.lib.stride_tricks.sliding_window_view(a, window)[::step, :]
        for block in blocks:
            hbin, c = entropy_bin_counts(block)
            output[hbin, :] += c
    return output.flatten()


def _strings(bytez):
    allstrings = _ALL_STRINGS.findall(bytez)
    if allstrings:
        avlength = sum(len(s) for s in allstrings) / len(allstrings)
        shifted = np.frombuffer(b"".join(allstrings), dtype=np.uint8) - 0x20
        c = np.bincount(shifted, minlength=96)
        csum = int(c.sum())
        p = c.astype(np.float32) / csum
        wh = np.where(c)[0]
        entropy = float(np.sum(-p[wh] * np.log2(p[wh])))
    else:
        avlength, c, csum, entropy = 0, np.zeros(96, dtype=np.int64), 0, 0.0
    return {
        "numstrings": len(allstrings),
        "avlength": avlength,
        "printabledist": c.tolist(),
        "printables": csum,
        "entropy": entropy,
        "paths": len(_PATHS.findall(bytez)),
        "urls": len(_URLS.findall(bytez)),
        "registry": len(_REGISTRY.findall(bytez)),
        "MZ": len(_MZ.findall(bytez)),
    }


def _general(bytez, binary):
    if binary is None:
        return {"size": len(bytez), **{k: 0 for k in GENERAL if k != "size"}}
    return {
        "size": len(bytez),
        "vsize": binary.virtual_size,
        "has_debug": int(binary.has_debug),
        "exports": len(binary.exported_functions),
        "imports": len(binary.imported_functions),
        "has_relocations": int(binary.has_relocations),
        "has_resources": int(binary.has_resources),
        "has_signature": int(binary.has_signatures),
        "has_tls": int(binary.has_tls),
        "symbols": len(binary.symbols),
    }


def raw_features(bytez, binary=None):
    """EMBER-format raw features for the given file bytes."""
    if binary is None:
        binary = parse_pe(bytez)
    a = np.frombuffer(bytez, dtype=np.uint8)
    return {
        "histogram": _byte_histogram(a).tolist(),
        "byteentropy": _byte_entropy_histogram(a).tolist(),
        "strings": _strings(bytez),
        "general": _general(bytez, binary),
    }


def _normalize(counts):
    counts = np.asarray(counts, dtype=np.float32)
    total = counts.sum()
    return counts / total if total > 0 else counts


def vectorize(raw):
    """Turn an EMBER-format raw feature record into the model input vector."""
    s, g = raw["strings"], raw["general"]
    return np.concatenate([
        _normalize(raw["histogram"]),
        _normalize(raw["byteentropy"]),
        np.array([s[k] for k in STRING_COUNTS], dtype=np.float32),
        _normalize(s["printabledist"]),
        np.array([g[k] for k in GENERAL], dtype=np.float32),
    ]).astype(np.float32)


def summary(raw):
    """Small human-readable summary of a file's features for display."""
    s, g = raw["strings"], raw["general"]
    return {**{k: g[k] for k in GENERAL}, **{k: s[k] for k in STRING_COUNTS}}
