"""Train the Emguard model and save it with its preprocessing parameters.

Produces:
    Virus_detect.keras  - the neural network (same 64-32-2 architecture as the notebook)
    scaler.json         - feature names, log transform flag and StandardScaler mean/scale

Why this replaces the notebook's model:
  * The notebook never saved its StandardScaler, so the app could not scale inputs.
  * The notebook model used 'appeared' (first-seen date) and 'avclass' (the AV family
    name, which leaks the label). An uploaded file has neither; without them that
    model scored ~55% on benign vs malicious.
  * The notebook trained on unlabeled samples (label -1) as if they were malicious.

This script uses only features computed from the file bytes (see features.py),
trains on labeled samples only, and applies log1p before scaling because the
count/size features are heavily skewed.

Usage (needs pandas and scikit-learn in addition to requirements.txt):
    python train_model.py path/to/ember2018/train_features_1.jsonl
The data comes from https://ember.elastic.co/ember_dataset_2018_2.tar.bz2
"""

import json
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, transform


def load_rows(path):
    """Compute the FEATURE_NAMES columns from EMBER raw feature records."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["label"] not in (0, 1):
                continue  # skip unlabeled samples
            s, g = d["strings"], d["general"]
            rows.append({
                "label": d["label"],
                "histogram_mean": np.mean(d["histogram"]),
                "histogram_std": np.std(d["histogram"]),
                "byteentropy_mean": np.mean(d["byteentropy"]),
                "byteentropy_std": np.std(d["byteentropy"]),
                "numstrings": s["numstrings"],
                "avg_length": s["avlength"],
                "printables": s["printables"],
                "strings_entropy": s["entropy"],
                "import_count": len(d["imports"]),
                "size": g["size"],
                "vsize": g["vsize"],
                "has_debug": g["has_debug"],
                "exports_count": g["exports"],
                "imports_count": g["imports"],
            })
    return pd.DataFrame(rows)


def main(path):
    df = load_rows(path)
    print(f"Loaded {len(df)} labeled samples: {df['label'].value_counts().to_dict()}")

    X = transform(df[FEATURE_NAMES].to_numpy(dtype=np.float64))
    y = df["label"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)

    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(len(FEATURE_NAMES),)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        scaler.transform(X_train), y_train,
        validation_split=0.1, epochs=100, batch_size=256, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
    )

    pred = model.predict(scaler.transform(X_test), verbose=0).argmax(axis=1)
    print(f"Test accuracy: {(pred == y_test).mean():.4f}")
    print(classification_report(y_test, pred, target_names=["benign", "malicious"]))

    model.save("Virus_detect.keras")
    with open("scaler.json", "w") as f:
        json.dump({
            "feature_names": FEATURE_NAMES,
            "log1p": True,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        }, f, indent=2)
    print("Saved Virus_detect.keras and scaler.json")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
