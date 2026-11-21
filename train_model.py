"""Train the Emguard neural network on EMBER2024.

Produces:
    Virus_detect.keras  - the neural network
    scaler.json         - StandardScaler mean/scale for the feature vector

Data: run prepare_data.py first. EMBER2024 covers Win32, Win64 and .NET files
first seen on VirusTotal from Sep 2023 to Dec 2024. The train split is the first
52 weeks; the test split is the 12 weeks after it, so the test score measures
how well the model handles malware newer than anything it was trained on. The
challenge set is malware that ~70 antivirus engines initially missed.

Usage (needs scikit-learn in addition to requirements.txt):
    python train_model.py
"""

import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score

from features import N_FEATURES


def build_model(n_features):
    layers = [tf.keras.Input(shape=(n_features,))]
    for units in (512, 256, 128):
        layers += [
            tf.keras.layers.Dense(units),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(0.2),
        ]
    layers.append(tf.keras.layers.Dense(2, activation="softmax"))
    return tf.keras.Sequential(layers)


def main():
    train = np.load("data/train.npz")
    X_train, y_train = train["X"], train["y"]
    assert X_train.shape[1] == N_FEATURES, "data/ was built with a different features.py"
    print(f"Train: {len(y_train)} samples ({int(y_train.sum())} malicious), {N_FEATURES} features")

    mean = X_train.mean(axis=0, dtype=np.float64)
    std = X_train.std(axis=0, dtype=np.float64)
    scale = np.where(std > 1e-8, std, 1.0)
    X_train -= mean.astype(np.float32)
    X_train /= scale.astype(np.float32)

    tf.keras.utils.set_random_seed(42)
    model = build_model(N_FEATURES)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    # Validate on a random 5% of the training set.
    idx = np.random.default_rng(42).permutation(len(y_train))
    n_val = len(idx) // 20
    val, tr = idx[:n_val], idx[n_val:]
    model.fit(
        X_train[tr], y_train[tr],
        validation_data=(X_train[val], y_train[val]),
        epochs=40, batch_size=1024, verbose=2,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        ],
    )
    del X_train

    def predict(path):
        d = np.load(path)
        X = ((d["X"] - mean) / scale).astype(np.float32)
        return model.predict(X, batch_size=8192, verbose=0)[:, 1], d["y"]

    prob, y_test = predict("data/test.npz")
    pred = (prob > 0.5).astype(int)
    print(f"Test (12 later weeks): accuracy {(pred == y_test).mean():.4f}, "
          f"ROC AUC {roc_auc_score(y_test, prob):.4f}")
    print(classification_report(y_test, pred, target_names=["benign", "malicious"], digits=4))

    prob_c, _ = predict("data/challenge.npz")
    print(f"Challenge set (evasive malware): detected {(prob_c > 0.5).mean():.4f} of {len(prob_c)}")

    model.save("Virus_detect.keras")
    with open("scaler.json", "w") as f:
        json.dump({"n_features": N_FEATURES, "mean": mean.tolist(), "scale": scale.tolist()}, f)
    print("Saved Virus_detect.keras and scaler.json")


if __name__ == "__main__":
    main()
