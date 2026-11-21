"""Train the Emguard neural network.

Produces:
    Virus_detect.keras  - the neural network
    scaler.json         - feature names, log1p columns and StandardScaler mean/scale

Data: run prepare_data.py first. It builds data/train.npz from all six EMBER 2018
training files (600k labeled samples) and data/test.npz from EMBER's official
test set (200k samples, collected after the training set), which is used here
only for the final evaluation.

Features are computed from the file itself only (see features.py): normalized
byte and byte-entropy histograms, string statistics and general PE info.

Usage (needs scikit-learn in addition to requirements.txt):
    python train_model.py
"""

import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, LOG_FEATURES


def preprocess(X, log_idx):
    X = X.astype(np.float32, copy=True)
    X[:, log_idx] = np.log1p(np.clip(X[:, log_idx], 0, None))
    return X


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
    train, test = np.load("data/train.npz"), np.load("data/test.npz")
    log_idx = [FEATURE_NAMES.index(n) for n in LOG_FEATURES]
    X_train, y_train = preprocess(train["X"], log_idx), train["y"]
    X_test, y_test = preprocess(test["X"], log_idx), test["y"]
    print(f"Train: {len(y_train)} samples, test: {len(y_test)} samples, {X_train.shape[1]} features")

    scaler = StandardScaler().fit(X_train)
    scale = np.where(scaler.scale_ > 0, scaler.scale_, 1.0)
    X_train = ((X_train - scaler.mean_) / scale).astype(np.float32)
    X_test = ((X_test - scaler.mean_) / scale).astype(np.float32)

    tf.keras.utils.set_random_seed(42)
    model = build_model(X_train.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        X_train, y_train,
        validation_split=0.1, epochs=60, batch_size=512, verbose=2, shuffle=True,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
            tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        ],
    )

    prob = model.predict(X_test, batch_size=4096, verbose=0)[:, 1]
    pred = (prob > 0.5).astype(int)
    print(f"Test accuracy: {(pred == y_test).mean():.4f}   ROC AUC: {roc_auc_score(y_test, prob):.4f}")
    print(classification_report(y_test, pred, target_names=["benign", "malicious"], digits=4))

    model.save("Virus_detect.keras")
    with open("scaler.json", "w") as f:
        json.dump({
            "feature_names": FEATURE_NAMES,
            "log1p": LOG_FEATURES,
            "mean": scaler.mean_.tolist(),
            "scale": scale.tolist(),
        }, f)
    print("Saved Virus_detect.keras and scaler.json")


if __name__ == "__main__":
    main()
