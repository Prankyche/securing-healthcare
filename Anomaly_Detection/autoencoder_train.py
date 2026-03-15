import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from tensorflow.keras import models, layers
from tensorflow.keras.layers import Input, Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import random

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

datapath = "C:\\Users\\prann\\Desktop\\ccp_securehealthcare_final\\Datasets\\dataset4.csv"
f = pd.read_csv(datapath,low_memory=False)
features = f.select_dtypes(include=['int64','float64']).columns.tolist()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "autoencoder_final.keras")
THRESHOLD_PATH = os.path.join(BASE_DIR, "anomaly_threshold.npy")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.save")


def build_autoencoder(input_shape):
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(input_shape, activation="linear")
    ])
    return model


def train_anomaly_model():
    print("[Anomaly] Training autoencoder...")

    df = pd.read_csv(datapath, low_memory=False)

    df[features] = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)

    normal_idx = df['label'].str.contains('Benign', na=False)
    X_normal = df.loc[normal_idx, features]

    scaler = StandardScaler()
    X_normal = scaler.fit_transform(X_normal)

    X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)

    model = build_autoencoder(X_normal.shape[1])

    model.compile(optimizer="adam", loss="mse")

    model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=50,
        batch_size=128,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )
    recon = model.predict(X_normal)
    mse = np.mean(np.square(X_normal - recon), axis=1)
    threshold = np.percentile(mse,99)

    print("Train MSE stats:")
    print("Min:", np.min(mse))
    print("Max:", np.max(mse))
    print("Mean:", np.mean(mse))
    print("99th percentile:", np.percentile(mse, 99))

    model.save(MODEL_PATH)
    np.save(THRESHOLD_PATH, threshold)
    joblib.dump(scaler,SCALER_PATH)

    print("Anomaly Threshold:", threshold)
    print("[Anomaly] Model & threshold saved")

    return scaler

_autoencoder = None
_threshold = None
_scaler = None

def _load():
    global _autoencoder, _threshold, _scaler

    if _autoencoder is None:
        _autoencoder = tf.keras.models.load_model(MODEL_PATH)
        _threshold = np.load(THRESHOLD_PATH)
        _scaler = joblib.load(SCALER_PATH)


def detect_anomaly(sample_df):
    _load()

    sampled = sample_df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    sample_scaled = _scaler.transform(sampled)

    recon = _autoencoder.predict(sample_scaled, verbose=0)
    mse = np.mean(np.square(sample_scaled - recon), axis=1)[0]

    return {
        "is_anomaly": int(mse > _threshold),
        "score": float(mse),
        "threshold": float(_threshold)
    }

if __name__ == "__main__":
    train_anomaly_model()
    df = pd.read_csv(datapath)
    sample = df.iloc[[100]]  # random sample
    result = detect_anomaly(sample)
    print(result)
