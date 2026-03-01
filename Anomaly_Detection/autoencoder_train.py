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

# ==============================
# PATHS
# ==============================

datapath = "C:\\Users\\prann\\Desktop\\ccp_securehealthcare_final\\Anomaly_Detection\\dataset2.csv"
f = pd.read_csv(datapath,low_memory=False)
# print(df['label'].value_counts())
features = f.select_dtypes(include=['int64','float64']).columns.tolist()
# print(features)
# features = [
#     'ts', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
#     'duration', 'orig_bytes', 'resp_bytes',
#     'conn_state', 'missed_bytes', 'orig_pkts',
#     'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes'
# ]
#
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "autoencoder_final.keras")
THRESHOLD_PATH = os.path.join(BASE_DIR, "anomaly_threshold.npy")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.save")

# ==============================
# BUILD MODEL
# ==============================

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

# ==============================
# TRAIN MODEL
# ==============================

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

    # Save model and threshold
    model.save(MODEL_PATH)
    np.save(THRESHOLD_PATH, threshold)
    joblib.dump(scaler,SCALER_PATH)

    print("Anomaly Threshold:", threshold)
    print("[Anomaly] Model & threshold saved")

    return scaler

# ==============================
# LOAD MODEL
# ==============================

_autoencoder = None
_threshold = None
_scaler = None

def _load():
    global _autoencoder, _threshold, _scaler

    if _autoencoder is None:
        _autoencoder = tf.keras.models.load_model(MODEL_PATH)
        _threshold = np.load(THRESHOLD_PATH)
        _scaler = joblib.load(SCALER_PATH)

        # df = pd.read_csv(datapath, low_memory=False)
        # df[features] = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
        # normal_idx = df['label'].str.contains('benign', na=False)
        #
        # _scaler = StandardScaler().fit(df.loc[normal_idx, features])

# ==============================
# DETECT ANOMALY
# ==============================

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

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    train_anomaly_model()
    df = pd.read_csv(datapath)
    sample = df.iloc[[100]]  # random sample
    result = detect_anomaly(sample)
    print(result)
















































# # import keras
# import os.path
#
# import numpy as np
# import pandas as pd
# from tqdm import tqdm, trange
# # from keras import models,layers
# # import tensorflow as tf
# # from tensorflow import keras
# # # from tensorflow.keras import models,layers
# #
# # from keras import layers, regularizers, Sequential
# # from keras.layers import Input, Dense
# # from keras.utils import plot_model
# # from keras.models import Sequential
# import tensorflow as tf
# from tensorflow.keras import models, layers, Sequential
# from tensorflow.keras.layers import Input, Dense
# # from tensorflow_model_optimization
# import tensorflow_model_optimization as tfmot
#
#
#
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
#
# datapath = "C:\\Users\\prann\\Desktop\\ccp_securehealthcare_final\\Anomaly_Detection\\dataset10.csv"
# features = ['ts', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'duration', 'orig_bytes', 'resp_bytes',
#             'conn_state', 'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
#
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR,"quantized_autoencoder_final.keras")
# THRESHOLD_PATH = os.path.join(BASE_DIR,"anomaly_threshold.npy")
#
# df = pd.read_csv(datapath, low_memory=False)
#
# df[features] = df[features].apply(pd.to_numeric, errors='coerce')
# df[features] = df[features].fillna(0)
# normal_idx = df['attack_type'].str.contains('Benign', na=False)
# X_normal = df.loc[normal_idx,features]
# X_normal = X_normal.select_dtypes(include=['number'])
# X_all = df[features]
# #
# scaler = StandardScaler()
# X_normal = scaler.fit_transform(X_normal)
# X_all = scaler.transform(X_all)
# #
# X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)
#
# def build_autoencoder(input_shape):
#     model = models.Sequential([
#         layers.Input(shape=(input_shape,)),
#         layers.Dense(128, activation="relu"),
#         layers.Dense(64, activation="relu"),
#         layers.Dense(32, activation="relu"),
#         layers.Dense(64, activation="relu"),
#         layers.Dense(128, activation="relu"),
#         layers.Dense(input_shape, activation="linear")
#     ])
#     return model
#
# def train_anomaly_model():
#     print("[Anomaly] Training autoencoder...")
#
#     df = pd.read_csv(datapath, low_memory=False)
#     df[features] = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
#
#     normal_idx = df['attack_type'].str.contains('Benign', na=False)
#     X_normal = df.loc[normal_idx, features]
#
#     scaler = StandardScaler()
#     X_normal = scaler.fit_transform(X_normal)
#
#     X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)
#
#     autoencoder = build_autoencoder(X_normal.shape[1])
#     q_model = tfmot.quantization.keras.quantize_model(autoencoder)
#
#     q_model.compile(optimizer="adam", loss="mse")
#     q_model.fit(
#         X_normal, X_normal,
#         epochs=50,
#         batch_size=128,
#         validation_split=0.2,
#         callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
#         verbose=1
#     )
#
#     recon = q_model.predict(X_normal)
#     mse = np.mean(np.square(X_normal - recon), axis=1)
#     threshold = np.mean(mse) + 3 * np.std(mse)
#
#     # final_model = strip_quantization(q_model)
#     # final_model = tfmot.quantization.keras.strip_quantization(q_model)
#     # final_model.save(MODEL_PATH)
#     q_model.save(MODEL_PATH)
#     print("Anomaly Threshold: ",threshold)
#     np.save(THRESHOLD_PATH, threshold)
#
#     print("[Anomaly] Model & threshold saved")
#     return scaler
#
# # autoencoder = build_autoencoder(X_train.shape[1])
# # quantize_model = tfmot.quantization.keras.quantize_model
# # q_aware_model = quantize_model(autoencoder)
# # q_aware_model.compile(
# #     optimizer='adam',
# #     loss='mse'
# # )
# #
# # q_aware_model.fit(
# #     X_train,
# #     X_train,
# #     validation_data=(X_val,X_val),
# #     epochs=50,
# #     batch_size=128,
# #     callbacks=[
# #         tf.keras.callbacks.EarlyStopping(
# #             patience=5,
# #             restore_best_weights=True
# #         )
# #     ],
# #     verbose=1
# # )
# #
# # recon = q_aware_model.predict(X_train)
# # mse = np.mean(np.square(X_train - recon), axis=1)
# #
# # threshold = np.mean(mse) + 3 * np.std(mse)
# # print("Anomaly Threshold:", threshold)
# #
# # X_test = scaler.transform(df[features])
# #
# # recon_test = q_aware_model.predict(X_test)
# # mse_test = np.mean(np.square(X_test - recon_test), axis=1)
# #
# # df['anomaly'] = mse_test > threshold
#
# # print(df.groupby('attack_type')['anomaly'].mean())
#
# #do later plis
# # q_aware_model.save("quantized_autoencoder")
# # np.save("anomaly_threshold.npy", threshold)
#
# _autoencoder = None
# _threshold = None
# _scaler = None
#
# def _load():
#     global _autoencoder, _threshold, _scaler
#
#     if _autoencoder is None:
#         _autoencoder = tf.keras.models.load_model(
#             MODEL_PATH,
#             custom_objects={
#                 "QuantizeLayer": tfmot.quantization.keras.QuantizeLayer
#             }
#         )
#         _threshold = np.load(THRESHOLD_PATH)
#
#         df = pd.read_csv(datapath, low_memory=False)
#         df[features] = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
#         normal_idx = df['attack_type'].str.contains('Benign', na=False)
#         _scaler = StandardScaler().fit(df.loc[normal_idx, features])
#
#
# def detect_anomaly(sample_df):
#
#     _load()
#
#     sample = sample_df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
#     sample_scaled = _scaler.transform(sample)
#
#     recon = _autoencoder.predict(sample_scaled, verbose=0)
#     mse = np.mean(np.square(sample_scaled - recon), axis=1)[0]
#
#     return {
#         "is_anomaly": int(mse > _threshold),
#         "score": float(mse),
#         "threshold": float(_threshold)
#     }
#
# if __name__ == "__main__":
#     # df = pd.read_csv(datapath)
#     # sample = df.iloc[[100]]
#     #
#     # out = detect_anomaly(sample)
#     # print(out)
#     autoencoder = build_autoencoder(X_train.shape[1])
#     train_anomaly_model()
#
#
#
#
#
#
#
#
#
#
#
#
#
# # RANDOM_SEED = 7
# # DATA_PATH = '/kaggle/input/real-time-internet-of-things-rt-iot2022/RT_IOT2022.csv'
# # #DATA_PATH = '../../rt-iot2022/RT_IOT2022.csv'
# # #DATA_PATH = 'RT_IOT2022.csv'
# # #LIGHT_CIC_PATH = '0.01percent_2classes.csv' # Light version of the CIC dataset
# # LIGHT_CIC_PATH = '/kaggle/input/creating-a-smaller-dataset-for-ciciot2023/0.01percent_2classes.csv' # Light version of the CIC dataset
# # LIGHT_CIC_34_PATH = '/kaggle/input/creating-a-smaller-dataset-for-ciciot2023/0.01percent_34classes.csv'
# # FIG_PATH = ""
# # TARGET = 'Attack_type'
# # TARGET_CIC = 'benign'
# # LAYERS_1_AND_7_NEURONS = 128
#
# # tf.random.set_seed(RANDOM_SEED)
# # np.random.seed(RANDOM_SEED)
#
# # print("NumPy version:", np.__version__)
# # print("TensorFlow version:", tf.__version__)
# # print("TensorFlow Model Optimization version:", tfmot.__version__)
# # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#
# # assert tf.__version__ == "2.13.0", 'TensorFlow 2.13 required for compatibility with tfmot 0.8.0.'
# # assert LAYERS_1_AND_7_NEURONS % 16 == 0, 'The specified layer number is not dividable by 16.'
#
# # pruning_params = {
# #     'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
# #         initial_sparsity=0.0,
# #         final_sparsity=0.5,
# #         begin_step=0,
# #         end_step=len(X_train) // 256 * 25
# #     )
# # }
#
# # pruned_model = tfmot.sparsity.keras.prune_low_magnitude(autoencoder, **pruning_params)
#
# # pruned_model.compile(
# #     optimizer='adam',
# #     loss='mse'
# # )
#
# # callbacks = [
# #     tfmot.sparsity.keras.UpdatePruningStep(),
# #     tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
# # ]
#
# # history = pruned_model.fit(
# #     X_train, X_train,
# #     validation_data=(X_val, X_val),
# #     epochs=50,
# #     batch_size=256,
# #     callbacks=callbacks,
# #     verbose=1
# # )
#
# # final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
# # final_model.save("pruned_autoencoder")
