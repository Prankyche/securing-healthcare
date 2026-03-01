from random import random, randint

import pandas as pd
import numpy as np
import joblib
import warnings
import os

from Anomaly_Detection.autoencoder_train import MODEL_PATH

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
datapath = r"C:\Users\prann\Desktop\ccp_securehealthcare_final\Classification\dataset2.csv"
MODEL_PATH = os.path.join(BASE_DIR,"rf_model.save")
FEATURES_PATH = os.path.join(BASE_DIR,"rf_features.save")

# print("Loading dataset...")
# df = pd.read_csv(datapath, low_memory=False)
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)
#
# split = int(len(df) * 0.8)
# train_df = df.iloc[:split].copy()
# test_df  = df.iloc[split:].copy()
#
# x_columns = [
#     'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
#     'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
#     'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
#     'ece_flag_number', 'cwr_flag_number', 'ack_count',
#     'syn_count', 'fin_count', 'urg_count', 'rst_count',
#     'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
#     'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
#     'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
#     'Magnitue',
#     'Radius', 'Covariance', 'Variance', 'Weight'
# ]
#
# y_column = 'label'
#
# dict_7classes = {
#     'DDoS-RSTFINFlood':'DDoS','DDoS-PSHACK_Flood':'DDoS','DDoS-SYN_Flood':'DDoS',
#     'DDoS-UDP_Flood':'DDoS','DDoS-TCP_Flood':'DDoS','DDoS-ICMP_Flood':'DDoS',
#     'DDoS-SynonymousIP_Flood':'DDoS','DDoS-ACK_Fragmentation':'DDoS',
#     'DDoS-UDP_Fragmentation':'DDoS','DDoS-ICMP_Fragmentation':'DDoS',
#     'DDoS-SlowLoris':'DDoS','DDoS-HTTP_Flood':'DDoS',
#
#     'DoS-UDP_Flood':'DoS','DoS-SYN_Flood':'DoS',
#     'DoS-TCP_Flood':'DoS','DoS-HTTP_Flood':'DoS',
#
#     'Mirai-greeth_flood':'Mirai','Mirai-greip_flood':'Mirai',
#     'Mirai-udpplain':'Mirai',
#
#     'Recon-PingSweep':'Recon','Recon-OSScan':'Recon',
#     'Recon-PortScan':'Recon','Recon-HostDiscovery':'Recon',
#     'VulnerabilityScan':'Recon',
#
#     'DNS_Spoofing':'Spoofing','MITM-ArpSpoofing':'Spoofing',
#
#     'BenignTraffic':'Benign',
#
#     'BrowserHijacking':'Web','Backdoor_Malware':'Web',
#     'XSS':'Web','Uploading_Attack':'Web',
#     'SqlInjection':'Web','CommandInjection':'Web',
#
#     'DictionaryBruteForce':'BruteForce'
# }
#
# train_df['label_7'] = train_df[y_column].map(dict_7classes)
# test_df['label_7']  = test_df[y_column].map(dict_7classes)
#
# train_df.dropna(subset=['label_7'], inplace=True)
# test_df.dropna(subset=['label_7'], inplace=True)
#
# train_df[x_columns] = train_df[x_columns].apply(pd.to_numeric, errors='coerce')
# test_df[x_columns]  = test_df[x_columns].apply(pd.to_numeric, errors='coerce')
#
# train_df.dropna(inplace=True)
# test_df.dropna(inplace=True)
#
# rf = RandomForestClassifier(
#     n_estimators=500,
#     min_samples_split=5,
#     min_samples_leaf=1,
#     class_weight="balanced_subsample",
#     n_jobs=-1,
#     random_state=42
# )
#
# print("\nTraining started: ")
# rf.fit(train_df[x_columns], train_df['label_7'])
#
# train_pd = rf.predict(train_df[x_columns])
# print("Train Accuracy",accuracy_score(train_df['label_7'],train_pd))
#
# y_pred = rf.predict(test_df[x_columns])
# y_true = test_df['label_7']
# y_prob = rf.predict_proba(test_df[x_columns])
#
# print("Test Accuracy",accuracy_score(test_df['label_7'],y_pred))
#
# print("\nPerformance Metrics")
# print("Accuracy :", accuracy_score(y_true, y_pred))
# print("Recall   :", recall_score(y_true, y_pred, average='macro'))
# print("Precision:", precision_score(y_true, y_pred, average='macro'))
# print("F1-score :", f1_score(y_true, y_pred, average='macro'))
#
# joblib.dump(rf,MODEL_PATH)
# joblib.dump(x_columns,FEATURES_PATH)

ATTACK_TO_ID = {
    "Benign": 0,
    "DDoS": 1,
    "DoS": 2,
    "Recon": 3,
    "Spoofing": 4,
    "Web": 5,
    "BruteForce": 6,
    "Mirai": 7
}

DANGER_LEVEL = {
    "Benign": 0,
    "Recon": 1,
    "Spoofing": 1,
    "Web": 1,
    "BruteForce": 2,
    "DoS": 2,
    "DDoS": 3,
    "Mirai": 3
}

def get_severity(predicted_class, confidence, anomaly_score):

    if predicted_class == "Benign":
        return 0
    if anomaly_score < 0.1:
        anomaly_factor = 0
    elif anomaly_score < 0.2:
        anomaly_factor = 1
    else:
        anomaly_factor = 2

    if confidence < 0.7:
        conf_factor = 0
    elif confidence < 0.9:
        conf_factor = 1
    else:
        conf_factor = 2

    severity = round((randint(0,2)+anomaly_factor + conf_factor) / 3)

    return min(severity, 2)


_rf_model = None
_rf_features = None

def _load_model():
    global _rf_model,_rf_features
    if _rf_model is None:
        import joblib
        _rf_model = joblib.load(MODEL_PATH)
        _rf_features = joblib.load(FEATURES_PATH)

def rf_to_rl(sample_df,anomaly_score):
    _load_model()
    sample_df = sample_df[_rf_features].apply(pd.to_numeric, errors='coerce')
    sample_df.dropna(inplace=True)

    probs = _rf_model.predict_proba(sample_df)[0]
    class_index = np.argmax(probs)
    confidence = probs[class_index]

    predicted_label = _rf_model.classes_[class_index]
    attack_type = ATTACK_TO_ID[predicted_label]
    severity = get_severity(predicted_label, confidence, anomaly_score)

    return {
        "attack_type": attack_type,
        "attack_name": predicted_label,
        "confidence": float(confidence),
        "severity": severity
    }

if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv(datapath, low_memory=False)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    x_columns = [
        'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
        'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
        'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
        'ece_flag_number', 'cwr_flag_number', 'ack_count',
        'syn_count', 'fin_count', 'urg_count', 'rst_count',
        'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
        'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
        'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
        'Magnitue',
        'Radius', 'Covariance', 'Variance', 'Weight'
    ]

    y_column = 'label'

    dict_7classes = {
        'DDoS-RSTFINFlood': 'DDoS', 'DDoS-PSHACK_Flood': 'DDoS', 'DDoS-SYN_Flood': 'DDoS',
        'DDoS-UDP_Flood': 'DDoS', 'DDoS-TCP_Flood': 'DDoS', 'DDoS-ICMP_Flood': 'DDoS',
        'DDoS-SynonymousIP_Flood': 'DDoS', 'DDoS-ACK_Fragmentation': 'DDoS',
        'DDoS-UDP_Fragmentation': 'DDoS', 'DDoS-ICMP_Fragmentation': 'DDoS',
        'DDoS-SlowLoris': 'DDoS', 'DDoS-HTTP_Flood': 'DDoS',

        'DoS-UDP_Flood': 'DoS', 'DoS-SYN_Flood': 'DoS',
        'DoS-TCP_Flood': 'DoS', 'DoS-HTTP_Flood': 'DoS',

        'Mirai-greeth_flood': 'Mirai', 'Mirai-greip_flood': 'Mirai',
        'Mirai-udpplain': 'Mirai',

        'Recon-PingSweep': 'Recon', 'Recon-OSScan': 'Recon',
        'Recon-PortScan': 'Recon', 'Recon-HostDiscovery': 'Recon',
        'VulnerabilityScan': 'Recon',

        'DNS_Spoofing': 'Spoofing', 'MITM-ArpSpoofing': 'Spoofing',

        'BenignTraffic': 'Benign',

        'BrowserHijacking': 'Web', 'Backdoor_Malware': 'Web',
        'XSS': 'Web', 'Uploading_Attack': 'Web',
        'SqlInjection': 'Web', 'CommandInjection': 'Web',

        'DictionaryBruteForce': 'BruteForce'
    }

    train_df['label_7'] = train_df[y_column].map(dict_7classes)
    test_df['label_7'] = test_df[y_column].map(dict_7classes)

    train_df.dropna(subset=['label_7'], inplace=True)
    test_df.dropna(subset=['label_7'], inplace=True)

    train_df[x_columns] = train_df[x_columns].apply(pd.to_numeric, errors='coerce')
    test_df[x_columns] = test_df[x_columns].apply(pd.to_numeric, errors='coerce')

    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    rf = RandomForestClassifier(
        n_estimators=500,
        min_samples_split=5,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )

    print("\nTraining started: ")
    rf.fit(train_df[x_columns], train_df['label_7'])

    train_pd = rf.predict(train_df[x_columns])
    print("Train Accuracy", accuracy_score(train_df['label_7'], train_pd))

    y_pred = rf.predict(test_df[x_columns])
    y_true = test_df['label_7']
    y_prob = rf.predict_proba(test_df[x_columns])

    print("Test Accuracy", accuracy_score(test_df['label_7'], y_pred))

    print("\nPerformance Metrics")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred, average='macro'))
    print("Precision:", precision_score(y_true, y_pred, average='macro'))
    print("F1-score :", f1_score(y_true, y_pred, average='macro'))

    joblib.dump(rf, MODEL_PATH)
    joblib.dump(x_columns, FEATURES_PATH)