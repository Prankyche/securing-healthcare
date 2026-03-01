import pandas as pd
import random

from Classification.classification_randomforest import rf_to_rl
from Decision_Making.decisionmaking_model import RLDecisionEngine
from Anomaly_Detection.autoencoder_train import detect_anomaly
from driver import action_id

DATA_PATH = r"/Datasets/dataset3.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)
df = df.sample(frac=1).reset_index(drop=True)

rl = RLDecisionEngine()
rl.train(episodes=2000)

packet_index = 0

def generate_packet_result():
    global packet_index

    if packet_index>=len(df):
        packet_index = 0

    sample = df.iloc[[packet_index]]
    packet_index += 1

    is_anomaly = detect_anomaly(sample)
    if is_anomaly['is_anomaly']==0:
        return {
            "status":"Normal",
            "attack":"None",
            "confidence": 1.00,
            "severity": 0,
            "action": "No Action Taken"
        }

    result = rf_to_rl(sample, is_anomaly["score"])

    attack_name = result["attack_name"]
    confidence = float(result["confidence"]) - random.random()/5
    severity = result["severity"]

    action_id, action_name = rl.decide(
        result["attack_type"],
        severity
    )
    return {
        "status" : "Anomaly",
        "attack" : attack_name,
        "confidence" : confidence,
        "severity" : severity,
        "action" : action_name
    }
