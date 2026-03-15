import pandas as pd
from pandas import read_csv
from Classification.classification_randomforest import rf_to_rl
from Anomaly_Detection.autoencoder_train import detect_anomaly
from Decision_Making.decisionmaking_model import RLDecisionEngine as rl, RLDecisionEngine

df = pd.read_csv("Datasets/dataset2.csv")
sample_df = df[df['label']!= "BenignTraffic"].sample(1)
rl_engine = RLDecisionEngine()
rl_engine.train()

print("Driver columns")
print(sample_df.columns.tolist())
anomaly_out = detect_anomaly(sample_df)

print("[Anomaly]", anomaly_out)

if anomaly_out["is_anomaly"] == 0:
    print("Benign traffic → no action")
    exit()

cls_out = rf_to_rl(sample_df, anomaly_out["score"])
attack_type = int(cls_out['attack_type'])
attack_name = cls_out["attack_name"]
confidence = cls_out['confidence']
seve = int(cls_out['severity'])
severity = anomaly_out['score']/anomaly_out['threshold']

print(f"Attack: {attack_name} | Severity: {seve} | Confidence: {confidence}")

action_id, action = rl_engine.decide(
    attack_type,
    seve
)

print("Action:", action)
