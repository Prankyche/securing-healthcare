import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

datapath = r"C:\Users\prann\Desktop\ccp_securehealthcare_final\Datasets\dataset4.csv"

print("Loading dataset...")
df = pd.read_csv(datapath, low_memory=False)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

split = int(len(df) * 0.8)
train_df = df.iloc[:split].copy()
test_df  = df.iloc[split:].copy()

x_columns = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
    'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
    'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count',
    'syn_count', 'fin_count', 'urg_count', 'rst_count',
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
    'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
    'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
    'Magnitue',   # NOTE: CICIoT spelling
    'Radius', 'Covariance', 'Variance', 'Weight'
]

y_column = 'label'

train_df[x_columns] = train_df[x_columns].apply(pd.to_numeric, errors='coerce')
test_df[x_columns]  = test_df[x_columns].apply(pd.to_numeric, errors='coerce')

train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

train_df[y_column] = train_df[y_column].astype(str)
test_df[y_column]  = test_df[y_column].astype(str)


scaler = StandardScaler()
scaler.fit(train_df[x_columns])

train_df[x_columns] = scaler.transform(train_df[x_columns])
test_df[x_columns]  = scaler.transform(test_df[x_columns])

model_34 = LogisticRegression(
    solver="lbfgs",
    multi_class="multinomial",
    class_weight="balanced",
    max_iter=500,
    n_jobs=-1
)

# print("\nTraining 34-class model...")
# model_34.fit(train_df[x_columns], train_df[y_column])
#
# y_pred_34 = model_34.predict(test_df[x_columns])
# y_test_34 = test_df[y_column]
#
# print("\n##### LogisticRegression (34 Classes) #####")
# print("Accuracy :", accuracy_score(y_test_34, y_pred_34))
# print("Recall   :", recall_score(y_test_34, y_pred_34, average='macro'))
# print("Precision:", precision_score(y_test_34, y_pred_34, average='macro'))
# print("F1-score :", f1_score(y_test_34, y_pred_34, average='macro'))
#

dict_7classes = {
    'DDoS-RSTFINFlood':'DDoS','DDoS-PSHACK_Flood':'DDoS','DDoS-SYN_Flood':'DDoS',
    'DDoS-UDP_Flood':'DDoS','DDoS-TCP_Flood':'DDoS','DDoS-ICMP_Flood':'DDoS',
    'DDoS-SynonymousIP_Flood':'DDoS','DDoS-ACK_Fragmentation':'DDoS',
    'DDoS-UDP_Fragmentation':'DDoS','DDoS-ICMP_Fragmentation':'DDoS',
    'DDoS-SlowLoris':'DDoS','DDoS-HTTP_Flood':'DDoS',

    'DoS-UDP_Flood':'DoS','DoS-SYN_Flood':'DoS',
    'DoS-TCP_Flood':'DoS','DoS-HTTP_Flood':'DoS',

    'Mirai-greeth_flood':'Mirai','Mirai-greip_flood':'Mirai',
    'Mirai-udpplain':'Mirai',

    'Recon-PingSweep':'Recon','Recon-OSScan':'Recon',
    'Recon-PortScan':'Recon','Recon-HostDiscovery':'Recon',
    'VulnerabilityScan':'Recon',

    'DNS_Spoofing':'Spoofing','MITM-ArpSpoofing':'Spoofing',

    'BenignTraffic':'Benign',

    'BrowserHijacking':'Web','Backdoor_Malware':'Web',
    'XSS':'Web','Uploading_Attack':'Web',
    'SqlInjection':'Web','CommandInjection':'Web',

    'DictionaryBruteForce':'BruteForce'
}

train_df['label_7'] = train_df[y_column].map(dict_7classes)
test_df['label_7']  = test_df[y_column].map(dict_7classes)

train_df.dropna(subset=['label_7'], inplace=True)
test_df.dropna(subset=['label_7'], inplace=True)

model_7 = LogisticRegression(
    solver="lbfgs",
    multi_class="multinomial",
    max_iter=500,
    n_jobs=-1
)

print("\nTraining 7+1 class model...")
model_7.fit(train_df[x_columns], train_df['label_7'])

y_pred_7 = model_7.predict(test_df[x_columns])
y_test_7 = test_df['label_7']

print("\n##### LogisticRegression (7+1 Classes) #####")
print("Accuracy :", accuracy_score(y_test_7, y_pred_7))
print("Recall   :", recall_score(y_test_7, y_pred_7, average='macro'))
print("Precision:", precision_score(y_test_7, y_pred_7, average='macro'))
print("F1-score :", f1_score(y_test_7, y_pred_7, average='macro'))
