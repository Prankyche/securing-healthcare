import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
# import matplotlib.pyplot as plt
# import seaborn as sns

# ==========================
# 1. Load dataset
# ==========================
datapath = r"C:\Users\prann\Desktop\ccp_securehealthcare_final\Classification\dataset2.csv"
df = pd.read_csv(datapath, low_memory=False)

# Columns to use as features
x_columns = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
    'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
    'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count',
    'syn_count', 'fin_count', 'urg_count', 'rst_count',
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
    'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
    'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue',
    'Radius', 'Covariance', 'Variance', 'Weight'
]

# Target column
y_column = 'label'

# ==========================
# 2. 7+1 class mapping
# ==========================
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

df['label_7'] = df[y_column].map(dict_7classes)
df.dropna(subset=['label_7'], inplace=True)  # remove any unmapped rows

# ==========================
# 3. Preprocessing
# ==========================
# Convert features to numeric
df[x_columns] = df[x_columns].apply(pd.to_numeric, errors='coerce')
df.dropna(subset=x_columns, inplace=True)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(df[x_columns])

# Encode target
le = LabelEncoder()
y = le.fit_transform(df['label_7'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# ==========================
# 4. Build MLP model
# ==========================
num_classes = len(np.unique(y))

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==========================
# 5. Train model
# ==========================
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=1024,
    verbose=1
)

# ==========================
# 6. Evaluate model
# ==========================
y_pred = np.argmax(model.predict(X_test), axis=1)

print("\n##### MLP (7+1 Classes) #####")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred, average='macro'))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("F1-score :", f1_score(y_test, y_pred, average='macro'))

# ==========================
# 7. Confusion Matrix
# ==========================
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10,8))
# sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.title('Confusion Matrix (MLP)')
# plt.show()
