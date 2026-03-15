import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

datapath = r"C:\Users\prann\Desktop\ccp_securehealthcare_final\Datasets\dataset4.csv"
df = pd.read_csv(datapath, low_memory=False)

x_columns = [
    'flow_duration','Header_Length','Protocol Type','Duration','Rate',
    'Srate','Drate','fin_flag_number','syn_flag_number','rst_flag_number',
    'psh_flag_number','ack_flag_number','ece_flag_number','cwr_flag_number',
    'ack_count','syn_count','fin_count','urg_count','rst_count',
    'HTTP','HTTPS','DNS','Telnet','SMTP','SSH','IRC','TCP','UDP',
    'DHCP','ARP','ICMP','IPv','LLC','Tot sum','Min','Max','AVG','Std',
    'Tot size','IAT','Number','Magnitue','Radius','Covariance',
    'Variance','Weight'
]

y_column = "label"

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

df["label_7"] = df[y_column].map(dict_7classes)
df.dropna(inplace=True)

df[x_columns] = df[x_columns].apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[x_columns])

le = LabelEncoder()
y_encoded = le.fit_transform(df["label_7"])
y_cat = to_categorical(y_encoded)

def create_sequences(X, y, window_size=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq)

WINDOW_SIZE = 10

X_seq, y_seq = create_sequences(X_scaled, y_cat, WINDOW_SIZE)

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq.argmax(axis=1)
)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu',
           input_shape=(WINDOW_SIZE, X_train.shape[2])),
    MaxPooling1D(pool_size=2),

    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),

    LSTM(64),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(y_train.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=256,
    validation_split=0.1,
    verbose=1
)

y_pred = model.predict(X_test)
y_pred_labels = y_pred.argmax(axis=1)
y_test_labels = y_test.argmax(axis=1)

print("\n##### CNN + LSTM (7+1 Classes) #####")
print("Accuracy :", accuracy_score(y_test_labels, y_pred_labels))
print("Recall   :", recall_score(y_test_labels, y_pred_labels, average='macro'))
print("Precision:", precision_score(y_test_labels, y_pred_labels, average='macro'))
print("F1-score :", f1_score(y_test_labels, y_pred_labels, average='macro'))
