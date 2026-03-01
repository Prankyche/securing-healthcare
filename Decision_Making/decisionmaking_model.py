import numpy as np
import random
import sys
import pandas as pd
sys.path.append("C:/Users/prann/Desktop/ccp_securehealthcare/Classification")

from Classification.classification_randomforest import rf_to_rl

class RLDecisionEngine:
    def __init__(self):
        self.num_attack_types = 8
        self.num_severity = 3
        self.num_states = self.num_attack_types * self.num_severity
        self.num_actions = 5

        self.Q = np.zeros((self.num_states, self.num_actions))

        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

        self.actions_map = {
            0: "Ignore",
            1: "Send Alert",
            2: "Isolate Device",
            3: "Block Traffic",
            4: "Emergency Shutdown"
        }

    def encode_state(self, attack_type, severity):
        return attack_type * self.num_severity + severity

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        return np.argmax(self.Q[state])

    def get_reward(self, attack_type, severity, action):
        if attack_type == 0:
            return 2 if action == 0 else -5

        if severity == 0 and action == 1:
            return 5
        if severity == 1 and action == 2:
            return 8
        if severity == 2 and action in [3, 4]:
            return 12

        return -8

    def train(self, episodes=3000):
        for _ in range(episodes):
            atk = random.randint(0, self.num_attack_types-1)
            sev = random.randint(0, 2)

            s = self.encode_state(atk, sev)
            a = self.choose_action(s)
            r = self.get_reward(atk, sev, a)

            next_s = self.encode_state(
                random.randint(0, 4),
                random.randint(0, 2)
            )

            self.Q[s, a] += self.alpha * (
                r + self.gamma * np.max(self.Q[next_s]) - self.Q[s, a]
            )

    def decide(self, attack_type, severity):
        s = self.encode_state(attack_type, severity)
        a = np.argmax(self.Q[s])
        return a, self.actions_map[a]


if __name__ == "__main__":
    rl_engine = RLDecisionEngine()
    rl_engine.train(episodes=3000)

    sample_df = pd.read_csv(
        "C:/Users/prann/Desktop/ccp_securehealthcare/Classification/dataset2.csv"
    ).sample(1)

    clf_output = rf_to_rl(sample_df, anomaly_score)

    attack_type = clf_output["attack_type"]
    attack_name = clf_output["attack_name"]
    confidence = clf_output["confidence"]
    severity = clf_output["severity"]

    action_id, action_name = rl_engine.decide(attack_type, severity)

    print("Attack Type:", attack_type)
    print("Severity:", severity)
    print("RL Decision:", action_name)