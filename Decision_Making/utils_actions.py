def get_severity(attack_type, confidence):
    if attack_type == 0:
        return 0   #benign

    if confidence < 0.6:
        return 0   #Low

    if confidence < 0.85:
        return 1   #Medium

    return 2       #High
