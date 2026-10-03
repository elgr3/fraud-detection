from fraud_detection import config


def test_missed_fraud_costs_more_than_false_alert():
    assert config.C_FN > config.C_FP > 0
