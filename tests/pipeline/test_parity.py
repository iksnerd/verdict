from verdict.train.parity import parity

QS = {"k": {"type": "choice", "instructions": "?", "criteria": ["a", "b"]}, "n": {"type": "noul", "instructions": "?"}}


class Agent:
    def __init__(self, k, n):
        self.k, self.n = k, n

    def predict(self, state, questions):
        pa = self.k
        return {"answers": {
            "k": {"type": "choice", "choice": "a" if pa >= 0.5 else "b", "probabilities": {"a": pa, "b": 1 - pa}, "confidence": 0.1},
            "n": {"type": "noul", "noul": self.n, "confidence": 0.5},
        }}


CASES = [("s1", QS), ("s2", QS)]


def test_identical_agents_agree_completely():
    r = parity(Agent(0.7, 0.9), Agent(0.7, 0.9), CASES)
    assert r["argmax_agreement"] == 1.0 and r["max_prob_diff"] == 0.0 and r["n_questions"] == 4 and r["ok"]


def test_small_numeric_drift_is_reported_but_passes_within_tolerance():
    r = parity(Agent(0.700, 0.900), Agent(0.705, 0.897), CASES, prob_tol=0.01)
    assert r["argmax_agreement"] == 1.0 and 0 < r["max_prob_diff"] < 0.01 and r["ok"]


def test_a_flipped_decision_fails_and_is_listed():
    r = parity(Agent(0.55, 0.9), Agent(0.45, 0.9), CASES)
    assert r["argmax_agreement"] < 1.0 and not r["ok"]
    assert r["disagreements"][0]["question"] == "k"


def test_drift_beyond_tolerance_fails_even_when_decisions_match():
    r = parity(Agent(0.9, 0.9), Agent(0.8, 0.9), CASES, prob_tol=0.02)
    assert r["argmax_agreement"] == 1.0 and not r["ok"]
