"""The student network, run with plain NumPy (no PyTorch needed).

PyTorch is only used to *train* it. Running it is just a few multiply-and-add steps,
written out here by hand, because this exact code is what we will shrink to INT8 and
run on the Pi.

Network: 10 inputs -> 64 -> 64 -> 2 outputs, ReLU in between.
"""
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "student_float.npz"


def relu(x):
    return np.maximum(x, 0.0)  # keep positives, turn negatives into 0


class Student:
    def __init__(self, path=MODEL_PATH):
        p = np.load(path)
        self.in_mean, self.in_std = p["in_mean"], p["in_std"]
        self.W1, self.b1 = p["W1"], p["b1"]
        self.W2, self.b2 = p["W2"], p["b2"]
        self.W3, self.b3 = p["W3"], p["b3"]

    def __call__(self, obs):
        x = (obs - self.in_mean) / self.in_std  # put all 10 inputs on a similar scale
        h1 = relu(x @ self.W1 + self.b1)          # layer 1: 10 -> 64
        h2 = relu(h1 @ self.W2 + self.b2)         # layer 2: 64 -> 64
        out = h2 @ self.W3 + self.b3              # layer 3: 64 -> 2 motor commands
        return np.clip(out, -1.0, 1.0)


_student = None


def student_policy(obs, env):
    """Uses only the 10 observation numbers. env is ignored: the student may not peek."""
    global _student
    if _student is None:
        _student = Student()  # load the weights once, on first use
    return _student(obs)
