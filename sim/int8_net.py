"""Run the INT8 student with integer maths inside every layer. Needs only NumPy.

This is the file that goes onto the Pi (and gets rewritten in C for the ESP32), so it
imports nothing from the rest of the project.

Inside each layer:  int8 inputs x int8 weights, summed into a 32-bit whole number, plus
an int32 bias. Only between layers is there one multiply by a real-number scale, to
re-round the result onto the next layer's int8 grid.
"""
from pathlib import Path

import numpy as np

INT8_PATH = Path(__file__).resolve().parent.parent / "models" / "student_int8.npz"


class Int8Student:
    def __init__(self, path=INT8_PATH):
        p = np.load(path)
        self.in_mean, self.in_std = p["in_mean"], p["in_std"]
        # Stored as int8 (1 byte each). Widened to int32 only so NumPy sums them without overflow.
        self.W = [p["W1"].astype(np.int32), p["W2"].astype(np.int32), p["W3"].astype(np.int32)]
        self.b = [p["b1"], p["b2"], p["b3"]]   # int32
        self.in_scale = p["in_scale"]          # real size of one integer step, for each layer's input
        self.w_scale = p["w_scale"]            # same, for each layer's weights

    def __call__(self, obs):
        x = (obs - self.in_mean) / self.in_std                                  # scale the 10 inputs
        q = np.clip(np.round(x / self.in_scale[0]), -127, 127).astype(np.int32)  # ... and round to int8

        for k in range(3):
            acc = q @ self.W[k] + self.b[k]                    # whole-number multiply-and-add (int32)
            real = acc * (self.in_scale[k] * self.w_scale[k])  # back to real units
            if k == 2:
                return np.clip(real, -1.0, 1.0)                # last layer: the 2 motor commands
            real = np.maximum(real, 0.0)                       # ReLU
            q = np.clip(np.round(real / self.in_scale[k + 1]), -127, 127).astype(np.int32)


_int8 = None


def int8_policy(obs, env):
    """Uses only the 10 observation numbers, like the float student."""
    global _int8
    if _int8 is None:
        _int8 = Int8Student()
    return _int8(obs)
