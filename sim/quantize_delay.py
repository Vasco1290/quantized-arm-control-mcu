"""INT8 versions of the delay-trained students, so the Pi runs every controller in INT8.

Exactly the same recipe as sim/quantize.py, only the input is bigger: 10 observation
numbers + 2k remembered commands. For each delay k:
  1. re-record the training inputs (same seeds, same noise) to measure the biggest values,
  2. save models/student_delay{k}_int8.npz,
  3. check the integer runner matches the 8-bit rounding simulation,
  4. score it on the usual 100 test episodes with a k-step (k x 20 ms) delay.

Usage:  python sim/quantize_delay.py      (~1 min per delay)
"""
import numpy as np

from delay_student import DELAYS, DelayStudent, collect_delayed, model_path
from evaluate import evaluate, print_row
from int8_net import Int8Student
from quantize import QuantStudent, calibrate, export_int8
from student import Student


def int8_path(k):
    return model_path(k).with_name(f"student_delay{k}_int8.npz")


def main():
    for k in DELAYS:
        print(f"--- delay {k} step(s) = {20 * k} ms ---")
        student = Student(model_path(k))
        inputs, _ = collect_delayed(k)
        ranges = calibrate(student, inputs)
        export_int8(student, ranges, int8_path(k))

        int8 = Int8Student(int8_path(k))
        sample = inputs[::20]
        sim8 = QuantStudent(student, ranges, w_bits=8, in_bits=8, act_bits=8)(sample)
        print(f"integer runner vs 8-bit simulation, max difference: {np.abs(sim8 - int8(sample)).max():.2e}")

        print_row("float", evaluate(DelayStudent(k), delay_steps=k))
        print_row("INT8", evaluate(DelayStudent(k, net=int8), delay_steps=k))


if __name__ == "__main__":
    main()
