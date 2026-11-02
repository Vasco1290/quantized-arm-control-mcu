"""Precision sweep: how few bits can the student use before reaching breaks?

For each bit width, we round ONE part of the network at a time (weights, inputs, or the
in-between values), then everything together, and run the usual 100-episode test.
This shows which part is most sensitive. Results go to results/precision_sweep.csv.
"""
import csv
from pathlib import Path

from evaluate import evaluate
from quantize import QuantStudent, load_calibration
from student import Student

BITS = [8, 6, 4, 3, 2]
OUT_PATH = Path(__file__).resolve().parent.parent / "results" / "precision_sweep.csv"


def modes(b):
    return {
        "weights only": dict(w_bits=b),
        "inputs only": dict(in_bits=b),
        "hidden only": dict(act_bits=b),
        "everything": dict(w_bits=b, in_bits=b, act_bits=b),
    }


def main():
    student = Student()
    ranges = load_calibration(student)
    rows = []
    print(f"{'bits':>4}  {'rounded part':<13} {'success':>7}   {'final distance':>16}")
    for b in BITS:
        for mode, kwargs in modes(b).items():
            net = QuantStudent(student, ranges, **kwargs)
            r = evaluate(lambda obs, env: net(obs))
            rows.append({"bits": b, "mode": mode, **r})
            print(f"{b:>4}  {mode:<13} {100 * r['success']:>6.0f}%   "
                  f"{r['dist_mean_cm']:6.2f} +- {r['dist_std_cm']:5.2f} cm")

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
