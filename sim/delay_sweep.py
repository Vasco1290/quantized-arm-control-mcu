"""Delay and message-loss sweep, with a fake network on the laptop (no hardware needed).

Three experiments, each on the usual 100 test episodes:
  A. Delay 0..100 ms, no loss: teacher, plain float student, INT8 student, and the
     delay-trained student built for that exact delay.
  B. Wrong delay: each delay-trained student (built for k steps) tested at every delay,
     because a real network's delay is never exactly what you planned for.
  C. Lost messages 0..30%, at 0, 20 and 60 ms delay.
Results go to results/delay_sweep.csv.
"""
import csv
from pathlib import Path

from delay_student import DELAYS, DelayStudent
from evaluate import evaluate
from int8_net import int8_policy
from student import student_policy
from teacher import teacher_policy

STEP_MS = 20
ALL_DELAYS = [0] + DELAYS
LOSSES = [0.0, 0.1, 0.2, 0.3]
OUT_PATH = Path(__file__).resolve().parent.parent / "results" / "delay_sweep.csv"

rows = []


def run(experiment, name, policy, k, loss):
    r = evaluate(policy, delay_steps=k, loss_prob=loss)
    rows.append({"experiment": experiment, "controller": name, "delay_ms": STEP_MS * k,
                 "loss_pct": round(100 * loss), **r})
    return f"{100 * r['success']:4.0f}% {r['dist_mean_cm']:6.2f}"


def matched(k):
    """The right controller for delay k: the plain student at 0, else the delay-trained one."""
    return student_policy if k == 0 else DelayStudent(k)


def main():
    print("A. Delay sweep, no loss   (success %, mean final distance cm)")
    print(f"{'delay':>6}  {'teacher':>13}  {'float student':>13}  {'INT8 student':>13}  {'delay-trained':>13}")
    for k in ALL_DELAYS:
        cells = [run("A", "teacher", teacher_policy, k, 0.0),
                 run("A", "float student", student_policy, k, 0.0),
                 run("A", "INT8 student", int8_policy, k, 0.0),
                 run("A", "delay-trained (matched)", matched(k), k, 0.0)]
        print(f"{STEP_MS * k:4d}ms  " + "  ".join(cells))

    print("\nB. Wrong delay: success % (rows = built for, columns = actual delay)")
    print("built for " + "".join(f"{STEP_MS * k:>8d}ms" for k in ALL_DELAYS))
    for built in DELAYS:
        cells = [run("B", f"delay-trained {STEP_MS * built}ms", DelayStudent(built), k, 0.0)
                 for k in ALL_DELAYS]
        print(f"{STEP_MS * built:7d}ms " + "".join(f"{c.split('%')[0]:>9}%" for c in cells))

    print("\nC. Lost messages   (success %, mean final distance cm)")
    for k in (0, 1, 3):
        print(f"delay {STEP_MS * k} ms:")
        for loss in LOSSES:
            cells = [run("C", "float student", student_policy, k, loss)]
            if k > 0:
                cells.append(run("C", f"delay-trained {STEP_MS * k}ms", DelayStudent(k), k, loss))
            print(f"  loss {round(100 * loss):3d}%  plain {cells[0]}" +
                  (f"   delay-trained {cells[1]}" if k > 0 else ""))

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()
