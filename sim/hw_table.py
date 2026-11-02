"""Print the hardware-loop results as Markdown table rows, ready to paste into the README.

Reads results/hw/summary.csv (one row per sim/hw_loop.py run).
Usage:  python sim/hw_table.py          (real-Pi runs only)
        python sim/hw_table.py --all    (also stand-in runs on the laptop)
"""
import csv
import sys
from pathlib import Path

SUMMARY = Path(__file__).resolve().parent.parent / "results" / "hw" / "summary.csv"

MODE_TEXT = {"lockstep": "lockstep (arm waits)", "asap": "as soon as it arrives"}


def describe(mode):
    if mode.startswith("buffer"):
        k = int(mode[len("buffer"):])
        return f"fixed {20 * k} ms buffer"
    return MODE_TEXT.get(mode, mode)


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to an older encoding without ± and µ
    with open(SUMMARY, newline="") as f:
        rows = list(csv.DictReader(f))
    if "--all" not in sys.argv:
        rows = [r for r in rows if r["host"] == "pi"]
    print("| Controller | How answers are used | Extra loss | Success | Final distance "
          "| Round trip median / 99% / worst | Chip compute |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        name = "INT8 student" if r["controller"] == "int8" else f"INT8 delay-trained ({r['controller']})"
        print(f"| {name} | {describe(r['mode'])} | {r['loss_pct']}% | {float(r['success_pct']):.0f}% "
              f"| {float(r['dist_mean_cm']):.2f} ± {float(r['dist_std_cm']):.2f} cm "
              f"| {float(r['rtt_p50_ms']):.1f} / {float(r['rtt_p99_ms']):.1f} / {float(r['rtt_max_ms']):.1f} ms "
              f"| {float(r['compute_mean_us']):.0f} µs |")


if __name__ == "__main__":
    main()
