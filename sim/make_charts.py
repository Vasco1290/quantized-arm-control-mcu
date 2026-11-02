"""Draw the README charts from the sweep results, in a light and a dark version.

Reads results/precision_sweep.csv and results/delay_sweep.csv, writes PNGs into media/.
Colours: a checked colour-blind-safe palette; each line also has its own marker shape,
so lines can be told apart without colour.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to files, no window
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MEDIA = ROOT / "media"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500"]),
}
MARKERS = ["o", "s", "^", "D"]

plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def read(name):
    with open(RESULTS / name, newline="") as f:
        return list(csv.DictReader(f))


def line_chart(theme, title, subtitle, series, xlabel, xticks, note, note_xy, out, invert_x=False):
    """series: list of (label, xs, ys) with ys as success rates 0..100."""
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=100)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    for i, (label, xs, ys) in enumerate(series):
        ax.plot(xs, ys, label=label, color=t["series"][i], lw=1.6,
                marker=MARKERS[i], ms=6.5, mec=t["surface"], mew=1.5,  # ring in the background colour
                solid_capstyle="round", solid_joinstyle="round", zorder=10 - i)

    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks(xticks)
    if invert_x:
        ax.invert_xaxis()
    ax.grid(axis="y", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=10, pad=6)
    ax.set_xlabel(xlabel, color=t["ink2"], fontsize=10, labelpad=8)
    ax.set_ylabel("success rate", color=t["ink2"], fontsize=10, labelpad=8)

    ax.text(*note_xy, note, color=t["ink2"], fontsize=10, va="top", linespacing=1.4)
    # Top to bottom: title, subtitle, legend row, then the plot.
    ax.legend(loc="lower left", bbox_to_anchor=(-0.015, 1.02), ncol=len(series), frameon=False,
              labelcolor=t["ink2"], fontsize=10, handlelength=1.8, columnspacing=1.6)
    ax.text(0.0, 1.22, title, transform=ax.transAxes, color=t["ink"], fontsize=14, weight="bold")
    ax.text(0.0, 1.15, subtitle, transform=ax.transAxes, color=t["ink2"], fontsize=10)

    MEDIA.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=t["surface"], bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def precision_series():
    rows = read("precision_sweep.csv")
    out = []
    for mode, label in [("everything", "everything"), ("weights only", "weights only"),
                        ("inputs only", "inputs only"), ("hidden only", "hidden values only")]:
        pts = sorted(((int(r["bits"]), 100 * float(r["success"])) for r in rows if r["mode"] == mode),
                     reverse=True)
        out.append((label, [p[0] for p in pts], [p[1] for p in pts]))
    return out


def delay_series():
    rows = [r for r in read("delay_sweep.csv") if r["experiment"] == "A"]
    out = []
    for name, label in [("delay-trained (matched)", "delay-trained student"),
                        ("float student", "float student"),
                        ("INT8 student", "INT8 student"),
                        ("teacher", "PD teacher")]:
        pts = sorted((int(r["delay_ms"]), 100 * float(r["success"])) for r in rows if r["controller"] == name)
        out.append((label, [p[0] for p in pts], [p[1] for p in pts]))
    return out


def main():
    for theme in THEMES:
        line_chart(theme,
                   "How few bits can the controller use?",
                   "Part of the network rounded to N bits · 100 test reaches per point · success = hand within 3 cm",
                   precision_series(), "bits per number (fewer = smaller, coarser)", [8, 6, 4, 3, 2],
                   "Safe down to 6 bits.\nBreaks between 6 and 4.", (7.85, 45),
                   MEDIA / f"precision_chart_{theme}.png", invert_x=True)
        line_chart(theme,
                   "Delay breaks plain controllers, not the delay-trained one",
                   "Simulated network delay · 100 test reaches per point · success = hand within 3 cm",
                   delay_series(), "delay between sending a command and the motor receiving it (ms)",
                   [0, 20, 40, 60, 80, 100],
                   "Delay-trained student: 100% at every delay\n(each one built for the delay it is tested at)",
                   (36, 80), MEDIA / f"delay_chart_{theme}.png")
    print("saved charts to", MEDIA)


if __name__ == "__main__":
    main()
