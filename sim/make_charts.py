"""Draw the README charts from the sweep results, in a light and a dark version.

Reads results/precision_sweep.csv and results/delay_sweep.csv, writes PNGs into media/.
Colours: a checked colour-blind-safe palette; each line also has its own marker shape,
so lines can be told apart without colour.
"""
import csv
from pathlib import Path

import matplotlib
import matplotlib.ticker

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


def loop_diagram(theme, out):
    """The hardware loop as a picture (an image works everywhere; GitHub's live diagrams can be blocked)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    t = THEMES[theme]
    box_fill = "#eef4fc" if theme == "light" else "#1f2a38"
    fig, ax = plt.subplots(figsize=(9, 2.7), dpi=100)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 2.7)
    ax.axis("off")

    boxes = [  # (centre x, title, detail)
        (1.25, "Laptop", "MuJoCo arm simulation"),
        (4.5, "Pi Zero W", "INT8 network (5.4 KB)"),
        (7.75, "Laptop", "fixed-delay buffer"),
    ]
    w, h, y = 2.3, 0.95, 1.55
    for cx, title, detail in boxes:
        ax.add_patch(FancyBboxPatch((cx - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=box_fill, ec=t["series"][0], lw=1.4))
        ax.text(cx, y + 0.15, title, ha="center", va="center", color=t["ink"], fontsize=12, weight="bold")
        ax.text(cx, y - 0.18, detail, ha="center", va="center", color=t["ink2"], fontsize=10.5)

    def arrow(p, q, label, label_xy, **kw):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color=t["muted"], lw=1.4, **kw))
        ax.text(*label_xy, label, ha="center", va="center", color=t["ink2"], fontsize=10)

    arrow((1.25 + w / 2, y), (4.5 - w / 2, y), "observation\n(10 numbers)", (2.875, y + 0.78))
    arrow((4.5 + w / 2, y), (7.75 - w / 2, y), "motor command\n(2 numbers)", (6.125, y + 0.78))
    arrow((7.75, y - h / 2), (1.25, y - h / 2), "applied to the motors every 20 ms", (4.5, 0.32),
          connectionstyle="arc3,rad=-0.15")

    fig.savefig(out, dpi=200, facecolor=t["surface"], bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def timing_chart(theme, messages_csv, title, out):
    """Histogram of round-trip times from one hardware run, with the buffer deadlines marked.

    The question it answers: "if the buffer waits k x 20 ms, what share of answers make it in time?"
    """
    import numpy as np

    with open(messages_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    rtt = np.array([float(r["rtt_ms"]) for r in rows if r["rtt_ms"]])
    comp = np.array([float(r["compute_us"]) for r in rows if r["compute_us"]])
    no_answer = 100 * (1 - len(rtt) / len(rows))

    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=100)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    right = max(50, 10 * np.ceil(np.percentile(rtt, 99.5) / 10))  # x-axis end, in steps of 10 ms
    beyond = 100 * np.mean(rtt > right)                            # answers slower than the axis shows
    bins = np.arange(0, right + 1, 1.0 if right <= 60 else 2.0)
    weights = np.full(len(rtt), 100 / len(rows))                   # each bar = % of all messages sent
    ax.hist(np.minimum(rtt, right), bins=bins, weights=weights, color=t["series"][0],
            edgecolor=t["surface"], linewidth=1.0)

    top = ax.get_ylim()[1]
    for k in (1, 2, 3):
        x = 20 * k
        if x >= right:
            break
        within = 100 * np.mean(rtt <= x) * len(rtt) / len(rows)
        ax.axvline(x, color=t["muted"], lw=1.2, ls=(0, (4, 3)))
        ax.text(x + 0.8, top * 0.95, f"buffer k={k} ({x} ms)\n{within:.1f}% in time",
                color=t["ink2"], fontsize=9.5, va="top", linespacing=1.35,
                bbox=dict(fc=t["surface"], ec="none", pad=1.5))  # keeps grid lines out of the text

    ax.grid(axis="y", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=10, pad=6)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_xlim(0, right)
    ax.set_xlabel("round trip: observation sent → command back (ms)", color=t["ink2"], fontsize=10, labelpad=8)
    ax.set_ylabel("share of messages", color=t["ink2"], fontsize=10, labelpad=8)

    stats = (f"median {np.median(rtt):.1f} ms · 99% {np.percentile(rtt, 99):.1f} ms · "
             f"worst {rtt.max():.1f} ms · no answer {no_answer:.1f}%"
             + (f" · slower than {right:.0f} ms {beyond:.1f}%" if beyond > 0 else "")
             + f"\nnetwork on the chip: mean {comp.mean():.0f} µs, worst {comp.max():.0f} µs"
             + f" · {len(rows)} messages")
    ax.text(0.0, 1.17, title, transform=ax.transAxes, color=t["ink"], fontsize=14, weight="bold")
    ax.text(0.0, 1.03, stats, transform=ax.transAxes, color=t["ink2"], fontsize=10, linespacing=1.4)

    fig.savefig(out, dpi=200, facecolor=t["surface"], bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


# Timing chart source: the real-Pi run with realistic pacing (one message every 20 ms) if it exists.
TIMING_RUNS = [RESULTS / "hw" / "pi_int8_asap_loss0_messages.csv",
               RESULTS / "hw" / "pi_int8_lockstep_loss0_messages.csv"]


def main():
    run = next((p for p in TIMING_RUNS if p.exists()), None)
    for theme in THEMES:
        if run is not None:
            timing_chart(theme, run, "How long does the Pi take to answer?", MEDIA / f"timing_chart_{theme}.png")
        loop_diagram(theme, MEDIA / f"loop_diagram_{theme}.png")
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
    print("saved charts to", MEDIA, "" if run else "(no real-Pi run yet, so no timing chart)")


if __name__ == "__main__":
    main()
