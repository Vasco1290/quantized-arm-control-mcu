"""Laptop side of the hardware loop: the arm is simulated here, the controller runs on the chip.

Every 20 ms of real time the laptop sends the arm's observation to the chip, and applies a
motor command that has come back. Three rules for which command to apply:

  lockstep  wait for the answer before moving the arm on. Not realistic timing (the arm
            "pauses"), but checks the chip gives the same answers as the laptop, and
            measures the round-trip time.
  asap      at the start of each step, use the newest answer that has arrived. Realistic:
            the answer to step t is used at step t+1 at the earliest (20 ms), later if the
            network is slow.
  buffer    fixed-delay buffer: the answer to step t is used at exactly step t+k.
            Answers that arrive later than that count as lost. Makes the delay constant,
            which is what the delay-trained student needs.

Scores the usual test episodes with the usual success rule, and records for every message
the round-trip time, how long the chip took to compute, and when the answer was used.

  python sim/hw_loop.py --controller int8 --mode asap                 (stand-in Pi on this laptop)
  python sim/hw_loop.py --host arm-pi.local --controller delay1 --mode buffer --k 1
"""
import argparse
import csv
import socket
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pi"))

import protocol as P  # noqa: E402
from exo_arm_env import EPISODE_STEPS, SUCCESS_DIST, ExoArmEnv  # noqa: E402

STEP_S = 0.02             # one control step = 20 ms
LOCKSTEP_TIMEOUT_S = 0.5  # lockstep: give up on an answer after this long
OUT_DIR = ROOT / "results" / "hw"


def wait_until(t, poll):
    """Wait until perf_counter() reaches t, calling poll() about every 0.5 ms meanwhile,
    so answers are picked up (and time-stamped) as soon as they arrive."""
    while True:
        poll()
        left = t - time.perf_counter()
        if left <= 0:
            return
        if left > 0.001:
            time.sleep(0.0005)


class Link:
    """The laptop's end of the UDP connection to the chip."""

    def __init__(self, host, port):
        self.addr = (socket.gethostbyname(host), port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if hasattr(socket, "SIO_UDP_CONNRESET"):  # Windows only: don't error if the chip isn't up yet
            self.sock.ioctl(socket.SIO_UDP_CONNRESET, False)
        self.sock.setblocking(False)

    def send(self, packet):
        self.sock.sendto(packet, self.addr)

    def receive_all(self):
        """Every message waiting right now, each with the time we picked it up."""
        out = []
        while True:
            try:
                data, _ = self.sock.recvfrom(1024)
            except (BlockingIOError, ConnectionResetError):
                return out
            msg = P.unpack_to_laptop(data)
            if msg is not None:
                out.append((msg, time.perf_counter()))

    def reset(self, episode, code):
        """Tell the chip a new episode starts; repeat until it confirms (a message can get lost)."""
        for _ in range(30):
            self.send(P.pack_reset(episode, code))
            deadline = time.perf_counter() + 0.1
            while time.perf_counter() < deadline:
                for msg, _ in self.receive_all():
                    if msg[0] == "ack" and msg[1] == episode:
                        return
                time.sleep(0.001)
        raise RuntimeError(f"no answer from the chip at {self.addr}: is pi_server.py running?")


def run_episode(link, env, episode, seed, code, mode, k, loss, log):
    link.reset(episode, code)
    obs, info = env.reset(seed=seed)
    lost = np.random.default_rng(seed).random(1000) < loss  # same losses as evaluate.py
    sent, arrived, used = {}, {}, {}   # step -> send time / (command, arrival time, compute us) / step used at
    applied = np.zeros(2)
    newest = -1
    overruns = 0
    total = 0.0

    def collect():
        for msg, t_arr in link.receive_all():
            if msg[0] == "cmd" and msg[1] == episode and msg[2] not in arrived:
                arrived[msg[2]] = (np.array(msg[3]), t_arr, msg[4])

    start = time.perf_counter() + 0.005
    for t in range(EPISODE_STEPS):
        slot = start + t * STEP_S
        wait_until(slot, collect)
        if time.perf_counter() - slot > 0.002:
            overruns += 1  # the laptop itself fell behind schedule

        # 1. At the start of each step, choose the command from answers that have ALREADY arrived.
        if mode == "asap":
            fresh = [s for s in arrived if s > newest]
            if fresh:
                newest = max(fresh)
                applied, used[newest] = arrived[newest][0], t
        elif mode == "buffer" and (t - k) in arrived:
            applied, used[t - k] = arrived[t - k][0], t

        # 2. Send this step's observation (lockstep: and wait for its answer).
        if not lost[t]:
            link.send(P.pack_obs(episode, t, obs))
            sent[t] = time.perf_counter()
            if mode == "lockstep":
                deadline = time.perf_counter() + LOCKSTEP_TIMEOUT_S
                while t not in arrived and time.perf_counter() < deadline:
                    collect()
                if t in arrived:
                    applied, used[t] = arrived[t][0], t

        # 3. Move the simulated arm on by 20 ms.
        obs, reward, terminated, truncated, info = env.step(applied)
        total += reward
        if mode == "lockstep":
            start = time.perf_counter() - (t + 1) * STEP_S  # lockstep: next step right away, no pacing

    wait_until(time.perf_counter() + 0.1, collect)  # let stragglers arrive so their timing is recorded
    for s, t_sent in sent.items():
        command, t_arr, us = arrived.get(s, (None, None, None))
        log.append({
            "episode": episode, "step": s,
            "rtt_ms": None if t_arr is None else round(1000 * (t_arr - t_sent), 3),
            "compute_us": us,
            "used_after_steps": None if s not in used else used[s] - s,
        })
    return info["distance"], total, overruns


def summarise(log):
    rtts = np.array([r["rtt_ms"] for r in log if r["rtt_ms"] is not None])
    comp = np.array([r["compute_us"] for r in log if r["compute_us"] is not None])
    delays = [r["used_after_steps"] for r in log if r["used_after_steps"] is not None]
    return {
        "messages": len(log),
        "no_answer_pct": 100 * (1 - len(rtts) / max(1, len(log))),
        "answer_used_pct": 100 * len(delays) / max(1, len(log)),
        "rtt_mean_ms": rtts.mean(), "rtt_p50_ms": np.percentile(rtts, 50),
        "rtt_p99_ms": np.percentile(rtts, 99), "rtt_max_ms": rtts.max(), "rtt_std_ms": rtts.std(),
        "compute_mean_us": comp.mean(), "compute_max_us": comp.max(),
        "delay_steps_hist": {d: delays.count(d) for d in sorted(set(delays))},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=P.PORT)
    ap.add_argument("--controller", default="int8", choices=sorted(P.CONTROLLERS))
    ap.add_argument("--mode", default="asap", choices=["lockstep", "asap", "buffer"])
    ap.add_argument("--k", type=int, default=1, help="buffer mode: use each answer exactly k steps later")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--loss", type=float, default=0.0, help="extra: drop this share of observations on purpose")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    where = "local" if args.host in ("127.0.0.1", "localhost") else "pi"
    mode = f"buffer{args.k}" if args.mode == "buffer" else args.mode
    label = args.label or f"{where}_{args.controller}_{mode}_loss{round(100 * args.loss)}"

    link = Link(args.host, args.port)
    env = ExoArmEnv()
    code = P.CONTROLLERS[args.controller]
    dists, totals, log, overruns = [], [], [], 0
    base = int(time.time()) % 1_000_000 * 1000  # episode numbers unique across runs
    for seed in range(args.episodes):
        d, tot, o = run_episode(link, env, base + seed, seed, code, args.mode, args.k, args.loss, log)
        dists.append(d)
        totals.append(tot)
        overruns += o
    env.close()

    dists = np.array(dists)
    s = summarise(log)
    print(f"\n{label}: {args.episodes} episodes")
    print(f"  success {100 * np.mean(dists < SUCCESS_DIST):.0f}%   final distance "
          f"{100 * dists.mean():.2f} +- {100 * dists.std():.2f} cm   reward {np.mean(totals):.2f}")
    print(f"  round trip ms: mean {s['rtt_mean_ms']:.2f}  median {s['rtt_p50_ms']:.2f}  "
          f"99% {s['rtt_p99_ms']:.2f}  worst {s['rtt_max_ms']:.2f}  jitter (std) {s['rtt_std_ms']:.2f}")
    print(f"  chip compute us: mean {s['compute_mean_us']:.0f}  worst {s['compute_max_us']:.0f}")
    print(f"  messages {s['messages']}  no answer {s['no_answer_pct']:.1f}%  answer used {s['answer_used_pct']:.1f}%"
          f"  used after (steps: count) {s['delay_steps_hist']}  laptop overruns {overruns}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{label}_messages.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log[0].keys())
        w.writeheader()
        w.writerows(log)
    summary_path = OUT_DIR / "summary.csv"
    row = {"label": label, "host": where, "controller": args.controller, "mode": mode,
           "loss_pct": round(100 * args.loss), "episodes": args.episodes,
           "success_pct": 100 * np.mean(dists < SUCCESS_DIST),
           "dist_mean_cm": 100 * dists.mean(), "dist_std_cm": 100 * dists.std(),
           **{key: round(float(v), 3) for key, v in s.items() if key not in ("delay_steps_hist", "messages")},
           "messages": s["messages"], "laptop_overruns": overruns}
    new = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"  saved {OUT_DIR / (label + '_messages.csv')} and a row in {summary_path}")


if __name__ == "__main__":
    main()
