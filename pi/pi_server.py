"""Runs on the Pi: receive an observation, run the network, send the motor command back.

Needs only Python 3 and NumPy (on the Pi:  sudo apt install python3-numpy).
On the Pi, keep these in one folder:  pi_server.py, protocol.py, int8_net.py, student.py,
and a models/ folder with student_int8.npz and student_delay*.npz.

  On the Pi:                    python3 pi_server.py
  Stand-in "Pi" on the laptop:  python pi/pi_server.py
  ... with a pretend slow network (for testing the laptop side):
                                python pi/pi_server.py --delay-ms 30 --jitter-ms 15
"""
import argparse
import heapq
import random
import socket
import sys
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "sim")]  # on the Pi everything sits next to this file

import numpy as np  # noqa: E402

import protocol as P  # noqa: E402
from int8_net import Int8Student  # noqa: E402
from student import Student  # noqa: E402


class PlainController:
    def __init__(self, net):
        self.net = net

    def reset(self):
        pass

    def __call__(self, obs):
        return self.net(obs)


class DelayController:
    """Delay-trained student: also remembers the last k commands it sent (same as sim/delay_student.py)."""

    def __init__(self, k, net):
        self.k, self.net = k, net

    def reset(self):
        self.sent = deque([np.zeros(2)] * self.k, maxlen=self.k)

    def __call__(self, obs):
        command = self.net(np.concatenate([obs, *self.sent]))
        self.sent.append(command)
        return command


def load_controllers(models_dir):
    controllers = {0: PlainController(Int8Student(models_dir / "student_int8.npz"))}
    for k in range(1, 6):
        path = models_dir / f"student_delay{k}.npz"
        if path.exists():
            controllers[k] = DelayController(k, Student(path))
    return controllers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=P.PORT)
    ap.add_argument("--models", type=Path,
                    default=HERE / "models" if (HERE / "models").exists() else HERE.parent / "models")
    ap.add_argument("--delay-ms", type=float, default=0.0, help="pretend network: hold each answer this long")
    ap.add_argument("--jitter-ms", type=float, default=0.0, help="pretend network: +- this much random extra")
    args = ap.parse_args()

    controllers = load_controllers(args.models)
    print(f"loaded controllers {sorted(controllers)} from {args.models}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if hasattr(socket, "SIO_UDP_CONNRESET"):  # Windows only: don't error when the laptop side goes away
        sock.ioctl(socket.SIO_UDP_CONNRESET, False)
    sock.bind(("0.0.0.0", args.port))
    print(f"listening on UDP port {args.port}")

    episode, active = None, None
    pending = []   # answers waiting to be sent (only used by the pretend slow network)
    counter = 0    # tie-breaker so the heap never compares two packets
    while True:
        # Wait for a message, but wake up in time to send any answer that is due.
        timeout = None if not pending else max(1e-4, pending[0][0] - time.perf_counter())
        sock.settimeout(timeout)
        try:
            data, addr = sock.recvfrom(1024)
        except (socket.timeout, BlockingIOError, ConnectionResetError):
            data = None
        while pending and pending[0][0] <= time.perf_counter():
            _, _, packet, to = heapq.heappop(pending)
            sock.sendto(packet, to)
        if data is None:
            continue

        msg = P.unpack_to_chip(data)
        if msg is None:
            continue
        if msg[0] == "reset":
            _, episode, code = msg
            active = controllers[code]
            active.reset()
            sock.sendto(P.pack_ack(episode), addr)
            if episode % 10 == 0:
                print(f"episode {episode}, controller code {code}")
        elif msg[0] == "obs" and active is not None and msg[1] == episode:
            _, ep, step, obs = msg
            t0 = time.perf_counter_ns()
            command = active(np.array(obs, dtype=np.float32))
            compute_us = (time.perf_counter_ns() - t0) // 1000
            packet = P.pack_cmd(ep, step, command, compute_us)
            delay = max(0.0, args.delay_ms + random.uniform(-args.jitter_ms, args.jitter_ms)) / 1000
            if delay == 0:
                sock.sendto(packet, addr)
            else:
                counter += 1
                heapq.heappush(pending, (time.perf_counter() + delay, counter, packet, addr))


if __name__ == "__main__":
    main()
