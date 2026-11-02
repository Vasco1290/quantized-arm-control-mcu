"""Delay-trained student: a network that plans for its commands arriving late.

With a delay of k steps, a command sent now only reaches the motors k steps later.
By then the arm has moved, pushed by the k commands that were already on their way.
A plain controller ignores this and reacts to where the arm *was*, so it overshoots.

The delay-trained student gets two things:
  - the current observation (10 numbers)
  - the k commands it already sent that have not landed yet (2k numbers)
and is trained to output what the teacher would do at the moment its command lands.
In other words, it learns to look ahead k steps.

Usage:  python sim/delay_student.py        (trains one student for each delay 1..5 steps)
"""
from collections import deque

import numpy as np

from collect import FIRST_SEED, NOISE_STD, NUM_EPISODES
from exo_arm_env import EPISODE_STEPS, ExoArmEnv
from student import MODEL_PATH, Student
from teacher import teacher_policy
from train_student import train

DELAYS = [1, 2, 3, 4, 5]  # steps of 20 ms -> 20..100 ms


def model_path(k):
    return MODEL_PATH.with_name(f"student_delay{k}.npz")


def collect_delayed(k):
    """Record training rows for delay k.

    We drive the arm the way a *perfect* delay-trained student would: for the first k steps
    nothing has arrived yet, so the motors get 0; after that, the command that lands at each
    step is exactly what the teacher wants at that step (plus the usual small wobble).
    Then each row is:  input = observation at t + commands landing at t..t+k-1,
                       label = the teacher's command at t+k (the one sent at t must equal it).
    """
    env = ExoArmEnv()
    inputs, labels = [], []
    for seed in range(FIRST_SEED, FIRST_SEED + NUM_EPISODES):
        obs, info = env.reset(seed=seed)
        O, A, U = [], [], []  # observations, clean teacher commands, commands actually applied
        for t in range(EPISODE_STEPS):
            a = teacher_policy(obs, env)
            u = np.zeros(2) if t < k else a + env.np_random.normal(0.0, NOISE_STD, size=2)
            O.append(obs)
            A.append(a)
            U.append(u)
            obs, reward, terminated, truncated, info = env.step(u)
        for t in range(EPISODE_STEPS - k):
            inputs.append(np.concatenate([O[t], *U[t:t + k]]))
            labels.append(A[t + k])
    env.close()
    return np.array(inputs, dtype=np.float32), np.array(labels, dtype=np.float32)


class DelayStudent:
    """Controller with a short memory: it remembers the last k commands it sent."""

    def __init__(self, k, path=None):
        self.k = k
        self.net = Student(path or model_path(k))

    def reset(self):
        # At the start nothing has been sent; the motors are getting 0, so remember zeros.
        self.sent = deque([np.zeros(2)] * self.k, maxlen=self.k)

    def __call__(self, obs, env):
        command = self.net(np.concatenate([obs, *self.sent]))
        self.sent.append(command)  # oldest one drops off automatically
        return command


def main():
    for k in DELAYS:
        print(f"--- delay {k} step(s) = {20 * k} ms ---")
        x, y = collect_delayed(k)
        train(x, y, EPISODE_STEPS - k, model_path(k))


if __name__ == "__main__":
    main()
