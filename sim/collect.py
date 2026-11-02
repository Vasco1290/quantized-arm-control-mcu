"""Record the teacher: run it on many practice episodes and save (observation -> action) pairs.

The student network learns from these pairs. Practice seeds start at 1000, so they never
overlap the test seeds 0..99: the student is always tested on targets it has never seen.
"""
from pathlib import Path

import numpy as np

from exo_arm_env import ExoArmEnv
from teacher import teacher_policy

NUM_EPISODES = 2000
FIRST_SEED = 1000
NOISE_STD = 0.1   # random wobble added to the motors while recording (see below)
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "teacher_data.npz"


def main():
    env = ExoArmEnv()
    observations, actions = [], []
    for seed in range(FIRST_SEED, FIRST_SEED + NUM_EPISODES):
        obs, info = env.reset(seed=seed)
        done = False
        while not done:
            action = teacher_policy(obs, env)
            observations.append(obs)
            actions.append(action)  # we save the teacher's clean answer ...

            # ... but the arm is moved with a slightly wrong command. This pushes the arm
            # into "slightly off course" states, so the recording also shows the teacher
            # correcting mistakes, which is exactly what a student needs to learn.
            noisy = action + env.np_random.normal(0.0, NOISE_STD, size=2)
            obs, reward, terminated, truncated, info = env.step(noisy)
            done = terminated or truncated
    env.close()

    OUT_PATH.parent.mkdir(exist_ok=True)
    np.savez(OUT_PATH,
             obs=np.array(observations, dtype=np.float32),
             act=np.array(actions, dtype=np.float32))
    print(f"saved {len(actions)} pairs from {NUM_EPISODES} episodes to {OUT_PATH}")


if __name__ == "__main__":
    main()
