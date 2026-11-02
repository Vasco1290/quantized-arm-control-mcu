"""Watch a controller move the arm in the 3D window, at real speed.

Usage:  python sim/watch.py student       (or: teacher, random, do-nothing)
"""
import sys
import time

from evaluate import do_nothing_policy, random_policy
from exo_arm_env import ExoArmEnv
from student import student_policy
from teacher import teacher_policy

POLICIES = {
    "random": random_policy,
    "do-nothing": do_nothing_policy,
    "teacher": teacher_policy,
    "student": student_policy,
}
NUM_EPISODES = 10


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "random"
    policy = POLICIES[name]
    env = ExoArmEnv(render_mode="human")
    for seed in range(NUM_EPISODES):
        obs, info = env.reset(seed=seed)
        time.sleep(0.5)  # pause so you can spot the new red target
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs, env))
            done = terminated or truncated
        print(f"seed {seed}: final distance {100 * info['distance']:5.1f} cm, success={info['success']}")
        time.sleep(0.5)
    env.close()


if __name__ == "__main__":
    main()
