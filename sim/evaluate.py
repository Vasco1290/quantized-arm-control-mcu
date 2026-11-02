"""Score controllers on the same 100 seeded episodes (seeds 0..99).

Every controller in the project is judged here, with the same rules:
  1. success rate  (hand within 3 cm of the target at the end)
  2. final distance (mean +- std, in cm)
  3. total reward   (secondary: it can be gamed)
"""
import numpy as np

from exo_arm_env import ExoArmEnv
from student import student_policy
from teacher import teacher_policy

NUM_EPISODES = 100


# A controller ("policy") is any function: (observation, env) -> 2 motor commands.
# It gets the env too, because the teacher is allowed to read the true simulator state.

def random_policy(obs, env):
    # env.np_random was seeded in reset(), so these "random" actions are the same on every run
    return env.np_random.uniform(-1.0, 1.0, size=2)


def do_nothing_policy(obs, env):
    return np.zeros(2)


def evaluate(policy, num_episodes=NUM_EPISODES):
    env = ExoArmEnv()
    final_dists, totals = [], []
    for seed in range(num_episodes):
        obs, info = env.reset(seed=seed)
        total = 0.0
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs, env))
            total += reward
            done = terminated or truncated
        final_dists.append(info["distance"])
        totals.append(total)
    env.close()

    final_dists = np.array(final_dists)
    totals = np.array(totals)
    return {
        "success": float(np.mean(final_dists < 0.03)),
        "dist_mean_cm": 100 * final_dists.mean(),
        "dist_std_cm": 100 * final_dists.std(),
        "reward_mean": totals.mean(),
        "reward_std": totals.std(),
    }


def print_row(name, r):
    print(f"{name:<14} {100 * r['success']:>6.0f}%   "
          f"{r['dist_mean_cm']:6.2f} +- {r['dist_std_cm']:5.2f} cm   "
          f"{r['reward_mean']:7.2f} +- {r['reward_std']:5.2f}")


def main():
    print(f"{'controller':<14} {'success':>7}   {'final distance':>18}   {'reward':>15}")
    for name, policy in [
        ("random", random_policy),
        ("do-nothing", do_nothing_policy),
        ("PD teacher", teacher_policy),
        ("float student", student_policy),
    ]:
        print_row(name, evaluate(policy))


if __name__ == "__main__":
    main()
