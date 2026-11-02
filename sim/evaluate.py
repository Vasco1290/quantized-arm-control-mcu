"""Score controllers on the same 100 seeded episodes (seeds 0..99).

Every controller in the project is judged here, with the same rules:
  1. success rate  (hand within 3 cm of the target at the end)
  2. final distance (mean +- std, in cm)
  3. total reward   (secondary: it can be gamed)
"""
import numpy as np

from exo_arm_env import ExoArmEnv
from int8_net import int8_policy
from student import student_policy
from teacher import teacher_policy

NUM_EPISODES = 100


# A controller ("policy") is any function: (observation, env) -> 2 motor commands.
# It gets the env too, because the teacher is allowed to read the true simulator state.
# A controller with memory can also have a reset() method, called at the start of each episode.

def random_policy(obs, env):
    # env.np_random was seeded in reset(), so these "random" actions are the same on every run
    return env.np_random.uniform(-1.0, 1.0, size=2)


def do_nothing_policy(obs, env):
    return np.zeros(2)


def run_episode(env, policy, seed, delay_steps=0, loss_prob=0.0, on_frame=None):
    """Play one episode through a fake network between the controller and the motors.

    Each step the controller sends a command. It reaches the motors `delay_steps` steps
    later (1 step = 20 ms), or never, if the message is lost (chance `loss_prob`).
    The motors keep using the last command that arrived; before the first one, they get 0.
    With delay 0 and loss 0 this is a normal episode.
    on_frame: optional function called after reset and after every step (used to record GIFs).
    """
    obs, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()  # controllers with memory start each episode fresh
    if on_frame:
        on_frame()
    # Which messages get lost depends only on the seed, so every controller faces the same losses.
    lost = np.random.default_rng(seed).random(1000) < loss_prob
    in_flight = {}           # arrival step -> command on its way
    applied = np.zeros(2)    # what the motors are using right now
    total, t, done = 0.0, 0, False
    while not done:
        command = policy(obs, env)
        if not lost[t]:
            in_flight[t + delay_steps] = command
        if t in in_flight:
            applied = in_flight.pop(t)
        obs, reward, terminated, truncated, info = env.step(applied)
        if on_frame:
            on_frame()
        total += reward
        t += 1
        done = terminated or truncated
    return info["distance"], total


def evaluate(policy, num_episodes=NUM_EPISODES, delay_steps=0, loss_prob=0.0):
    env = ExoArmEnv()
    final_dists, totals = [], []
    for seed in range(num_episodes):
        dist, total = run_episode(env, policy, seed, delay_steps, loss_prob)
        final_dists.append(dist)
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
        ("INT8 student", int8_policy),
    ]:
        print_row(name, evaluate(policy))


if __name__ == "__main__":
    main()
