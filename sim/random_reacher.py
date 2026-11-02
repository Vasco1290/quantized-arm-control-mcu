import gymnasium as gym
import numpy as np

ENV_ID = "Reacher-v5"
NUM_EPISODES = 10
SEED = 0


def run_episode(env, seed):
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    return total_reward, steps


def main():
    env = gym.make(ENV_ID, render_mode="human")
    try:
        print("Observation space:", env.observation_space)
        print("Observation size:", env.observation_space.shape[0])
        print("Action space:", env.action_space)
        print("Action size:", env.action_space.shape[0])
        print("Action low:", env.action_space.low)
        print("Action high:", env.action_space.high)

        env.action_space.seed(SEED)

        totals = []
        for i in range(NUM_EPISODES):
            total, steps = run_episode(env, seed=SEED + i)
            totals.append(total)
            print(f"Episode {i + 1}: steps={steps}, total reward={total:.2f}")

        totals = np.array(totals)
        print(f"Random baseline: mean={totals.mean():.2f}, std={totals.std():.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()