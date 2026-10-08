# ============================================================
# BÀI 35: MC Control Learning Curve
# ============================================================

import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def epsilon_greedy_action(
    Q,
    state,
    n_actions,
    epsilon,
    rng
):

    if rng.random() < epsilon:

        return int(
            rng.integers(0, n_actions)
        )

    values = Q[state]

    max_value = np.max(values)

    best_actions = np.flatnonzero(
        values == max_value
    )

    return int(
        rng.choice(best_actions)
    )


def train_mc_control(
    env,
    n_episodes=100000,
    gamma=1.0,
    epsilon=0.1
):

    n_actions = env.action_space.n

    Q = defaultdict(
        lambda: np.zeros(n_actions)
    )

    sums = defaultdict(
        lambda: np.zeros(n_actions)
    )

    counts = defaultdict(
        lambda: np.zeros(
            n_actions,
            dtype=int
        )
    )

    rng = np.random.default_rng(42)

    episode_rewards = []

    for episode_number in range(n_episodes):

        state, info = env.reset(
            seed=episode_number
        )

        episode = []

        terminated = False
        truncated = False

        total_reward = 0

        while not (terminated or truncated):

            action = epsilon_greedy_action(
                Q,
                state,
                n_actions,
                epsilon,
                rng
            )

            next_state, reward, terminated, truncated, info = env.step(
                action
            )

            episode.append(
                (state, action, reward)
            )

            total_reward += reward

            state = next_state

        episode_rewards.append(
            total_reward
        )

        G = 0
        visited = set()

        for t in reversed(range(len(episode))):

            state, action, reward = episode[t]

            G = reward + gamma * G

            key = (state, action)

            if key not in visited:

                visited.add(key)

                sums[state][action] += G

                counts[state][action] += 1

                Q[state][action] = (
                    sums[state][action]
                    /
                    counts[state][action]
                )

    return Q, episode_rewards


env = gym.make("Blackjack-v1")

Q, rewards = train_mc_control(
    env,
    n_episodes=100000,
    gamma=1.0,
    epsilon=0.1
)


# Moving average
window = 1000

moving_average = np.convolve(
    rewards,
    np.ones(window) / window,
    mode="valid"
)


os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(10, 5))

plt.plot(
    moving_average
)

plt.xlabel("Episode")
plt.ylabel("Average reward")

plt.title(
    "Monte Carlo Control Learning Curve"
)

plt.grid(True)

plt.savefig(
    "figures/mc_convergence.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved: figures/mc_convergence.png"
)

env.close()