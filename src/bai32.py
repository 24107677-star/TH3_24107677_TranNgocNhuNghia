# ============================================================
# BÀI 32: On-Policy First-Visit Monte Carlo Control
# ============================================================

import gymnasium as gym
import numpy as np
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

    returns_sum = defaultdict(
        lambda: np.zeros(n_actions)
    )

    returns_count = defaultdict(
        lambda: np.zeros(
            n_actions,
            dtype=int
        )
    )

    rng = np.random.default_rng(42)

    for episode_number in range(n_episodes):

        state, info = env.reset(
            seed=episode_number
        )

        episode = []

        terminated = False
        truncated = False

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

            state = next_state

        G = 0.0
        visited = set()

        for t in reversed(range(len(episode))):

            state, action, reward = episode[t]

            G = reward + gamma * G

            key = (state, action)

            if key not in visited:

                visited.add(key)

                returns_sum[state][action] += G

                returns_count[state][action] += 1

                Q[state][action] = (
                    returns_sum[state][action]
                    /
                    returns_count[state][action]
                )

    return Q


env = gym.make("Blackjack-v1")

Q = train_mc_control(
    env,
    n_episodes=100000,
    gamma=1.0,
    epsilon=0.1
)

print(
    "Number of learned states:",
    len(Q)
)

print("\nSample learned Q values:")

for state, values in list(Q.items())[:10]:

    print(
        state,
        "->",
        values
    )

env.close()