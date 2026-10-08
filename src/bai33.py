# ============================================================
# BÀI 33: Train Blackjack with MC Control
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

    return int(
        np.argmax(Q[state])
    )


def train(
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

    for i in range(n_episodes):

        state, info = env.reset(seed=i)

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

    return Q


env = gym.make("Blackjack-v1")

Q = train(env)

print("Training completed.")

sample_states = [
    (20, 10, False),
    (18, 10, False),
    (15, 5, False),
    (20, 10, True)
]

print("\nSample states:")

for state in sample_states:

    print(
        state,
        "Q =",
        Q[state],
        "Best action =",
        int(np.argmax(Q[state]))
    )

env.close()