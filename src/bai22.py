# ============================================================
# BÀI 22: Mean Absolute Difference
# ============================================================

import gymnasium as gym
import numpy as np
from collections import defaultdict


def generate_episode(env, policy, seed=None):

    state, info = env.reset(seed=seed)

    episode = []

    terminated = False
    truncated = False

    while not (terminated or truncated):

        action = policy(state)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))

        state = next_state

    return episode


def random_policy(state):
    return np.random.randint(0, 2)


def mc_prediction(env, n_episodes, first_visit=True):

    sums = defaultdict(float)
    counts = defaultdict(int)

    for i in range(n_episodes):

        episode = generate_episode(
            env,
            random_policy,
            seed=i
        )

        G = 0
        visited = set()

        for t in reversed(range(len(episode))):

            state, action, reward = episode[t]

            G = reward + G

            if first_visit:

                if state in visited:
                    continue

                visited.add(state)

            sums[state] += G
            counts[state] += 1

    return {
        s: sums[s] / counts[s]
        for s in sums
    }


env = gym.make("Blackjack-v1")

V_first = mc_prediction(
    env,
    10000,
    True
)

V_every = mc_prediction(
    env,
    10000,
    False
)

common = set(V_first) & set(V_every)

differences = [
    abs(V_first[s] - V_every[s])
    for s in common
]

print(
    "Mean Absolute Difference:",
    np.mean(differences)
)

env.close()