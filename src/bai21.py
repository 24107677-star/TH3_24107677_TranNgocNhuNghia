# ============================================================
# BÀI 21: Compare First-Visit and Every-Visit MC
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


def first_visit(env, n_episodes):

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

            if state not in visited:

                visited.add(state)

                sums[state] += G
                counts[state] += 1

    return {
        s: sums[s] / counts[s]
        for s in sums
    }


def every_visit(env, n_episodes):

    sums = defaultdict(float)
    counts = defaultdict(int)

    for i in range(n_episodes):

        episode = generate_episode(
            env,
            random_policy,
            seed=i
        )

        G = 0

        for t in reversed(range(len(episode))):

            state, action, reward = episode[t]

            G = reward + G

            sums[state] += G
            counts[state] += 1

    return {
        s: sums[s] / counts[s]
        for s in sums
    }


env = gym.make("Blackjack-v1")

V_first = first_visit(env, 10000)
V_every = every_visit(env, 10000)

common_states = set(V_first) & set(V_every)

print("Common states:", len(common_states))

for state in list(common_states)[:10]:

    print(
        state,
        "| First-Visit:",
        round(V_first[state], 4),
        "| Every-Visit:",
        round(V_every[state], 4)
    )

env.close()