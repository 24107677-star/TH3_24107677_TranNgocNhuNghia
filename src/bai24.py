# ============================================================
# BÀI 24: Collect returns for state-action pairs
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


env = gym.make("Blackjack-v1")

returns = defaultdict(list)

for episode_number in range(10000):

    episode = generate_episode(
        env,
        random_policy,
        seed=episode_number
    )

    G = 0

    for t in reversed(range(len(episode))):

        state, action, reward = episode[t]

        G = reward + G

        returns[(state, action)].append(G)


print(
    "Number of state-action pairs:",
    len(returns)
)

for key, values in list(returns.items())[:10]:

    print(
        key,
        "->",
        values[:5]
    )

env.close()