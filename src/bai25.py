# ============================================================
# BÀI 25: Monte Carlo Action-Value Prediction
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


def mc_action_value_prediction(
    env,
    policy,
    n_episodes,
    gamma=1.0
):

    sums = defaultdict(float)
    counts = defaultdict(int)

    for episode_number in range(n_episodes):

        episode = generate_episode(
            env,
            policy,
            seed=episode_number
        )

        G = 0

        for t in reversed(range(len(episode))):

            state, action, reward = episode[t]

            G = reward + gamma * G

            key = (state, action)

            sums[key] += G
            counts[key] += 1

    Q = {}

    for key in sums:

        Q[key] = (
            sums[key]
            / counts[key]
        )

    return Q


env = gym.make("Blackjack-v1")

Q = mc_action_value_prediction(
    env,
    random_policy,
    10000
)

print(
    "Number of state-action pairs:",
    len(Q)
)

for key, value in list(Q.items())[:10]:

    print(
        key,
        "->",
        round(value, 4)
    )

env.close()