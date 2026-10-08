# ============================================================
# BÀI 20: Every-Visit Monte Carlo Prediction
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


def every_visit_mc_prediction(
    env,
    policy,
    n_episodes,
    gamma=1.0
):

    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)

    V = {}

    for episode_number in range(n_episodes):

        episode = generate_episode(
            env,
            policy,
            seed=episode_number
        )

        G = 0.0

        for t in reversed(range(len(episode))):

            state, action, reward = episode[t]

            G = reward + gamma * G

            returns_sum[state] += G
            returns_count[state] += 1

            V[state] = (
                returns_sum[state]
                / returns_count[state]
            )

    return V


env = gym.make("Blackjack-v1")

V = every_visit_mc_prediction(
    env,
    random_policy,
    10000
)

print("Number of states:", len(V))

for state, value in list(V.items())[:10]:
    print(state, "->", value)

env.close()