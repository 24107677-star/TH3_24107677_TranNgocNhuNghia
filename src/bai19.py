# ============================================================
# BÀI 19: Blackjack value heatmap
# ============================================================

import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
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


def first_visit_mc(env, n_episodes):

    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)

    for i in range(n_episodes):

        episode = generate_episode(
            env,
            random_policy,
            seed=i
        )

        G = 0.0
        visited = set()

        for t in reversed(range(len(episode))):

            state, action, reward = episode[t]

            G = reward + G

            if state not in visited:

                visited.add(state)

                returns_sum[state] += G
                returns_count[state] += 1

    V = {}

    for state in returns_sum:
        V[state] = (
            returns_sum[state]
            / returns_count[state]
        )

    return V


env = gym.make("Blackjack-v1")

V = first_visit_mc(
    env,
    50000
)


# usable_ace = False
player_sums = range(12, 22)
dealer_cards = range(1, 11)

heatmap = np.full(
    (len(player_sums), len(dealer_cards)),
    np.nan
)

for i, player_sum in enumerate(player_sums):

    for j, dealer_card in enumerate(dealer_cards):

        state = (
            player_sum,
            dealer_card,
            False
        )

        if state in V:
            heatmap[i, j] = V[state]


os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(10, 6))

plt.imshow(
    heatmap,
    origin="lower",
    aspect="auto"
)

plt.colorbar(label="Value V(s)")

plt.xticks(
    range(10),
    dealer_cards
)

plt.yticks(
    range(len(player_sums)),
    player_sums
)

plt.xlabel("Dealer showing card")
plt.ylabel("Player sum")

plt.title(
    "Blackjack State Value - No Usable Ace"
)

plt.savefig(
    "figures/blackjack_value_no_usable_ace.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

env.close()

print(
    "Saved: figures/blackjack_value_no_usable_ace.png"
)