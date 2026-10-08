# ============================================================
# BÀI 23: First-Visit vs Every-Visit convergence
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


def estimate_state(
    env,
    target_state,
    checkpoints
):

    first_values = []
    every_values = []

    first_sum = 0
    first_count = 0

    every_sum = 0
    every_count = 0

    for episode_number in range(1, max(checkpoints) + 1):

        episode = generate_episode(
            env,
            random_policy,
            seed=episode_number
        )

        G = 0

        visited = False

        for t in reversed(range(len(episode))):

            state, action, reward = episode[t]

            G = reward + G

            if state == target_state:

                every_sum += G
                every_count += 1

                if not visited:

                    first_sum += G
                    first_count += 1

                    visited = True

        if episode_number in checkpoints:

            first_value = (
                first_sum / first_count
                if first_count > 0 else np.nan
            )

            every_value = (
                every_sum / every_count
                if every_count > 0 else np.nan
            )

            first_values.append(first_value)
            every_values.append(every_value)

    return first_values, every_values


env = gym.make("Blackjack-v1")

checkpoints = [
    100,
    500,
    1000,
    5000,
    10000,
    20000,
    50000
]

states = [
    (20, 10, False),
    (18, 10, False),
    (15, 5, False)
]

os.makedirs("figures", exist_ok=True)

all_results = {}

for target_state in states:

    first, every = estimate_state(
        env,
        target_state,
        checkpoints
    )

    all_results[target_state] = (
        first,
        every
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        checkpoints,
        first,
        marker="o",
        label="First-Visit"
    )

    plt.plot(
        checkpoints,
        every,
        marker="s",
        label="Every-Visit"
    )

    plt.xlabel("Number of episodes")
    plt.ylabel("Estimated V(s)")

    plt.title(
        f"Convergence for state {target_state}"
    )

    plt.legend()
    plt.grid(True)

    filename = (
        "figures/first_vs_every_visit_"
        + str(target_state[0])
        + "_"
        + str(target_state[1])
        + "_"
        + str(target_state[2])
        + ".png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# Combined required figure
plt.figure(figsize=(9, 5))

target_state = states[0]

first, every = all_results[target_state]

plt.plot(
    checkpoints,
    first,
    marker="o",
    label="First-Visit"
)

plt.plot(
    checkpoints,
    every,
    marker="s",
    label="Every-Visit"
)

plt.xlabel("Number of episodes")
plt.ylabel("Estimated V(s)")

plt.title(
    f"First-Visit vs Every-Visit: {target_state}"
)

plt.legend()
plt.grid(True)

plt.savefig(
    "figures/first_vs_every_visit.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

env.close()

print("Saved convergence figures.")