# ============================================================
# BÀI 29: Compare different epsilon values
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt


Q = np.array([1.0, 5.0])

epsilons = [
    0.01,
    0.05,
    0.10,
    0.20,
    0.50
]

rng = np.random.default_rng(42)

action_0_probabilities = []

for epsilon in epsilons:

    counts = [0, 0]

    for _ in range(10000):

        if rng.random() < epsilon:

            action = rng.integers(0, 2)

        else:

            action = int(
                np.argmax(Q)
            )

        counts[action] += 1

    probability = (
        counts[0] / 10000
    )

    action_0_probabilities.append(
        probability
    )


os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(8, 5))

plt.plot(
    epsilons,
    action_0_probabilities,
    marker="o"
)

plt.xlabel("Epsilon")

plt.ylabel(
    "Probability of selecting Action 0"
)

plt.title(
    "Effect of Epsilon on Exploration"
)

plt.grid(True)

plt.savefig(
    "figures/epsilon_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "Saved: figures/epsilon_comparison.png"
)