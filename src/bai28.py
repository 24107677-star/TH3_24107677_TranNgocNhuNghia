# ============================================================
# BÀI 28
# ============================================================

import numpy as np


Q = np.array([1.0, 5.0])

epsilon = 0.1

rng = np.random.default_rng(42)

counts = [0, 0]

for _ in range(10000):

    if rng.random() < epsilon:

        action = rng.integers(0, 2)

    else:

        action = int(
            np.argmax(Q)
        )

    counts[action] += 1


print("Q =", Q)

print("Epsilon =", epsilon)

print("Action 0:", counts[0])

print("Action 1:", counts[1])

print(
    "Action 0 probability:",
    counts[0] / 10000
)

print(
    "Action 1 probability:",
    counts[1] / 10000
)