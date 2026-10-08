# ============================================================
# BÀI 27: Epsilon-Greedy action selection
# ============================================================

import numpy as np


def epsilon_greedy_action(
    Q,
    state,
    n_actions,
    epsilon,
    rng
):

    if rng.random() < epsilon:

        return rng.integers(
            0,
            n_actions
        )

    values = [
        Q.get((state, action), 0.0)
        for action in range(n_actions)
    ]

    return int(np.argmax(values))


Q = {
    ("state", 0): 1.0,
    ("state", 1): 5.0
}

rng = np.random.default_rng(42)

epsilon = 0.1

counts = [0, 0]

for _ in range(10000):

    action = epsilon_greedy_action(
        Q,
        "state",
        2,
        epsilon,
        rng
    )

    counts[action] += 1


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