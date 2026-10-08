import gymnasium as gym
from collections import defaultdict
import numpy as np
import sys

sys.path.append(".")

from mc_utils import (
    generate_episode,
    compute_returns
)


env = gym.make(
    "Blackjack-v1"
)


def stick_on_20_policy(state):

    return (
        0
        if state[0] >= 20
        else 1
    )


returns = defaultdict(list)


for seed in range(1000):

    episode = generate_episode(
        env,
        stick_on_20_policy,
        seed
    )

    rewards = [
        x[2]
        for x in episode
    ]

    Gs = compute_returns(
        rewards
    )

    for t, item in enumerate(episode):

        state = item[0]

        returns[state].append(
            Gs[t]
        )


V = {
    state: np.mean(values)
    for state, values
    in returns.items()
}


for state in list(V)[:10]:

    print(
        state,
        "->",
        V[state]
    )


env.close()