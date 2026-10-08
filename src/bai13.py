import gymnasium as gym
from collections import defaultdict
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

    player_sum, dealer_card, usable_ace = state

    return 0 if player_sum >= 20 else 1


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


print(
    "Number of states:",
    len(returns)
)


env.close()