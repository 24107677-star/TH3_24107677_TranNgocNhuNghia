import gymnasium as gym
import numpy as np
import sys

sys.path.append(".")

from mc_utils import generate_episode


env = gym.make(
    "Blackjack-v1"
)


def stick_on_20_policy(state):

    player_sum, dealer_card, usable_ace = state

    if player_sum >= 20:

        return 0

    return 1


wins = 0
losses = 0
draws = 0


for seed in range(100):

    episode = generate_episode(
        env,
        stick_on_20_policy,
        seed
    )

    total_reward = sum(
        x[2]
        for x in episode
    )

    if total_reward > 0:

        wins += 1

    elif total_reward < 0:

        losses += 1

    else:

        draws += 1


print("Wins:", wins)

print("Losses:", losses)

print("Draws:", draws)

print(
    "Win rate:",
    wins / 100
)


env.close()