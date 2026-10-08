import gymnasium as gym
import sys

sys.path.append(".")

from mc_utils import (
    generate_episode,
    compute_returns
)


env = gym.make(
    "Blackjack-v1"
)


def random_policy(state):

    return env.action_space.sample()


episode = generate_episode(
    env,
    random_policy,
    seed=42
)


rewards = [
    x[2]
    for x in episode
]


returns = compute_returns(
    rewards,
    gamma=0.9
)


trajectory = []

for item, G in zip(
    episode,
    returns
):

    state, action, reward = item

    trajectory.append(
        (
            state,
            action,
            reward,
            G
        )
    )


for item in trajectory:

    print(item)


env.close()