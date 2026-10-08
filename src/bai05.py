import gymnasium as gym
import sys

sys.path.append(".")

from mc_utils import generate_episode


env = gym.make("Blackjack-v1")


def random_policy(state):

    return env.action_space.sample()


episode = generate_episode(
    env,
    random_policy,
    seed=42
)


print("Episode:")

for step in episode:

    print(step)


env.close()