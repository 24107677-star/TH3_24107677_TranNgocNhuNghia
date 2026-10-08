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

rewards = [
    reward
    for state, action, reward in episode
]

print("Episode length:", len(episode))

print("Rewards:", rewards)

print("Total reward:", sum(rewards))


env.close()