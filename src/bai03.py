import gymnasium as gym

env = gym.make("Blackjack-v1")

ACTION_NAMES = {
    0: "STICK",
    1: "HIT"
}

print("Action space:", env.action_space)

for action in range(env.action_space.n):

    print(
        action,
        "->",
        ACTION_NAMES[action]
    )

env.close()