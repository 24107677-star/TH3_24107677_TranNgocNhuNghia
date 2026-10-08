import gymnasium as gym

env = gym.make("Blackjack-v1")

state, info = env.reset(seed=42)

terminated = False
truncated = False

while not (terminated or truncated):

    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)

    print("-------------------------")

    print("State:", state)

    print(
        "Action:",
        action
    )

    print(
        "Reward:",
        reward
    )

    print(
        "Next state:",
        next_state
    )

    print(
        "Terminated:",
        terminated
    )

    print(
        "Truncated:",
        truncated
    )

    state = next_state

env.close()