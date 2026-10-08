import gymnasium as gym

env = gym.make("Blackjack-v1")

for i in range(10):

    state, info = env.reset()

    player_sum, dealer_card, usable_ace = state

    print(
        f"Episode {i+1}:",
        f"player_sum={player_sum},",
        f"dealer_showing={dealer_card},",
        f"usable_ace={usable_ace}"
    )

env.close()