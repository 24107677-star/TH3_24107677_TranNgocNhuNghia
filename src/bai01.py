import gymnasium as gym

env = gym.make("Blackjack-v1")

print("Observation space:")
print(env.observation_space)

print("\nAction space:")
print(env.action_space)

state, info = env.reset()

print("\nObservation:")
print(state)

print("\nInfo:")
print(info)

env.close()