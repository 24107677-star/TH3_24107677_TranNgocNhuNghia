import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# BÀI 10: Compare G0 with different discount factors gamma
# ============================================================

# ------------------------------------------------------------
# 1. Generate an episode
# ------------------------------------------------------------
def generate_episode(env, policy, seed=None):
    state, info = env.reset(seed=seed)

    episode = []

    terminated = False
    truncated = False

    while not (terminated or truncated):

        action = policy(state)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append((state, action, reward))

        state = next_state

    return episode


# ------------------------------------------------------------
# 2. Random policy
# ------------------------------------------------------------
def random_policy(state):
    return np.random.randint(0, 2)


# ------------------------------------------------------------
# 3. Calculate G0
# ------------------------------------------------------------
def calculate_return(episode, gamma):

    G = 0.0

    for _, _, reward in reversed(episode):
        G = reward + gamma * G

    return G


# ------------------------------------------------------------
# 4. Create environment
# ------------------------------------------------------------
env = gym.make("Blackjack-v1")


# ------------------------------------------------------------
# 5. Generate ONE episode
# ------------------------------------------------------------
episode = generate_episode(
    env,
    random_policy,
    seed=42
)


print("Episode:")
print(episode)

print("\nNumber of steps:", len(episode))


# ------------------------------------------------------------
# 6. Compare different gamma values
# ------------------------------------------------------------
gammas = [0.0, 0.5, 0.9, 0.99, 1.0]

returns = []

print("\nReturn G0 for different gamma:")

for gamma in gammas:

    G0 = calculate_return(
        episode,
        gamma
    )

    returns.append(G0)

    print(f"Gamma = {gamma:<4} -> G0 = {G0}")


# ------------------------------------------------------------
# 7. Create figures folder
# ------------------------------------------------------------
os.makedirs("figures", exist_ok=True)


# ------------------------------------------------------------
# 8. Plot
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    gammas,
    returns,
    marker="o"
)

plt.xlabel("Gamma (γ)")
plt.ylabel("Return G0")

plt.title("Effect of Discount Factor Gamma on Return G0")

plt.grid(True)

plt.xticks(gammas)


# ------------------------------------------------------------
# 9. Save figure
# ------------------------------------------------------------
save_path = "figures/gamma_comparison.png"

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\nFigure saved successfully:")
print(save_path)


# ------------------------------------------------------------
# 10. Close environment
# ------------------------------------------------------------
env.close()