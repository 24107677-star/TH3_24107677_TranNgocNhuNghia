import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# BÀI 15: Track the return of a specific state
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
# 2. Fixed policy
#    If player sum >= 20 -> Stick
#    Otherwise -> Hit
# ------------------------------------------------------------
def stick_on_20_policy(state):

    player_sum, dealer_card, usable_ace = state

    if player_sum >= 20:
        return 0       # Stick
    else:
        return 1       # Hit


# ------------------------------------------------------------
# 3. Calculate return Gt
# ------------------------------------------------------------
def calculate_returns(episode, gamma=1.0):

    returns = [0.0] * len(episode)

    G = 0.0

    for t in reversed(range(len(episode))):

        reward = episode[t][2]

        G = reward + gamma * G

        returns[t] = G

    return returns


# ------------------------------------------------------------
# 4. Target state
# ------------------------------------------------------------
TARGET_STATE = (20, 10, False)


# ------------------------------------------------------------
# 5. Create environment
# ------------------------------------------------------------
env = gym.make("Blackjack-v1")


# ------------------------------------------------------------
# 6. Collect returns for target state
# ------------------------------------------------------------
state_returns = []

N_EPISODES = 10000

for episode_number in range(N_EPISODES):

    episode = generate_episode(
        env,
        stick_on_20_policy,
        seed=episode_number
    )

    returns = calculate_returns(
        episode,
        gamma=1.0
    )

    # Check every state in the episode
    for t, (state, action, reward) in enumerate(episode):

        if state == TARGET_STATE:

            state_returns.append(returns[t])


# ------------------------------------------------------------
# 7. Print results
# ------------------------------------------------------------
print("Target state:", TARGET_STATE)

print("Number of visits:", len(state_returns))

if len(state_returns) > 0:

    print("Returns:", state_returns[:20])

    print(
        "Average return:",
        np.mean(state_returns)
    )

    print(
        "Minimum return:",
        np.min(state_returns)
    )

    print(
        "Maximum return:",
        np.max(state_returns)
    )

else:

    print("Target state was not visited.")


# ------------------------------------------------------------
# 8. Plot returns
# ------------------------------------------------------------
if len(state_returns) > 0:

    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(9, 5))

    plt.plot(
        range(1, len(state_returns) + 1),
        state_returns,
        marker="o",
        markersize=3,
        linewidth=1
    )

    plt.axhline(
        np.mean(state_returns),
        linestyle="--",
        label="Average Return"
    )

    plt.xlabel("Visit number")
    plt.ylabel("Return Gt")

    plt.title(
        "Returns for State (20, 10, False)"
    )

    plt.legend()

    plt.grid(True)

    save_path = "figures/mc_convergence_state.png"

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\nFigure saved successfully:")
    print(save_path)


# ------------------------------------------------------------
# 9. Close environment
# ------------------------------------------------------------
env.close()