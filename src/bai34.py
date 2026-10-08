# ============================================================
# BÀI 34: Evaluate policies
# ============================================================

import gymnasium as gym
import numpy as np


def evaluate_policy(
    env,
    policy,
    n_episodes=10000
):

    rewards = []

    for i in range(n_episodes):

        state, info = env.reset(
            seed=100000 + i
        )

        terminated = False
        truncated = False

        total_reward = 0

        while not (terminated or truncated):

            action = policy(state)

            next_state, reward, terminated, truncated, info = env.step(
                action
            )

            total_reward += reward

            state = next_state

        rewards.append(total_reward)

    rewards = np.array(rewards)

    win_rate = np.mean(
        rewards > 0
    )

    draw_rate = np.mean(
        rewards == 0
    )

    loss_rate = np.mean(
        rewards < 0
    )

    return {
        "average_reward": np.mean(rewards),
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "loss_rate": loss_rate
    }


env = gym.make("Blackjack-v1")


# Random policy
def random_policy(state):
    return np.random.randint(0, 2)


# Fixed policy
def fixed_policy(state):

    player_sum, dealer_card, usable_ace = state

    if player_sum >= 20:
        return 0

    return 1


print("Random policy:")

print(
    evaluate_policy(
        env,
        random_policy
    )
)


print("\nFixed policy:")

print(
    evaluate_policy(
        env,
        fixed_policy
    )
)


env.close()