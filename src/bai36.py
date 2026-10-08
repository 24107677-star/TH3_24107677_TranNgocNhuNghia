# ============================================================
# BÀI 36: Monte Carlo Blackjack Mini Project
# ============================================================

import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# ============================================================
# 1. Epsilon-Greedy
# ============================================================

def epsilon_greedy_action(
    Q,
    state,
    n_actions,
    epsilon,
    rng
):

    if rng.random() < epsilon:

        return int(
            rng.integers(0, n_actions)
        )

    values = Q[state]

    max_value = np.max(values)

    best_actions = np.flatnonzero(
        values == max_value
    )

    return int(
        rng.choice(best_actions)
    )


# ============================================================
# 2. Train MC Control
# ============================================================

def train_mc_control(
    env,
    n_episodes=100000,
    gamma=1.0,
    epsilon=0.1
):

    n_actions = env.action_space.n

    Q = defaultdict(
        lambda: np.zeros(n_actions)
    )

    returns_sum = defaultdict(
        lambda: np.zeros(n_actions)
    )

    returns_count = defaultdict(
        lambda: np.zeros(
            n_actions,
            dtype=int
        )
    )

    rng = np.random.default_rng(42)

    rewards = []

    for episode_number in range(n_episodes):

        state, info = env.reset(
            seed=episode_number
        )

        episode = []

        terminated = False
        truncated = False

        total_reward = 0

        while not (terminated or truncated):

            action = epsilon_greedy_action(
                Q,
                state,
                n_actions,
                epsilon,
                rng
            )

            next_state, reward, terminated, truncated, info = env.step(
                action
            )

            episode.append(
                (state, action, reward)
            )

            total_reward += reward

            state = next_state

        rewards.append(
            total_reward
        )

        # First-Visit MC
        G = 0.0
        visited = set()

        for t in reversed(range(len(episode))):

            state, action, reward = episode[t]

            G = reward + gamma * G

            key = (state, action)

            if key not in visited:

                visited.add(key)

                returns_sum[state][action] += G

                returns_count[state][action] += 1

                Q[state][action] = (
                    returns_sum[state][action]
                    /
                    returns_count[state][action]
                )

    return Q, rewards


# ============================================================
# 3. Evaluate Learned Policy
# ============================================================

def evaluate_policy(
    env,
    Q,
    n_episodes=10000
):

    rewards = []

    for i in range(n_episodes):

        state, info = env.reset(
            seed=1000000 + i
        )

        terminated = False
        truncated = False

        total_reward = 0

        while not (terminated or truncated):

            action = int(
                np.argmax(Q[state])
            )

            next_state, reward, terminated, truncated, info = env.step(
                action
            )

            total_reward += reward

            state = next_state

        rewards.append(
            total_reward
        )

    rewards = np.array(rewards)

    return {
        "average_reward": np.mean(rewards),
        "win_rate": np.mean(rewards > 0),
        "draw_rate": np.mean(rewards == 0),
        "loss_rate": np.mean(rewards < 0)
    }


# ============================================================
# 4. Random Policy
# ============================================================

def random_policy(state):

    return np.random.randint(0, 2)


def evaluate_random_policy(
    env,
    n_episodes=10000
):

    rewards = []

    for i in range(n_episodes):

        state, info = env.reset(
            seed=2000000 + i
        )

        terminated = False
        truncated = False

        total_reward = 0

        while not (terminated or truncated):

            action = random_policy(state)

            next_state, reward, terminated, truncated, info = env.step(
                action
            )

            total_reward += reward

            state = next_state

        rewards.append(
            total_reward
        )

    rewards = np.array(rewards)

    return {
        "average_reward": np.mean(rewards),
        "win_rate": np.mean(rewards > 0),
        "draw_rate": np.mean(rewards == 0),
        "loss_rate": np.mean(rewards < 0)
    }


# ============================================================
# 5. Main
# ============================================================

env = gym.make("Blackjack-v1")

print("=" * 60)
print("MONTE CARLO BLACKJACK MINI PROJECT")
print("=" * 60)


print("\nTraining...")

Q, training_rewards = train_mc_control(
    env,
    n_episodes=100000,
    gamma=1.0,
    epsilon=0.1
)

print("Training completed.")

print(
    "\nLearned states:",
    len(Q)
)


# ============================================================
# 6. Evaluate
# ============================================================

print("\nEvaluating Random Policy...")

random_result = evaluate_random_policy(
    env,
    10000
)

print(random_result)


print("\nEvaluating Learned Policy...")

learned_result = evaluate_policy(
    env,
    Q,
    10000
)

print(learned_result)


# ============================================================
# 7. Performance comparison
# ============================================================

labels = [
    "Random",
    "MC Learned"
]

average_rewards = [
    random_result["average_reward"],
    learned_result["average_reward"]
]


os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(8, 5))

plt.bar(
    labels,
    average_rewards
)

plt.ylabel("Average reward")

plt.title(
    "Random Policy vs Monte Carlo Learned Policy"
)

plt.grid(
    axis="y"
)

plt.savefig(
    "figures/policy_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 8. Learning curve
# ============================================================

window = 1000

moving_average = np.convolve(
    training_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(10, 5))

plt.plot(
    moving_average
)

plt.xlabel("Episode")
plt.ylabel("Average reward")

plt.title(
    "Monte Carlo Control Learning Curve"
)

plt.grid(True)

plt.savefig(
    "figures/mc_convergence.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 9. Inspect sample states
# ============================================================

sample_states = [
    (20, 10, False),
    (18, 10, False),
    (15, 5, False),
    (20, 10, True)
]

print("\nSample learned policy:")

for state in sample_states:

    print(
        state,
        "Q =",
        Q[state],
        "Action =",
        int(np.argmax(Q[state]))
    )


print("\nFigures saved in:")
print("figures/")

env.close()