import numpy as np
from collections import defaultdict


# ============================================================
# BÀI 5 - GENERATE EPISODE
# ============================================================

def generate_episode(env, policy, seed=None):
    """
    Sinh một episode hoàn chỉnh.

    Mỗi phần tử:
        (state, action, reward)
    """

    state, info = env.reset(seed=seed)

    episode = []

    terminated = False
    truncated = False

    while not (terminated or truncated):

        action = int(policy(state))

        next_state, reward, terminated, truncated, info = env.step(action)

        episode.append(
            (state, action, reward)
        )

        state = next_state

    return episode


# ============================================================
# BÀI 7 - COMPUTE RETURN
# ============================================================

def compute_returns(rewards, gamma=1.0):
    """
    Tính G_t cho toàn bộ episode.

    G_t = R_(t+1) + gamma * G_(t+1)
    """

    returns = [0.0] * len(rewards)

    G = 0.0

    for t in range(len(rewards) - 1, -1, -1):

        G = rewards[t] + gamma * G

        returns[t] = G

    return returns


# ============================================================
# BÀI 17 - FIRST VISIT MC PREDICTION
# ============================================================

def first_visit_mc_prediction(
    env,
    policy,
    n_episodes,
    gamma=1.0,
    seed=42
):

    returns_sum = defaultdict(float)

    returns_count = defaultdict(int)

    rng = np.random.default_rng(seed)

    for episode_index in range(n_episodes):

        episode_seed = int(
            rng.integers(0, 2**32 - 1)
        )

        episode = generate_episode(
            env,
            policy,
            seed=episode_seed
        )

        rewards = [
            item[2]
            for item in episode
        ]

        Gs = compute_returns(
            rewards,
            gamma
        )

        visited_states = set()

        for t, (state, action, reward) in enumerate(episode):

            # First Visit
            if state in visited_states:
                continue

            visited_states.add(state)

            returns_sum[state] += Gs[t]

            returns_count[state] += 1

    V = {}

    for state in returns_sum:

        V[state] = (
            returns_sum[state]
            /
            returns_count[state]
        )

    return V, dict(returns_count)


# ============================================================
# BÀI 20 - EVERY VISIT MC PREDICTION
# ============================================================

def every_visit_mc_prediction(
    env,
    policy,
    n_episodes,
    gamma=1.0,
    seed=42
):

    returns_sum = defaultdict(float)

    returns_count = defaultdict(int)

    rng = np.random.default_rng(seed)

    for episode_index in range(n_episodes):

        episode_seed = int(
            rng.integers(0, 2**32 - 1)
        )

        episode = generate_episode(
            env,
            policy,
            seed=episode_seed
        )

        rewards = [
            item[2]
            for item in episode
        ]

        Gs = compute_returns(
            rewards,
            gamma
        )

        for t, (state, action, reward) in enumerate(episode):

            returns_sum[state] += Gs[t]

            returns_count[state] += 1

    V = {}

    for state in returns_sum:

        V[state] = (
            returns_sum[state]
            /
            returns_count[state]
        )

    return V, dict(returns_count)


# ============================================================
# BÀI 25 - MC ACTION VALUE
# ============================================================

def mc_action_value_prediction(
    env,
    policy,
    n_episodes,
    gamma=1.0,
    seed=42
):

    returns_sum = defaultdict(float)

    returns_count = defaultdict(int)

    rng = np.random.default_rng(seed)

    for episode_index in range(n_episodes):

        episode_seed = int(
            rng.integers(0, 2**32 - 1)
        )

        episode = generate_episode(
            env,
            policy,
            seed=episode_seed
        )

        rewards = [
            item[2]
            for item in episode
        ]

        Gs = compute_returns(
            rewards,
            gamma
        )

        for t, (state, action, reward) in enumerate(episode):

            key = (state, action)

            returns_sum[key] += Gs[t]

            returns_count[key] += 1

    Q = defaultdict(
        lambda: np.zeros(
            env.action_space.n
        )
    )

    for (state, action), total in returns_sum.items():

        Q[state][action] = (
            total
            /
            returns_count[
                (state, action)
            ]
        )

    return Q, dict(returns_count)


# ============================================================
# BÀI 26 - GREEDY ACTION
# ============================================================

def greedy_action(
    Q,
    state,
    n_actions
):

    if state not in Q:

        return np.random.randint(
            n_actions
        )

    return int(
        np.argmax(Q[state])
    )


# ============================================================
# BÀI 27 - EPSILON GREEDY
# ============================================================

def epsilon_greedy_action(
    Q,
    state,
    n_actions,
    epsilon,
    rng
):

    # Exploration
    if rng.random() < epsilon:

        return int(
            rng.integers(n_actions)
        )

    # State chưa xuất hiện
    if state not in Q:

        return int(
            rng.integers(n_actions)
        )

    # Exploitation
    return int(
        np.argmax(Q[state])
    )


# ============================================================
# BÀI 32 - ON POLICY MC CONTROL
# ============================================================

def on_policy_mc_control(
    env,
    n_episodes,
    gamma=1.0,
    epsilon=0.1,
    seed=42
):

    rng = np.random.default_rng(seed)

    Q = defaultdict(
        lambda: np.zeros(
            env.action_space.n
        )
    )

    returns_sum = defaultdict(float)

    returns_count = defaultdict(int)

    episode_rewards = []

    for episode_index in range(n_episodes):

        state, info = env.reset(
            seed=int(
                rng.integers(
                    0,
                    2**32 - 1
                )
            )
        )

        episode = []

        total_reward = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):

            action = epsilon_greedy_action(
                Q,
                state,
                env.action_space.n,
                epsilon,
                rng
            )

            next_state, reward, terminated, truncated, info = env.step(action)

            episode.append(
                (
                    state,
                    action,
                    reward
                )
            )

            total_reward += reward

            state = next_state

        episode_rewards.append(
            total_reward
        )

        rewards = [
            item[2]
            for item in episode
        ]

        Gs = compute_returns(
            rewards,
            gamma
        )

        visited_state_actions = set()

        for t, (state, action, reward) in enumerate(episode):

            key = (
                state,
                action
            )

            # First Visit
            if key in visited_state_actions:
                continue

            visited_state_actions.add(key)

            returns_sum[key] += Gs[t]

            returns_count[key] += 1

            Q[state][action] = (
                returns_sum[key]
                /
                returns_count[key]
            )

    policy = {}

    for state in Q:

        policy[state] = int(
            np.argmax(Q[state])
        )

    return (
        Q,
        policy,
        episode_rewards
    )


# ============================================================
# BÀI 34 - EVALUATE POLICY
# ============================================================

def evaluate_policy(
    env,
    policy,
    n_episodes=10000,
    seed=123
):

    rng = np.random.default_rng(seed)

    wins = 0
    losses = 0
    draws = 0

    rewards = []

    for episode_index in range(n_episodes):

        state, info = env.reset(
            seed=int(
                rng.integers(
                    0,
                    2**32 - 1
                )
            )
        )

        total_reward = 0

        terminated = False
        truncated = False

        while not (terminated or truncated):

            action = int(
                policy(state)
            )

            state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

        rewards.append(
            total_reward
        )

        if total_reward > 0:
            wins += 1

        elif total_reward < 0:
            losses += 1

        else:
            draws += 1

    return {

        "win_rate":
            wins / n_episodes,

        "loss_rate":
            losses / n_episodes,

        "draw_rate":
            draws / n_episodes,

        "mean_reward":
            float(
                np.mean(rewards)
            )
    }


# ============================================================
# MOVING AVERAGE
# ============================================================

def moving_average(
    values,
    window=1000
):

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) < window:

        return np.array([])

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid"
    )