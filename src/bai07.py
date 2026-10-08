import sys

sys.path.append(".")

from mc_utils import compute_returns


rewards = [
    0,
    0,
    1
]


returns = compute_returns(
    rewards,
    gamma=1.0
)


print("Rewards:", rewards)

print("Returns:", returns)