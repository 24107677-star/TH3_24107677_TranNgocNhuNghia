import sys

sys.path.append(".")

from mc_utils import compute_returns


rewards = [
    0,
    0,
    1
]


for gamma in [
    1.0,
    0.9,
    0.5
]:

    returns = compute_returns(
        rewards,
        gamma
    )

    print(
        "Gamma =",
        gamma,
        "Returns =",
        returns
    )