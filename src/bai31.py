# ============================================================
# BÀI 31: Incremental mean update
# ============================================================

def incremental_mean_update(
    old_mean,
    count,
    new_value
):

    new_mean = (
        old_mean
        + (new_value - old_mean)
        / (count + 1)
    )

    return new_mean


mean = 0.0
count = 0

returns = [1, -1, 1, 1, -1]

for G in returns:

    mean = incremental_mean_update(
        mean,
        count,
        G
    )

    count += 1

    print(
        f"n = {count}, "
        f"G = {G}, "
        f"mean = {mean:.4f}"
    )