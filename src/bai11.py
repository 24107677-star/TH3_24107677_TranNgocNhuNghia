def stick_on_20_policy(state):

    player_sum, dealer_card, usable_ace = state

    if player_sum >= 20:

        return 0

    return 1


print(
    stick_on_20_policy(
        (20, 10, False)
    )
)

print(
    stick_on_20_policy(
        (18, 6, False)
    )
)