# ============================================================
# BÀI 16: Find first-visit indices of states
# ============================================================

def get_first_visit_indices(episode):
    first_visit = {}

    for t, (state, action, reward) in enumerate(episode):
        if state not in first_visit:
            first_visit[state] = t

    return first_visit


# Example episode
episode = [
    ("A", 0, 1),
    ("B", 1, 0),
    ("A", 0, 2),
    ("C", 1, 1),
    ("B", 0, -1)
]

result = get_first_visit_indices(episode)

print("Episode:")
print(episode)

print("\nFirst-visit indices:")

for state, index in result.items():
    print(f"State {state}: t = {index}")