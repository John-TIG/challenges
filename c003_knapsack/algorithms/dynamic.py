from c003_knapsack.challenge import Challenge
import numpy as np

def solveChallenge(challenge: Challenge) -> list:
    max_weight = challenge.max_weight
    min_value = challenge.min_value
    num_items = challenge.difficulty.num_items
    combinations = [[]] + [None for _ in range(max_weight)]
    combination_values = [0 for _ in range(max_weight + 1)]
    # prioritize high value to weight items
    sorted_items = np.argsort(challenge.values / challenge.weights)[::-1]
    for i in range(challenge.weights.shape[0]):
        item = int(sorted_items[i])
        # create new combos with current item
        for weight in reversed(range(max_weight + 1)):
            if combinations[weight] is None:
                continue
            new_weight = weight + challenge.weights[item]
            new_value = combination_values[weight] + challenge.values[item]
            new_combo = combinations[weight] + [item]
            if new_weight > max_weight:
                continue
            if new_value >= min_value:
                return new_combo
            if new_value > combination_values[new_weight]:
                combinations[new_weight] = new_combo
                combination_values[new_weight] = new_value
    return []