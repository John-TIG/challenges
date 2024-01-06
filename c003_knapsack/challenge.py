from typing import Union, Dict
from dataclasses import dataclass
import numpy as np

@dataclass
class Difficulty:
    # The number of items from which you need to select a subset to put in the knapsack
    num_items: int
    # Can be interpreted as a percentage by dividing by 10
    # The value of items you select must be at least (better_than_baseline / 10)% higher than the baseline greedy algorithm
    better_than_baseline: int

@dataclass
class Challenge:
    seed: int
    difficulty: Difficulty
    weights: np.ndarray
    values: np.ndarray
    max_weight: int
    min_value: int

    def verifySolution(self, solution: list) -> bool:
        try:
            selected_items = np.array(solution, dtype=int)
            if len(selected_items.shape) != 1 or len(selected_items) > len(self.weights) or set(selected_items) - set(range(len(self.weights))) or len(selected_items) != len(set(selected_items)):
                raise Exception("Invalid items. Expecting a subset of [0, ..., num_items - 1]")        
            total_weight = np.sum(self.weights[selected_items])
            total_value = np.sum(self.values[selected_items])
            return total_weight <= self.max_weight and total_value >= self.min_value
        except Exception as e:
            raise Exception("Invalid solution") from e

    @classmethod
    def generateInstance(cls, seed: int, difficulty: Union[Difficulty, Dict[str, int]]):
        if isinstance(difficulty, dict):
            difficulty = Difficulty(**difficulty)
        np.random.seed(seed)
        weights = np.random.randint(1, 50, size=difficulty.num_items)
        values = np.random.randint(1, 50, size=difficulty.num_items)
        max_weight = int(np.sum(weights) / 2)
        # baseline greedy algorithm
        sorted_value_to_weight_ratio = np.argsort(-values / weights)
        total_weight, min_value = 0, 0
        for item in sorted_value_to_weight_ratio:
            if total_weight + weights[item] > max_weight:
                continue
            min_value += values[item]
            total_weight += weights[item]
        min_value = int(min_value * (1 + difficulty.better_than_baseline / 1000))
        return cls(
            seed=seed,
            difficulty=difficulty,
            weights=weights,
            values=values,
            max_weight=max_weight,
            min_value=min_value
        )