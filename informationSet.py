import numpy as np

class InformationSet():
    def __init__(self, num_actions, is_random_player):
        self.cumulative_regrets = np.zeros(shape=num_actions)
        self.strategy_sum = np.zeros(shape=num_actions)
        self.num_actions = num_actions
        self.is_random_player = is_random_player

    def normalize(self, strategy: np.array) -> np.array:
        """Normalize a strategy. If there are no positive regrets,
        use a uniform random strategy"""
        if sum(strategy) > 0:
            strategy /= sum(strategy)
        else:
            strategy = np.array([1.0 / self.num_actions] * self.num_actions)
        return strategy

    def get_strategy(self, reach_probability: float) -> np.array:
        """Return regret-matching strategy"""
        if self.is_random_player:
            return np.array([1.0 / self.num_actions] * self.num_actions)
        strategy = np.maximum(0, self.cumulative_regrets)
        strategy = self.normalize(strategy)

        self.strategy_sum += reach_probability * strategy
        return strategy

    def get_average_strategy(self) -> np.array:
        if self.is_random_player:
            return np.array([1.0 / self.num_actions] * self.num_actions)
        return self.normalize(self.strategy_sum.copy())

    def get_average_strategy_with_threshold(self, threshold: float) -> np.array:
        if self.is_random_player:
            return np.array([1.0 / self.num_actions] * self.num_actions)
        avg_strat = self.get_average_strategy()
        avg_strat[avg_strat < threshold] = 0
        return self.normalize(avg_strat)
