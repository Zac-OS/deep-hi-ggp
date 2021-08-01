import numpy as np
import itertools
import random
from informationSet import InformationSet

class CFRTrainer():
    def __init__(self, imperfectNode):
        self.infoset_map = {}
        self.node = imperfectNode
        self.propnet = imperfectNode.propnet
        self.players = self.propnet.roles
        self.num_players = len(self.players)
        self.state_names = {}
        # self.players = [role for role in self.propnet.roles if role != "random"]
        # self.num_players = len(self.players)

    def reset(self):
        """reset strategy sums"""
        for n in self.infoset_map.values():
            n.strategy_sum = np.zeros(n.num_actions)

    def get_information_set(self, player_state, num_actions, is_random_player) -> InformationSet:
        """add if needed and return"""
        if player_state not in self.infoset_map:
            self.infoset_map[player_state] = InformationSet(num_actions, is_random_player)
        return self.infoset_map[player_state]

    def get_counterfactual_reach_probability(self, probs: np.array, player: int):
        """compute counterfactual reach probability"""
        return np.prod(probs[:player]) * np.prod(probs[player + 1:])

    def cfr(self, data, reach_probabilities: np.array, active_player: int, moves) -> np.array:
        if self.propnet.is_terminal(data):
            scores = self.propnet.scores(data)
            # print(scores)
            return np.array([scores[player] if player != "random" else 0 for player in self.players])

        legal_moves = list(self.propnet.legal_moves_for(self.players[active_player], data))
        assert len(legal_moves) > 0

        data_num = self.node.data2num(data) * self.num_players + active_player
        info_set = self.get_information_set(data_num, len(legal_moves), self.players[active_player] == "random")
        self.state_names[data_num] = f"{self.players[active_player]}: {[m.gdl for m in legal_moves]}"

        strategy = info_set.get_strategy(reach_probabilities[active_player])
        next_player = (active_player + 1) % self.num_players

        counterfactual_values = [None] * len(legal_moves)

        data_copy = data
        for ix, action in enumerate(legal_moves):
            action_probability = strategy[ix]

            # compute new reach probabilities after this action
            new_reach_probabilities = reach_probabilities.copy()
            new_reach_probabilities[active_player] *= action_probability

            moves_copy = moves.copy()
            # moves_copy.append(action.input_id)
            moves_copy.append(action)
            if next_player == 0:
                data_copy = data.copy()
                self.propnet.do_step(data_copy, [move.input_id for move in moves_copy])
                moves_copy = []

            # recursively call cfr method, next player to act is the opponent
            counterfactual_values[ix] = self.cfr(data_copy, new_reach_probabilities, next_player, moves_copy)

        # Value of the current game state is just counterfactual values weighted by action probabilities

        node_values = strategy.dot(counterfactual_values)  # counterfactual_values.dot(strategy)
        for ix, action in enumerate(legal_moves):
            cf_reach_prob = self.get_counterfactual_reach_probability(reach_probabilities, active_player)
            regrets = counterfactual_values[ix][active_player] - node_values[active_player]
            info_set.cumulative_regrets[ix] += cf_reach_prob * regrets
        return node_values

    def get_root_policy_for_player(self, player, num_iterations):
        player_num = self.players.index(player)
        data = next(self.node.generate_posible_games())
        policy = np.zeros(len(list(self.propnet.legal_moves_for(self.players[player_num], data))))
        i = 1
        for data in self.node.generate_posible_games():
            state = self.node.data2num(data) * self.num_players + player_num
            if state in self.infoset_map:
                # print(self.infoset_map[state].get_average_strategy())
                policy += self.infoset_map[state].get_average_strategy()
                i += 1
                # if i % 10 == 0:
                #     print(i)
                if i > num_iterations:
                    break
        return policy / num_iterations

    def train_(self, num_iterations: int) -> int:
        utils = np.zeros(self.num_players)
        i = 0
        for data in self.node.generate_posible_games():
            utils += self.cfr(data, np.ones(self.num_players), 0, [])
            i += 1
            if i > num_iterations:
                break
            # if i % 10 == 0:
            #     print(i)
        return utils

    def train(self, num_iterations: int) -> int:
        self.train_(num_iterations//10)
        self.reset()
        utils = self.train_(num_iterations)
        return utils
