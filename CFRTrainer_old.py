import numpy as np
import itertools
import random
from informationSet import InformationSet

MAX_PLAYERS = 1000

class CFRTrainer():
    def __init__(self, imperfectNode):
        self.infoset_map = {}
        self.node = imperfectNode
        self.propnet = imperfectNode.propnet
        self.players = [role for role in self.propnet.roles if role != "random"]
        self.num_players = len(self.players)
        self.actions = {}

    def reset(self):
        """reset strategy sums"""
        for n in self.infoset_map.values():
            n.strategy_sum = np.zeros(n.num_actions)

    def get_information_set(self, player, player_state, data) -> InformationSet:
        """add if needed and return"""
        if player_state not in self.infoset_map:
            self.actions[player_state] = list(self.propnet.legal_moves_for(player, data))
            self.infoset_map[player_state] = InformationSet(len(self.actions[player_state]))
        return self.infoset_map[player_state]

    def get_counterfactual_reach_probability(self, probs: np.array, player: int):
        """compute counterfactual reach probability"""
        return np.prod(probs[:player]) * np.prod(probs[player + 1:])

    def cfr(self, data, reach_probabilities: np.array) -> np.array:
        if self.propnet.is_terminal(data):
            scores = self.propnet.scores(data)
            return np.array([scores[player] for player in self.players])

        data_num = self.node.data2num(data) * MAX_PLAYERS
        legal = self.propnet.legal_moves_dict(data)
        moves = []
        for role in self.propnet.roles:
            # moves.append(tuple(legal[role][i].input_id for i in random.sample(range(len(legal[role])), len(legal[role]))))
            moves.append(tuple(legal[role][i] for i in random.sample(range(len(legal[role])), len(legal[role]))))
        all_moves = list(itertools.product(*moves))

        info_sets = [None] * self.num_players
        strategies = [None] * self.num_players
        node_values = [0] * self.num_players
        new_reach_probabilities = reach_probabilities.copy()
        for i, player in enumerate(self.players):
            info_sets[i] = self.get_information_set(player, data_num+i, data)
            strategies[i] = info_sets[i].get_strategy(reach_probabilities[i])
            for ix, action in enumerate(self.actions[data_num+i]):
                new_reach_probabilities[i] *= strategies[i][ix]

        counterfactual_values = [np.zeros(len(self.actions[data_num+i])) for i in range(self.num_players)]
        move_counts = [{} for i in range(self.num_players)]
        for moves in all_moves:
            # print([move.gdl for move in moves])
            # print(self.propnet.roles)
            for player_num, player in enumerate(self.propnet.roles):
                if player == "random":
                    continue
                if moves[player_num] in move_counts[player_num]:
                    move_counts[player_num][moves[player_num]] += 1
                else:
                    move_counts[player_num][moves[player_num]] = 1
        for moves in all_moves:
            new_data = data.copy()
            self.propnet.do_step(new_data, [move.input_id for move in moves])
            # self.propnet.do_step(new_data, moves)
            res = self.cfr(new_data, new_reach_probabilities)
            for player_num, player in enumerate(self.propnet.roles):
                if player == "random":
                    continue
                print(counterfactual_values[player_num], res, move_counts[player_num][moves[player_num]])
                counterfactual_values[player_num] += res / move_counts[player_num][moves[player_num]]

        # Value of the current game state is just counterfactual values weighted by action probabilities
        for i, player in enumerate(self.players):
            # print(strategies[i], counterfactual_values[i])
            node_values += strategies[i].dot(counterfactual_values[i])  # counterfactual_values.dot(strategy)
        node_values /= len(self.players)
        for i, player in enumerate(self.players):
            for ix, action in enumerate(self.actions[data_num+i]):
                cf_reach_prob = self.get_counterfactual_reach_probability(reach_probabilities, i)
                regrets = counterfactual_values[i][ix] - node_values[i]
                info_sets[i].cumulative_regrets[ix] += cf_reach_prob * regrets
        return node_values

    def train_(self, num_iterations: int) -> int:
        utils = np.zeros(self.num_players)
        i = 0
        for data in self.node.generate_posible_games():
            utils += self.cfr(data, np.ones(self.num_players))
            i += 1
            if i > num_iterations:
                break
        return utils

    def train(self, num_iterations: int) -> int:
        self.train_(num_iterations//10)
        self.reset()
        utils = self.train_(num_iterations)
        # for infoset in self.infoset_map.items():
        #     info_set[0] =
        return utils

    def state_num2name(self, num):
        return "place holder"
