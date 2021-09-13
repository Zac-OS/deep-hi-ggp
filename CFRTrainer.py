import numpy as np
import itertools
import random
from informationSet import InformationSet

class CFRTrainer():
    def __init__(self, node, depth=999999999, model=None):
        self.infoset_map = {}
        self.node = node
        self.propnet = node.propnet
        self.players = self.propnet.roles
        self.seen_states = {role: [] for role in self.players}
        self.num_players = len(self.players)
        self.state_names = {}
        self.depth = depth
        self.model = model

        # self.players = [role for role in self.propnet.roles if role != "random"]
        # self.num_players = len(self.players)

    def reset(self):
        """reset strategy sums"""
        for n in self.infoset_map.values():
            n.strategy_sum = np.zeros(n.num_actions)

    def get_information_set(self, player_state, num_actions, player, init_policy=None) -> InformationSet:
        """add if needed and return"""
        if player_state not in self.infoset_map:
            self.seen_states[player].append(player_state)
            self.infoset_map[player_state] = InformationSet(num_actions, player == "random", init_policy)
        return self.infoset_map[player_state]

    def get_counterfactual_reach_probability(self, probs: np.array, player: int):
        """compute counterfactual reach probability"""
        return np.prod(probs[:player]) * np.prod(probs[player + 1:])

    # def get_scores(self, data):
    #     _, values = self.model.eval(self.propnet.get_state(data))
    #     return np.array([values[player] for player in values if player != "random"])

    # @profile
    def cfr(self, data, reach_probabilities: np.array, active_player: int, moves, probs, values, depth, data_num) -> np.array:
        if self.propnet.is_terminal(data):
            scores = self.propnet.scores(data)
            return np.array([scores[player] if player != "random" else 0 for player in self.players])

        if depth == self.depth:
            return np.array([values[player] if player != "random" else 0 for player in values])

        assert depth < self.depth

        legal_moves = list(self.propnet.legal_moves_for(self.players[active_player], data))
        assert len(legal_moves) > 0

        state_num = data_num + active_player
        if self.model:
            probs_dict = probs[self.players[active_player]]
            probs_array = np.array([probs_dict[move.id] for move in legal_moves])
            info_set = self.get_information_set(state_num, len(legal_moves), self.players[active_player], probs_array)
        else:
            info_set = self.get_information_set(state_num, len(legal_moves), self.players[active_player])
        self.state_names[state_num] = f"{self.players[active_player]}: {[m.gdl for m in legal_moves]}"

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
            moves_copy.append(action)
            if next_player == 0:
                data_copy = data.copy()
                self.propnet.do_step(data_copy, [move.input_id for move in moves_copy])
                next_data_num = self.propnet.data2num(data_copy) * self.num_players
                moves_copy = []
                if self.model:
                    probs, values = self.model.eval(self.propnet.get_state(data_copy))
                counterfactual_values[ix] = self.cfr(data_copy, new_reach_probabilities, next_player, moves_copy, probs, values, depth+1, next_data_num)
            else:
                counterfactual_values[ix] = self.cfr(data_copy, new_reach_probabilities, next_player, moves_copy, probs, values, depth, data_num)

        # Value of the current game state is just counterfactual values weighted by action probabilities
        node_values = strategy.dot(counterfactual_values)
        for ix, action in enumerate(legal_moves):
            cf_reach_prob = self.get_counterfactual_reach_probability(reach_probabilities, active_player)
            regrets = counterfactual_values[ix][active_player] - node_values[active_player]
            info_set.cumulative_regrets[ix] += cf_reach_prob * regrets
        return node_values

    def train_(self, num_iterations: int) -> int:
        utils = np.zeros(self.num_players)
        i = 0
        for data in self.data_generator():
            probs, values = None, None
            if self.model:
                probs, values = self.model.eval(self.propnet.get_state(data))
            utils += self.cfr(data, np.ones(self.num_players), 0, [], probs, values, 0, self.propnet.data2num(data) * self.num_players)
            i += 1
            if i == num_iterations:
                break
            # print(i)
            # if i % 10 == 0:
            #     print(i)
        return utils
