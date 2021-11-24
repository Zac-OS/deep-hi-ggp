import numpy as np
import itertools
import random
from informationSet import InformationSet
from CFRTrainer import CFRTrainer

# def tonum(data):
#     return str(str(int("".join(str(int(x)) for x in data), 2)).__hash__())[:5]

class CFRTrainerImperfect(CFRTrainer):
    def __init__(self, node, depth=999999999, model=None, max_time=30):
        super().__init__(node, depth, model, max_time)

    def get_root_policy_for_player(self, player, num_iterations, approximate_for_traing=True):
        player_num = self.players.index(player)
        data = next(self.node.generate_posible_games())
        if data is None:
            policy = np.ones(len(list(self.propnet.legal_for[self.players[player_num]])))
            return policy / policy.shape[0]
        if player == "random":
            policy = np.ones(len(list(self.propnet.legal_moves_for(self.players[player_num], data))))
            return policy / policy.shape[0]
        policy = np.zeros(len(list(self.propnet.legal_moves_for(self.players[player_num], data))))
        if approximate_for_traing:
            num_states = 0
            for state in self.seen_states[player]:
                if self.infoset_map[state].num_actions == policy.shape[0]:
                    policy += self.infoset_map[state].get_average_strategy()
                    num_states += 1
            if num_states > 0:
                policy /= num_states
            else:
                policy += 1 / policy.shape[0]
            print("num_states: ", num_states)
        else:
            i = 1
            for data in self.node.generate_posible_games():
                state = self.node.propnet.data2num(data) * self.num_players + player_num
                if state in self.infoset_map:
                    policy += self.infoset_map[state].get_average_strategy()
                    i += 1
                    if i > num_iterations:
                        break
            policy /= num_iterations
        return policy

    def data_generator(self):
        return self.node.generate_posible_games()

    def train(self, num_iterations: int) -> int:
        if not self.model:
            self.train_(num_iterations//10)
            self.reset()
        utils = self.train_(num_iterations)
        return utils
