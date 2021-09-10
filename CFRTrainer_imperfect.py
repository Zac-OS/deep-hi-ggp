import numpy as np
import itertools
import random
from informationSet import InformationSet
from CFRTrainer import CFRTrainer

class CFRTrainerImperfect(CFRTrainer):
    def __init__(self, node, depth=999999999, model=None):
        super().__init__(node, depth, model)

    def get_root_policy_for_player(self, player, num_iterations):
        player_num = self.players.index(player)
        data = next(self.node.generate_posible_games())
        policy = np.zeros(len(list(self.propnet.legal_moves_for(self.players[player_num], data))))
        i = 1
        for data in self.node.generate_posible_games():
            state = self.node.propnet.data2num(data) * self.num_players + player_num
            if state in self.infoset_map:
                # print(self.infoset_map[state].get_average_strategy())
                policy += self.infoset_map[state].get_average_strategy()
                i += 1
                # if i % 10 == 0:
                #     print(i)
                if i > num_iterations:
                    break
        return policy / num_iterations

    def data_generator(self):
        return self.node.generate_posible_games()

    def train(self, num_iterations: int) -> int:
        if not self.model:
            self.train_(num_iterations//10)
            self.reset()
        utils = self.train_(num_iterations) / (num_iterations+1)
        return utils
