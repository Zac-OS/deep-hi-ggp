import numpy as np
import itertools
import random
from informationSet import InformationSet
from CFRTrainer import CFRTrainer

class CFRTrainerPerfect(CFRTrainer):
    def __init__(self, node, depth=999999999, model=None, max_time=999999999):
        super().__init__(node, depth, model, max_time)

    def get_root_policy_for_player(self, player, num_iterations=None):
        player_num = self.players.index(player)
        state = self.propnet.data2num(self.node.data) * self.num_players + player_num
        return self.infoset_map[state].get_average_strategy()

    def data_generator(self):
        while True:
            yield self.node.data.copy()

    def train(self, num_iterations: int) -> int:
        if not self.model:
            self.train_(num_iterations//10)
            self.reset()
        # utils = self.train_(1)
        utils = self.train_(num_iterations)
        return utils
