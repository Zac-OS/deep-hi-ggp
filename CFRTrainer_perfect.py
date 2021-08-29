import numpy as np
import itertools
import random
from informationSet import InformationSet
from CFRTrainer import CFRTrainer

class CFRTrainerPerfect(CFRTrainer):
    def __init__(self, node, depth=999999999, model=None):
        super().__init__(node, depth, model)

    def get_root_policy_for_player(self, player, num_iterations=None):
        player_num = self.players.index(player)
        state = self.propnet.data2num(self.node.data) * self.num_players + player_num
        return self.infoset_map[state].get_average_strategy()

    def data_generator(self):
        while True:
            yield self.node.data.copy()
