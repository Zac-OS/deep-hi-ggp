from perfectNode import PerfectNode
import random

# store all known information for each player + no probabilities
# also store true state for game runner
# have a generator for possible states based on this players knowledge

class ImperfectNode:
    def __init__(self, propnet, data, role):
        self.propnet = propnet
        self.data = data
        self.history = []
        self.role = role

    def add_history(self, round):
        self.history.append(round)

    def generate_posible_games(self):
        """Maybe come back to add a cache between rounds
        Not implemented yet as it might bias towards states
        similar to previously sampled on earlier rounds.
        Also could change to a dfs of bfs if it is taking a long time
        in copying the data for states"""
        print(self.history)
        while True:
            print("start---------------")
            x = self.generate_single_state(list(self.data.values()), 0)
            if x is not None:
                print("end---------------")
                yield x
            else:
                print("failed----------------------------------------")

    def generate_single_state(self, data, depth):
        # assert(valid_data(self.propnet.visible(data), depth))
        if depth == len(self.history):
            return data
        legal = self.propnet.legal_moves_dict(data)
        roles = [role for role in self.propnet.roles if role != self.role]

        moves = [self.history[depth][0]]
        # print("we do:", moves[0])
        # for move in legal.values():
        #     moves.append(random.choice(move).input_id)
        for role in roles:
            x = random.choice(legal[role])
            print(f"{role} make move:", x.move_gdl)
            moves.append(x.input_id)
        self.propnet.do_step(data, moves)
        if self.propnet.is_terminal(data):
            return None
        if not self.valid_data(data, depth):
            return None
        return self.generate_single_state(data, depth+1)


    def valid_data(self, data, depth):
        visible = self.propnet.visible_ids(data)
        if len(visible) != len(self.history[depth][1]):
            return False
        for i, x in enumerate(self.history[depth][1]):
            if x != visible[i]:
                return False
        return True
