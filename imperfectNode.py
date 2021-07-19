from perfectNode import PerfectNode
import random
import itertools

# store all known information for each player + no probabilities
# also store true state for game runner
# have a generator for possible states based on this players knowledge

class ImperfectNode:
    def __init__(self, propnet, data, role):
        self.propnet = propnet
        self.data = data
        self.history = []
        self.role = role
        self.other_roles = [role for role in self.propnet.roles if role != self.role]
        self.invalid = {}
        self.move_generator = {}

    def add_history(self, round):
        self.history.append(round)

    def generate_posible_games(self):
        """Maybe come back to add a cache between rounds
        Not implemented yet as it might bias towards states
        similar to previously sampled on earlier rounds.
        Also could change to a dfs of bfs if it is taking a long time
        in copying the data for states"""
        while True:
            x = self.generate_single_state(list(self.data.values()), 0)
            assert x != -1, "No valid set of moves"
            if x is not None:
                yield x

    def generate_single_state(self, data, depth):
        if depth == len(self.history):
            return data
        legal = self.propnet.legal_moves_dict(data)
        state_num = self.data2num(data)
        if state_num not in self.invalid:
            self.invalid[state_num] = set()

        moves = self.choose_move(legal, depth, state_num)
        if moves is None:
            return -1
        if not self.valid_data(data, moves, depth):
            self.invalid[state_num].add(moves)
            return None

        self.propnet.do_non_sees_step(data, moves)
        if self.propnet.is_terminal(data):
            self.invalid[state_num].add(moves)
            return None

        res = self.generate_single_state(data, depth+1)
        if res == -1:
            self.invalid[state_num].add(moves)
            return None
        return res

    def data2num(self, data):
        return int("".join(str(int(x)) for x in data), 2)

    def set_generator(self, legal, depth, state_num):
        moves = [[self.history[depth][0]]]
        for role in self.other_roles:
            moves.append(tuple(legal[role][i].input_id for i in random.sample(range(len(legal[role])), len(legal[role]))))

        self.move_generator[state_num] = itertools.product(*moves)

    def choose_move(self, legal, depth, state_num):
        if state_num not in self.move_generator:
            self.set_generator(legal, depth, state_num)
        else:
            for moves in self.move_generator[state_num]:
                if moves not in self.invalid[state_num]:
                    return moves
            self.set_generator(legal, depth, state_num)

        for moves in self.move_generator[state_num]:
            if moves not in self.invalid[state_num]:
                return moves
        return None

    def valid_data(self, data, moves, depth):
        self.propnet.do_sees_step(data, tuple(moves))
        visible = self.propnet.sees_ids_for(self.role, data)
        if len(visible) != len(self.history[depth][1]):
            return False
        for i, x in enumerate(self.history[depth][1]):
            if x != visible[i]:
                return False
        return True
