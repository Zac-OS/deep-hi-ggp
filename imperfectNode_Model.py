import random

class Node():
    def __init__(self, probs, moves, my_move):
        self.my_move = my_move
        self.probs = probs
        self.moves = moves
        self.size = 1
        for role in self.moves:
            total = 0
            count = 0
            for id in self.probs[role]:
                if id in self.moves[role]:
                    count += 1
                else:
                    self.probs[role][id] = 0
            for id in self.probs[role]:
                if id in self.moves[role]:
                    self.probs[role][id] += 1/count
                    total += self.probs[role][id]
            if not total:
                continue
            for id in self.probs[role]:
                if id in self.moves[role]:
                    self.probs[role][id] /= total
            self.size *= max(1, count)
        self.seen = set()


    def choose_single(self, role):
        prob = random.random()
        for id, p in self.probs[role].items():
            if prob < p and id in self.moves[role]:
                return id
            prob -= p
        return None

    def __iter__(self):
        self.seen = set()
        return self

    def __next__(self):
        if len(self.seen) == self.size:
            raise StopIteration
        res = [self.my_move]
        for role in self.moves:
            if role == "random":
                res.append(random.choice(self.moves[role]))
            else:
                res.append(self.choose_single(role))

        res = tuple(res)
        self.seen.add(res)
        return res

class ImperfectNode:
    def __init__(self, propnet, data, role, model, cache):
        self.propnet = propnet
        self.data = data
        self.history = []
        self.role = role
        self.model = model
        self.other_roles = [role for role in self.propnet.roles if role != self.role]
        self.invalid = {}
        self.move_generator = {}
        self.cache = cache

    def add_history(self, round):
        self.history.append(round)

    def generate_posible_games(self):
        while True:
            x = self.generate_single_state(list(self.data.values()), 0)
            assert x != -1, "No valid set of moves"
            if x is not None:
                yield x

    # @profile
    def generate_single_state(self, data, depth):
        if depth == len(self.history):
            return data
        legal = self.propnet.legal_moves_dict(data)
        state_num = self.propnet.data2num(data)
        if state_num not in self.invalid:
            self.invalid[state_num] = set()

        moves = self.choose_move(legal, depth, state_num, data.copy())
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

    # @profile
    def choose_move(self, legal, depth, state_num, data):
        if state_num in self.move_generator:
            for moves in self.move_generator[state_num]:
                if moves not in self.invalid[state_num]:
                    return moves
            # print("empty path")
            return None

        moves = {}
        for role in self.other_roles:
            moves[role] = tuple(legal[role][i].input_id for i in random.sample(range(len(legal[role])), len(legal[role])))
        if state_num in self.cache:
            self.move_generator[state_num] = self.cache[state_num]
        else:
            out = self.model.eval(self.propnet.get_state(data))[0]
            new_out = {}
            for role in out:
                new_out[role] = {self.propnet.id_to_move[key].input_id: val for key, val in out[role].items()}
                # out[role] = {self.propnet.id_to_move[key].input_id: val for key, val in out[role].items()}
            self.move_generator[state_num] = Node(new_out, moves, self.propnet.id_to_move[self.history[depth][0]].input_id)
            self.cache[state_num] = self.move_generator[state_num]

        for moves in self.move_generator[state_num]:
            if moves not in self.invalid[state_num]:
                return moves
            else:
                print("b")
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
