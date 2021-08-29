from perfectNode import PerfectNode
import random
import itertools

# store all known information for each player + no probabilities
# also store true state for game runner
# have a generator for possible states based on this players knowledge

class Node:
    def __init__(self, legal, moves, other_roles, prob):
        # print("starting with", prob, moves)

        # self.children = []
        self.prob = prob

        if legal is None:
            self.tmp = True
            return

        self.probs = []
        self.child_moves = []
        self.child_nodes = []
        self.finished = False
        self.tmp = False


        for role in other_roles:
            moves.append(tuple(legal[role][i].input_id for i in random.sample(range(len(legal[role])), len(legal[role]))))
        self.moves = list(itertools.product(*moves))
        self.move_generator = iter(self.moves)
        self.num_children = len(self.moves)


    def update_prob(self):
        assert self.finished
        # print("updating", self.prob, self.child_nodes)
        prob = 0
        new_child_nodes = [None] * self.num_children
        new_child_moves = [None] * self.num_children
        new_probs = [None] * self.num_children
        count = 0
        for i, child in enumerate(self.child_nodes):
            if child.prob > 0:
                prob += child.prob
                new_probs[count] = child.prob
                new_child_nodes[count] = child
                new_child_moves[count] = self.child_moves[i]
                count += 1

        self.probs = new_probs[:count]
        self.child_moves = new_child_moves[:count]
        self.child_nodes = new_child_nodes[:count]
        self.prob = prob
        for i in range(count):
            self.probs[i] /= self.prob

        self.move_generator = self.prob_generator()

    def prob_generator(self):
        for _ in range(10):
            yield self.choose_child()
            self.update_prob()
        yield self.choose_child()

    def choose_child(self):
        prob = random.random()
        for i, p in enumerate(self.probs):
            if prob < p:
                return self.child_moves[i]
            prob -= p
        return None
        assert False, "sum of child probs is less than 1"


class ImperfectNode:
    def __init__(self, propnet, data, role):
        self.propnet = propnet
        self.data = data
        self.history = []
        self.role = role
        self.other_roles = [role for role in self.propnet.roles if role != self.role]
        self.nodes = {}
        self.leaf_nodes = {}

    def add_history(self, round):
        self.history.append(round)

    def generate_posible_games(self):
        while True:
            x = self.generate_single_state(list(self.data.values()), 0)
            assert x != -1, "No valid set of moves"
            if x is not None:
                yield x

    def generate_single_state(self, data, depth, parentNode=None):
        state_num = self.propnet.data2num(data)
        if state_num not in self.nodes:
            # assert parentNode is None
            # print(state_num.__hash__(), depth, parentNode)
            # if parentNode is not None:
            #     exit()
            if depth == len(self.history):
                self.nodes[state_num] = Node(None, None, self.other_roles, parentNode.prob/parentNode.num_children if parentNode else 1)
            else:
                self.nodes[state_num] = Node(self.propnet.legal_moves_dict(data), [[self.history[depth][0]]], self.other_roles, parentNode.prob/parentNode.num_children if parentNode else 1)
            if parentNode:
                parentNode.child_nodes.append(self.nodes[state_num])

        if depth == len(self.history):
            # print(self.data2num(data).__hash__())
            # if parentNode and state_num not in self.leaf_nodes:
            #     self.leaf_nodes[state_num] = Node(None, None, None, parentNode.prob/parentNode.num_children)
            #     parentNode.child_nodes.append(self.leaf_nodes[state_num])
            return data

        node = self.nodes[state_num]
        if node.tmp:
            node.__init__(self.propnet.legal_moves_dict(data), [[self.history[depth][0]]], self.other_roles, parentNode.prob/parentNode.num_children if parentNode else 1)

        moves = self.choose_move(state_num)
        if moves is None:
            return -1

        self.propnet.do_sees_step(data, tuple(moves))
        if not self.valid_data(data, moves, depth):
            return None

        self.propnet.do_non_sees_step(data, moves)
        if self.propnet.is_terminal(data):
            return None

        res = self.generate_single_state(data, depth+1, node)
        if res == -1:
            return None
        node.child_moves.append(moves)
        # x = Node(self.propnet.legal_moves_dict(data), [[self.history[depth+1][0]]], self.other_roles, self.nodes[state_num].prob/self.nodes[state_num].num_children)
        # self.nodes[state_num].child_nodes.append(x)
        # self.nodes[self.data2num(data)] = x
        return res

    def choose_move(self, state_num):
        node = self.nodes[state_num]

        for moves in node.move_generator:
            return moves


        # node.move_generator = iter(node.moves)
        # for moves in node.move_generator:
        #     return moves




        assert not node.finished, "finished generator a second time"
        node.finished = True

        node.update_prob()
        if len(node.child_moves) == 0:
            return None

        for moves in node.move_generator:
            return moves

        assert False, "umm shouldn't be here i think"
        return None

    def valid_data(self, data, moves, depth):
        visible = self.propnet.sees_ids_for(self.role, data)
        if len(visible) != len(self.history[depth][1]):
            return False
        for i, x in enumerate(self.history[depth][1]):
            if x != visible[i]:
                return False
        return True
