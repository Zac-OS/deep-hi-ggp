from imperfectNode_fast_invalids import ImperfectNode
from CFRTrainer_imperfect import CFRTrainerImperfect
from perfectNode import PerfectNode
from propnet import load_propnet
from model import Model
import time
import sys
import random
import numpy as np
from print_conect4 import PrintConect4

game = sys.argv[1]

game_printer = PrintConect4(game)

data, propnet = load_propnet(game)
model = Model(propnet)
model.load_most_recent(game)
model.save(game + "_student", 0)

num_iterations = 10

# @profile
def do_game(cur, propnet, model):
    game_printer.reset()

    states = []
    imperfectNodes = {role: ImperfectNode(propnet, cur.data.copy(), role) for role in propnet.roles}
    for step in range(1000):
        start = time.time()
        depth = 0
        while time.time() - start < 0.1:
            depth += 1
            utils = {}
            cfr_trainers = {}
            for role, node in imperfectNodes.items():
                cfr_trainers[role] = CFRTrainerImperfect(node, depth, model)
                utils[role] = cfr_trainers[role].train(num_iterations)
        print(model.eval(propnet.get_state(cur.data)))
        formatted_probs = {}
        valid_probs = {}
        moves_dict = propnet.legal_moves_dict(cur.data)
        moves_dict_ids = {role: [x.id for x in moves_dict[role]] for role in moves_dict}
        for role in propnet.roles:
            policy = cfr_trainers[role].get_root_policy_for_player(role, num_iterations*2)
            formatted_probs[role] = np.zeros(len(propnet.legal_for[role]))
            valid_probs[role] = []
            for i, legal in enumerate(propnet.legal_for[role]):
                if legal.id in moves_dict_ids[role]:
                    valid_probs[role].append(policy[i])
                    formatted_probs[role][i] = valid_probs[role][-1]

            total = sum(formatted_probs[role])
            if total < 0.01:
                print("!!!!!!!!! very low total: ", total)
            formatted_probs[role] /= total
            valid_probs[role] = np.array(valid_probs[role]) / total

            total = sum(formatted_probs[role])
            assert 0.99 < total < 1.01, (total, role, len(list(policy)), sum(cfr_trainers[role].get_root_policy_for_player(role, num_iterations*2)), formatted_probs)
            total = sum(valid_probs[role])
            assert 0.99 < total < 1.01, (total, role, len(list(policy)), sum(cfr_trainers[role].get_root_policy_for_player(role, num_iterations*2)), valid_probs)

        state = propnet.get_state(cur.data)
        qs = {role: utils[role][i] for i, role in enumerate(propnet.roles)}
        states.append((state, formatted_probs, qs))
        moves = []
        for player in propnet.roles:
            choice = random.random()
            for i, p in enumerate(valid_probs[player]):
                if choice < p:
                    moves.append(moves_dict_ids[player][i])
                    break
                else:
                    choice -= p

        game_printer.make_moves(moves, propnet)
        game_printer.print_moves()

        cur = cur.get_or_make_child(tuple(moves))
        print('Play took %.4f seconds' % (time.time() - start))
        if cur.terminal:
            break
    scores = cur.scores
    for s, p, q in states:
        model.add_sample(s, p, q)


start = time.time()
for i in range(50000):
    cur = PerfectNode(propnet, data.copy())
    print('Game number', i)
    do_game(cur, propnet, model)
    model.train(epochs=10)
    if i and i % 50 == 0:
        model.save(game + "_student", i)
        with open(f'models/times-{game + "_student"}', 'a') as f:
            f.write(f'{i} {time.time()-start}\n')
