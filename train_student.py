from imperfectNode_fast_invalids import ImperfectNode
from CFRTrainer_imperfect import CFRTrainerImperfect
from CFRTrainer_perfect import CFRTrainerPerfect
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
model.load_most_recent(game+"_student")
model.save(game + "_tru_dist", 0)

num_iterations = 8

# @profile
def do_game(cur, propnet, model):
    game_printer.reset()

    states = []
    imperfectNodes = {role: ImperfectNode(propnet, cur.data.copy(), role) for role in propnet.roles}
    for step in range(1000):
        start = time.time()

        depth = 0
        while time.time() - start < 0.2:
            depth += 1
            cfr_trainer = CFRTrainerPerfect(cur, depth, model)
            utils = cfr_trainer.train()
        print(model.eval(propnet.get_state(cur.data)))
        formatted_probs = {}
        valid_probs = {}
        moves_dict = propnet.legal_moves_dict(cur.data)
        moves_dict_ids = {role: [x.id for x in moves_dict[role]] for role in moves_dict}
        for role in propnet.roles:
            policy = iter(cfr_trainer.get_root_policy_for_player(role))
            formatted_probs[role] = [0] * len(propnet.legal_for[role])
            valid_probs[role] = []
            for i, legal in enumerate(propnet.legal_for[role]):
                if legal.id in moves_dict_ids[role]:
                    valid_probs[role].append(next(policy))
                    formatted_probs[role][i] = valid_probs[role][-1]

            total = sum(formatted_probs[role])
            assert 0.99 < total < 1.01, (total, role, formatted_probs[role], formatted_probs)

        state = propnet.get_state(cur.data)
        qs = {role: utils[i] for i, role in enumerate(propnet.roles)}
        states.append((state, formatted_probs, qs))



        depth = 0
        cfr_trainers = {}
        start2 = time.time()
        while time.time() - start2 < 0.1:
            depth += 1
            if depth > 1:
                print("Depth: ", depth)
            for role, node in imperfectNodes.items():
                cfr_trainers[role] = CFRTrainerImperfect(node, depth, model)
                if role == "random":
                    continue
                cfr_trainers[role].train(random.randint(1, num_iterations))
        valid_probs = {}
        moves_dict = propnet.legal_moves_dict(cur.data)
        moves_dict_ids = {role: [x.id for x in moves_dict[role]] for role in moves_dict}
        for role in propnet.roles:
            policy = iter(cfr_trainers[role].get_root_policy_for_player(role, num_iterations))
            valid_probs[role] = []
            for i, legal in enumerate(propnet.legal_for[role]):
                if legal.id in moves_dict_ids[role]:
                    valid_probs[role].append(next(policy))

            total = sum(valid_probs[role])
            assert 0.99 < total < 1.01, (total, role, len(list(policy)), sum(cfr_trainers[role].get_root_policy_for_player(role, num_iterations*2)), valid_probs)

        moves = []
        for player in propnet.roles:
            choice = random.random()
            for i, p in enumerate(valid_probs[player]):
                if choice < p:
                    moves.append(moves_dict_ids[player][i])
                    break
                else:
                    choice -= p

        # cur = cur.get_or_make_child(tuple(moves))

        data = cur.data.copy()
        propnet.do_sees_step(data, tuple(moves))
        visible = propnet.visible_dict(data)
        for i, role in enumerate(propnet.roles):
            imperfectNodes[role].add_history((moves[i], propnet.sees_ids_for(role, data)))
        data = data.copy()
        propnet.do_non_sees_step(data, tuple(moves))
        cur.data = data.copy()
        # game_printer.make_moves(moves, propnet)
        game_printer.make_moves(moves, propnet)
        game_printer.print_moves()

        print(f'Play took {start2 - start:.4f} seconds and {time.time() - start2:.4f}')
        if propnet.is_terminal(data):
            break
    for s, p, q in states:
        model.add_sample(s, p, q)


start = time.time()
for i in range(50000):
    cur = PerfectNode(propnet, data.copy())
    print('Game number', i)
    do_game(cur, propnet, model)
    model.train(epochs=10)
    if i and i % 50 == 0:
        model.save(game + "_tru_dist", i)
        with open(f'models/times-{game + "_tru_dist"}', 'a') as f:
            f.write(f'{i} {time.time()-start}\n')
