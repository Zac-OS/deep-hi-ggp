from imperfectNode_fast_invalids import ImperfectNode
# from imperfectNode import ImperfectNode
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
model.load_most_recent(game)
model.save(game, 0)
# model.load_most_recent(game+"_tru_dist")
# model.save(game + "_tru_dist", 0)

num_iterations = 6

def tonum(data):
    return str(str(int("".join(str(int(x)) for x in data), 2)).__hash__())[:5]

# @profile
def do_game(cur, propnet, model):
    game_printer.reset()

    data = cur.data
    states = []
    imperfectNodes = {role: ImperfectNode(propnet, data.copy(), role) for role in propnet.roles}
    for step in range(1000):
        start = time.time()

        depth = 0
        while time.time() - start < 0.2:
            depth += 1
            cfr_trainer = CFRTrainerPerfect(cur, depth, model)
            utils = cfr_trainer.train(random.randint(num_iterations//2 + 1, num_iterations*2))
            # break

        print(model.eval(propnet.get_state(data)))
        formatted_probs = {}
        valid_probs = {}
        moves_dict = propnet.legal_moves_dict(data)
        moves_dict_ids = {role: [x.id for x in moves_dict[role]] for role in moves_dict}
        for role in propnet.roles:
            policy = cfr_trainer.get_root_policy_for_player(role)
            print("first", role, policy)
            policy = iter(policy)
            formatted_probs[role] = [0] * len(propnet.legal_for[role])
            valid_probs[role] = []
            for i, legal in enumerate(propnet.legal_for[role]):
                if legal.id in moves_dict_ids[role]:
                    valid_probs[role].append(next(policy))
                    formatted_probs[role][i] = valid_probs[role][-1]

            total = sum(formatted_probs[role])
            assert 0.99 < total < 1.01, (total, role, formatted_probs[role], formatted_probs, list(policy))

        state = propnet.get_state(data)
        qs = {role: utils[i] for i, role in enumerate(propnet.roles)}
        states.append((state, formatted_probs, qs))



        depth = 0
        cfr_trainers = {}
        start2 = time.time()
        while time.time() - start2 < 0.1:
            depth += 1
            for role, node in imperfectNodes.items():
                if role == "random":
                    continue
                cfr_trainers[role] = CFRTrainerImperfect(node, depth, model)
                cfr_trainers[role].train(random.randint(num_iterations//2+1, num_iterations+1))
            # break
        valid_probs = {}
        moves_dict = propnet.legal_moves_dict(data)
        moves_dict_ids = {role: [x.id for x in moves_dict[role]] for role in moves_dict}
        for role in propnet.roles:
            if role == "random":
                continue
            policy = cfr_trainers[role].get_root_policy_for_player(role, num_iterations)
            print("second", role, policy)
            policy = iter(policy)
            valid_probs[role] = []
            for i, legal in enumerate(propnet.legal_for[role]):
                if legal.id in moves_dict_ids[role]:
                    valid_probs[role].append(next(policy))

            total = sum(valid_probs[role])
            assert 0.99 < total < 1.01, (total, role, len(list(policy)), sum(cfr_trainers[role].get_root_policy_for_player(role, num_iterations)), valid_probs)

        moves = []
        for player in propnet.roles:
            if player == "random":
                moves.append(random.choice(moves_dict_ids[player]))
            else:
                choice = random.random()
                for i, p in enumerate(valid_probs[player]):
                    if choice < p:
                        moves.append(moves_dict_ids[player][i])
                        break
                    else:
                        choice -= p

        propnet.do_sees_step(data, tuple(moves))
        visible = propnet.visible_dict(data)
        for i, role in enumerate(propnet.roles):
            imperfectNodes[role].add_history((moves[i], propnet.sees_ids_for(role, data)))
        propnet.do_non_sees_step(data, tuple(moves))
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
        model.save(game, i)
        with open(f'models/times-{game}', 'a') as f:
            f.write(f'{i} {time.time()-start}\n')
