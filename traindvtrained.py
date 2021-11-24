# from imperfectNode_fast_valids import ImperfectNode
from imperfectNode import ImperfectNode
# from imperfectNode_Model import ImperfectNode
from perfectNode import PerfectNode
from CFRTrainer_imperfect import CFRTrainerImperfect
from print_conect4 import PrintConect4
import random
from propnet import load_propnet
# from model import Model
from model_pytorch import Model
import time
import numpy as np
import sys
from lru import LRU

game = "blindtictactoeXbias"

num_iterations = 90

game_printer = PrintConect4(game)

# [0,  1,   2,   3,   4,   5,   6,   7,   8,   9,  11,  12,  14,  16,  18,   22,   23,   24,   25,   28,   30,   32,   34,   37,   40,   45,   50, 58]
# [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 550, 600, 700, 800, 900, 1100, 1150, 1200, 1250, 1400, 1500, 1600, 1700, 1850, 2000, 2250, 2500, 2900]
# [32,25,  27,  39,  41,  38,  47,  45,  46,  41,  47,  48,  49,  48,  60,   52,   38,   64,   45,   49,   56,   56,   47,   55,   56,   46,   54, 48]

# (0,32),(50,25),(100,27),(200,41),(300,47),(450,41),(600,48),(1200,64),(2900,48)

[3, 5, 7, 8, 11, 14, 16, 18, 22, 28, 34, 40, 50]

data_base, propnet = load_propnet(game)
modelx = Model(propnet)
modelx.load(f"models/{game}/step-000148.ckpt")
modelo = Model(propnet)
modelo.load(f"models/{game}/step-000148.ckpt")

state_cache = {"x":{}, "o":{}}

def get_policy(data, myNode, model, my_role, depth=0):
    data_num = propnet.data2num(data)
    if data_num in state_cache[my_role]:
        return state_cache[my_role][data_num]
    start = time.time()
    while time.time() - start < 0.5:
        myNode.data = myNode.data.copy()
        depth += 1
        print("depth: ", depth)
        # cfr_trainer = CFRTrainerImperfect(myNode)
        cfr_trainer = CFRTrainerImperfect(myNode, depth, model)
        utils = cfr_trainer.train(num_iterations)
        break

    for i, player in enumerate(cfr_trainer.players):
        print(f"Computed average game value for player {player}: {utils[i] :.3f}")

    policy = cfr_trainer.get_root_policy_for_player(my_role, num_iterations, False)
    print(policy)
    # if depth == 2:
    state_cache[my_role][data_num] = policy
    assert 0.99 < sum(policy) < 1.01, (sum(policy), policy)
    return policy

def run_game(i):
    data = data_base.copy()
    myNodex = ImperfectNode(propnet, data.copy(), "x")
    myNodeo = ImperfectNode(propnet, data.copy(), "o")
    # myNodex = ImperfectNode(propnet, data.copy(), "x", modelx, LRU(2000))
    # myNodeo = ImperfectNode(propnet, data.copy(), "o", modelo, LRU(2000))
    visible = propnet.visible_dict(data)

    for step in range(1000):
        legal = propnet.legal_moves_dict(data)
        taken_moves = {}
        for role in propnet.roles:
            print(f"visible for {role}: ", [x.gdl for x in visible[role]])

        for role in propnet.roles:
            moves = legal[role]
            if len(moves) == 1:
                taken_moves[role] = moves[0]
            elif role == "random":
                if step == 0:
                    taken_moves[role] = moves[i % len(moves)]
                else:
                    taken_moves[role] = random.choice(moves)

        if "x" not in taken_moves:
            policy = get_policy(data, myNodex, modelx, "x", 1 if step == 0 else 0)

            choice = random.random()
            for i, p in enumerate(policy):
                if choice < p:
                    taken_moves["x"] = legal["x"][i]
                    break
                else:
                    choice -= p
        if "o" not in taken_moves:
            policy = get_policy(data, myNodeo, modelo, "o", 1 if step == 0 else 0)
            choice = random.random()
            for i, p in enumerate(policy):
                if choice < p:
                    taken_moves["o"] = legal["o"][i]
                    break
                else:
                    choice -= p

        moves = [taken_moves[role].id for role in propnet.roles]

        data = data.copy()
        propnet.do_sees_step(data, tuple(moves))
        visible = propnet.visible_dict(data)
        myNodex.add_history((taken_moves["x"].id, propnet.sees_ids_for("x", data)))
        myNodeo.add_history((taken_moves["o"].id, propnet.sees_ids_for("o", data)))
        propnet.do_non_sees_step(data, tuple(moves))
        game_printer.make_moves(moves, propnet)
        game_printer.print_moves()
        if propnet.is_terminal(data):
            game_printer.reset()
            break

    print("Terminal state reaced")
    for role in propnet.roles:
        if role != "random":
            print(f"visible for {role}: ", [x.gdl for x in visible[role]])
    for role, score in propnet.scores(data).items():
        print(role, 'got', score)
    return {player: propnet.scores(data)[player] if player != "random" else 0 for player in propnet.roles}


num_games = 100
scorex = 0
scoreo = 0
other_scores = [0] * len(propnet.roles)
for i in range(num_games):
    scores = run_game(i)
    scorex += scores["x"] / num_games
    scoreo += scores["o"] / num_games
    print(f"x: {scorex}, o: {scoreo}")
