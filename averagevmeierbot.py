# from imperfectNode_fast_valids import ImperfectNode
# from imperfectNode import ImperfectNode
from imperfectNode_Model import ImperfectNode
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

game = "meier"
my_role = sys.argv[1]

num_iterations = 20

game_printer = PrintConect4(game)

data_base, propnet = load_propnet(game)
model = Model(propnet)
# model.load_most_recent(game)
model.load(f"models/{game}/step-000202.ckpt")

lru_cache = LRU(2000)

def run_game(i):
    data = data_base.copy()
    # myNode = ImperfectNode(propnet, data.copy(), my_role)
    myNode = ImperfectNode(propnet, data.copy(), my_role, model, lru_cache)
    cur = PerfectNode(propnet, data.copy())
    visible = propnet.visible_dict(data)

    selected_corners = []
    got_center = False
    for step in range(1000):
        print(f"Step: {step}")
        legal = propnet.legal_moves_dict(data)
        taken_moves = {}
        for role in propnet.roles:
            if role == my_role:
                print(f"visible for {role}: ", [x.gdl for x in visible[role]])

        for role in propnet.roles:
            moves = legal[role]
            if len(moves) == 1:
                taken_moves[role] = moves[0]
                continue
            elif role == "random" or role == my_role:
                if step == 0:
                    taken_moves[role] = moves[i % 36]
                else:
                    taken_moves[role] = random.choice(moves)
                continue
            elif role == my_role:
                continue
            dice = visible[role][-1].gdl[20:23]

            options = ["claim " + dice, "claim " + dice[::-1]] # for transit

            # actions = [x for x in moves if x.gdl in actions]
            actions = []
            for m in moves:
                for o in options:
                    if o in m.gdl:
                        # print("-----", m.gdl)
                        actions.append(m)

            if actions:
                taken_moves[role] = random.choice(actions)
                # print("making action")
            else:
                # print("random")
                taken_moves[role] = random.choice(moves)

        if my_role not in taken_moves:
            start = time.time()
            depth = 1
            while time.time() - start < 1:
                myNode.data = myNode.data.copy()
                depth += 1
                # print("depth: ", depth)
                # cfr_trainer = CFRTrainerImperfect(myNode)
                cfr_trainer = CFRTrainerImperfect(myNode, depth, model)
                utils = cfr_trainer.train(num_iterations)
                break

            for i, player in enumerate(cfr_trainer.players):
                print(f"Computed average game value for player {player}: {utils[i] :.3f}")

            policy = cfr_trainer.get_root_policy_for_player(my_role, num_iterations, False)
            assert 0.99 < sum(policy) < 1.01, (sum(policy), my_role, policy)

            print(f"policy = {policy}")
            # model.print_eval(propnet.get_state(cur.data))
            choice = random.random()
            for i, p in enumerate(policy):
                if choice < p:
                    taken_moves[my_role] = legal[my_role][i]
                    break
                else:
                    choice -= p

        moves = [taken_moves[role].id for role in propnet.roles]

        data = data.copy()
        propnet.do_sees_step(data, tuple(moves))
        visible = propnet.visible_dict(data)
        myNode.add_history((taken_moves[my_role].id, propnet.sees_ids_for(my_role, data)))
        data = data.copy()
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
my_score = 0
other_scores = [0] * len(propnet.roles)
for i in range(num_games):
    scores = run_game(i)
    my_score += scores[my_role] / num_games
    for i, role in enumerate(scores):
        if role != my_role:
            other_scores[i] += scores[role] / num_games
    print(my_score, other_scores)
