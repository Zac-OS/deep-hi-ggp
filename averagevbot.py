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

# game = "transit"
game = "blindtictactoe"
my_role = sys.argv[1]

num_iterations = 80

game_printer = PrintConect4(game)

data_base, propnet = load_propnet(game)
model = Model(propnet)
# model.load_most_recent(game)
# model.load(f"models/{game}/step-000981.ckpt")
model.load(f"models/{game}/step-003643.ckpt")

lru_cache = LRU(2000)

def run_game(i):
    data = data_base.copy()
    myNode = ImperfectNode(propnet, data.copy(), my_role)
    # myNode = ImperfectNode(propnet, data.copy(), my_role, model, lru_cache)
    cur = PerfectNode(propnet, data.copy())
    visible = propnet.visible_dict(data)

    selected_corners = []
    got_center = False
    for step in range(1000):
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
            elif role == "random":
                if step == 0:
                    taken_moves[role] = moves[i % len(moves)]
                else:
                    taken_moves[role] = random.choice(moves)
                continue
            elif role == my_role:
                continue
            # for m in moves:
            #     print(m.gdl, m.id)

            # corners = [1394, 2118] # for transit

            corners = [423, 284, 474, 149, 573, 541, 438, 550]
            # corners = [35, 471, 392, 559, 503, 588, 224, 364]
            # corners = [198, 520, 488, 343, 401, 214, 365, 480]
            corners = [propnet.id_to_move[x] for x in corners]
            corners = [x for x in corners if x.gdl in [m.gdl for m in moves]]

            # if corners:
            #     taken_moves[role] = corners[0]
            # else:
            #     taken_moves[role] = random.choice(moves)
            # continue

            # print("here", [m.gdl for m in corners])

            if step == 0:
                taken_moves[role] = moves[4]
            elif step == 1:
                if visible[role]:
                    got_center = True
                last_index = random.choice([0, 2, 5, 7])
                taken_moves[role] = moves[last_index]
            elif step == 2:
                if visible[role] and got_center:
                    if last_index == 0:
                        last_index = 6
                        selected_corners.append(1)
                    if last_index == 2:
                        last_index = 4
                        selected_corners.append(3)
                    if last_index == 5:
                        last_index = 2
                        selected_corners.append(7)
                    if last_index == 7:
                        last_index = 0
                        selected_corners.append(9)
                else:
                    if last_index == 0:
                        last_index = 1
                        if visible[role]:
                            selected_corners.append(1)
                    if last_index == 2:
                        last_index = 0
                        if visible[role]:
                            selected_corners.append(3)
                    if last_index == 5:
                        last_index = 0
                        if visible[role]:
                            selected_corners.append(7)
                    if last_index == 7:
                        last_index = 2
                        if visible[role]:
                            selected_corners.append(9)
                taken_moves[role] = moves[last_index]
            elif step == 3 and visible[role]:
                if last_index == 1 or last_index == 2:
                    selected_corners.append(3)
                elif last_index == 0:
                    selected_corners.append(1)
                if 1 in selected_corners and 3 in selected_corners:
                    taken_moves[role] = moves[0]
                elif 1 in selected_corners and 7 in selected_corners:
                    taken_moves[role] = moves[2]
                elif 3 in selected_corners and 9 in selected_corners:
                    taken_moves[role] = moves[3]
                elif 1 in selected_corners:
                    taken_moves[role] = moves[5]
                elif 3 in selected_corners:
                    taken_moves[role] = moves[4]
            elif corners:
                taken_moves[role] = random.choice(corners)
            else:
                taken_moves[role] = random.choice(moves)

        if my_role not in taken_moves:
            start = time.time()
            depth = 0
            while time.time() - start < 1:
                myNode.data = myNode.data.copy()
                depth += 1
                print("depth: ", depth)
                # cfr_trainer = CFRTrainerImperfect(myNode)
                cfr_trainer = CFRTrainerImperfect(myNode, depth, model)
                utils = cfr_trainer.train(num_iterations)
                # break

            for i, player in enumerate(cfr_trainer.players):
                print(f"Computed average game value for player {player}: {utils[i] :.3f}")

            policy = cfr_trainer.get_root_policy_for_player(my_role, num_iterations, False)
            assert 0.99 < sum(policy) < 1.01, (sum(policy), my_role, policy)

            print(f"policy = {policy}")
            # model.print_eval(propnet.get_state(cur.data))

            # index, prob = max(enumerate(policy), key= lambda x: x[1])
            # print("prob:", prob)
            # taken_moves[my_role] = legal[my_role][index]
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
