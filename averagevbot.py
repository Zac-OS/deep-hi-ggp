from imperfectNode_fast_invalids import ImperfectNode
# from imperfectNode import ImperfectNode
from perfectNode import PerfectNode
from CFRTrainer_imperfect import CFRTrainerImperfect
from print_conect4 import PrintConect4
import random
from propnet import load_propnet
from model import Model
import time
import numpy as np
import sys

game = "blindtictactoe"
my_role = "o"

num_iterations = 40

game_printer = PrintConect4(game)

data_base, propnet = load_propnet(game)
model = Model(propnet)
model.load_most_recent(game)

def run_game(i):
    data = data_base.copy()
    myNode = ImperfectNode(propnet, data.copy(), my_role)
    cur = PerfectNode(propnet, data.copy())
    visible = propnet.visible_dict(data)
    for step in range(1000):
        legal = propnet.legal_moves_dict(data)
        taken_moves = {}
        for role in propnet.roles:
            if role != "random" and role != "o":
                print(f"visible for {role}: ", [x.gdl for x in visible[role]])

        for role in propnet.roles:
            moves = legal[role]
            corners = [423, 284, 474, 149, 573, 541, 438, 550]
            corners = [propnet.id_to_move[x] for x in corners]
            corners = [x for x in corners if x.gdl in [m.gdl for m in moves]]
            # print("here", [m.gdl for m in corners])
            if len(moves) == 1:
                taken_moves[role] = moves[0]
            elif role == "random":
                if step == 0:
                    taken_moves[role] = moves[i % 2]
                else:
                    taken_moves[role] = random.choice(moves)
            elif role != my_role:
                if step == 0:
                    taken_moves[role] = moves[4]
                elif step == 1:
                    last_index = random.choice([0, 2, 5, 7])
                    taken_moves[role] = moves[last_index]
                elif step == 2:
                    if visible[role]:
                        if last_index == 0:
                            last_index = 6
                        if last_index == 2:
                            last_index = 4
                        if last_index == 5:
                            last_index = 2
                        if last_index == 7:
                            last_index = 0
                    else:
                        if last_index == 0:
                            last_index = 1
                        if last_index == 2:
                            last_index = 0
                        if last_index == 5:
                            last_index = 0
                        if last_index == 7:
                            last_index = 2
                    taken_moves[role] = moves[last_index]
                elif corners:
                    taken_moves[role] = random.choice(corners)
                else:
                    taken_moves[role] = random.choice(moves)

        start = time.time()
        depth = 0
        while time.time() - start < 1:
            myNode.data = myNode.data.copy()
            depth += 1
            print("depth: ", depth)
            # cfr_trainer = CFRTrainerImperfect(myNode)
            cfr_trainer = CFRTrainerImperfect(myNode, depth, model)
            utils = cfr_trainer.train(num_iterations)
            break

        for i, player in enumerate(cfr_trainer.players):
            print(f"Computed average game value for player {player}: {utils[i] :.3f}")

        policy = cfr_trainer.get_root_policy_for_player(my_role, num_iterations)
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


num_games = 30
my_score = 0
other_scores = [0] * len(propnet.roles)
for i in range(num_games):
    scores = run_game(i)
    my_score += scores[my_role] / num_games
    for i, role in enumerate(scores):
        if role != my_role:
            other_scores[i] += scores[role] / num_games
    print(my_score, other_scores)
