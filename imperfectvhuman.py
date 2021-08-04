from imperfectNode_fast_and_approx import ImperfectNode
from perfectNode import PerfectNode
from CFRTrainer import CFRTrainer
import random
from propnet import load_propnet
# from model import Model
import time
import numpy as np
import sys

game = sys.argv[1]
my_role = sys.argv[2]

print("We're playing", game)
print('I am', my_role)

data, propnet = load_propnet(game)
# model = Model(propnet)
# model.load_most_recent(game)


myNode = ImperfectNode(propnet, data.copy(), my_role)
cur = PerfectNode(propnet, data.copy())
# sees_data = data.copy()
visible = propnet.visible_dict(data)
for step in range(1000):
    legal = propnet.legal_moves_dict(data)
    taken_moves = {}
    for role in propnet.roles:
        if role != "random":
            print(f"visible for {role}: ", [x.gdl for x in visible[role]])
    for role in propnet.roles:

        moves = legal[role]
        if len(moves) == 1:
            taken_moves[role] = moves[0]
        elif role == "random":
            taken_moves[role] = moves[step % len(moves)]
            # taken_moves[role] = random.choice(moves)
        elif role != my_role:
            print('Valid moves for', role)
            print(*(move.move_gdl for move in moves), sep='\n')
            m = input('Enter move: ')
            matching = [move for move in moves if m in move.move_gdl]
            while not matching:
                print('No moves containing %r' % m)
                m = input('Enter move: ')
                matching = [move for move in moves if m in move.move_gdl]
            print('Making move', matching[0].move_gdl)
            taken_moves[role] = matching[0]

    start = time.time()
    if len(legal[my_role]) == 1:
        taken_moves[my_role] = legal[my_role][0]
    else:
        if step > 50:
            cfr_trainer = CFRTrainer(myNode)
            num_iterations = 100
            utils = cfr_trainer.train(num_iterations)
            for i, player in enumerate(cfr_trainer.players):
                print(f"Computed average game value for player {player}: {(utils[i] / num_iterations):.3f}")

            policy = cfr_trainer.get_root_policy_for_player(my_role, num_iterations * 2)
            print(f"policy = {policy}")
            choice = random.random()
            for i, p in enumerate(policy):
                if choice < p:
                    print("chosen", i)
                    taken_moves[my_role] = legal[my_role][i]
                    break
                else:
                    choice -= p
        else:
            # taken_moves[my_role] = random.choice(legal[my_role])
            taken_moves[my_role] = legal[my_role][0]

        if step > -6:
            states = set()
            for i, state in enumerate(myNode.generate_posible_games()):
                states.add(int("".join(str(int(x)) for x in state), 2))
                if i > 1000:
                    break
            print(len(states))

        # print("should eval net now:")
        # model.print_eval(propnet.get_state(cur.data))
#         for i in range(N):
#             simulation(cur)
#         probs = cur.get_probs(1)
#         taken_moves[my_role] = None
#         best = 0
#         # print('Counts were:')
#         counts = probs[my_role]
#         for id, count in counts.items():
#             # print(propnet.id_to_move[id].move_role,
#             #       propnet.id_to_move[id].move_gdl, count)
#             if count > best:
#                 best = count
#                 taken_moves[my_role] = propnet.id_to_move[id]
    moves = [taken_moves[role].id for role in propnet.roles]

    data = data.copy()
    propnet.do_sees_step(data, tuple(moves))
    visible = propnet.visible_dict(data)
    myNode.add_history((taken_moves[my_role].id, propnet.sees_ids_for(my_role, data)))
    data = data.copy()
    propnet.do_non_sees_step(data, tuple(moves))
    print('Moves were:')
    for move in propnet.legal:
        if move.id in moves and move.move_gdl.strip() != 'noop':
            print(move.move_role, move.move_gdl)
    # print('Play took %.4f seconds' % (time.time() - start))
    if propnet.is_terminal(data):
        break

print("Terminal state reaced")
for role in propnet.roles:
    if role != "random":
        print(f"visible for {role}: ", [x.gdl for x in visible[role]])
for role, score in propnet.scores(data).items():
    print(role, 'got', score)
