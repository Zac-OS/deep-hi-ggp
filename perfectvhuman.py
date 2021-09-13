from perfectNode import PerfectNode
from CFRTrainer_perfect import CFRTrainerPerfect
from print_conect4 import PrintConect4
import random
from propnet import load_propnet
from model import Model
import time
import numpy as np
import sys

game = sys.argv[1]
my_role = sys.argv[2]

print("We're playing", game)
print('I am', my_role)

game_printer = PrintConect4(game)

data, propnet = load_propnet(game)
model = Model(propnet)
model.load_most_recent(game)

num_iterations = 20

cur = PerfectNode(propnet, data.copy())
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
            # taken_moves[role] = moves[step % len(moves)]
            taken_moves[role] = random.choice(moves)
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
    cur.data = data.copy()
    cfr_trainer = CFRTrainerPerfect(cur)
    # cfr_trainer = CFRTrainerPerfect(cur, 1, model)
    utils = cfr_trainer.train(num_iterations)
    for i, player in enumerate(cfr_trainer.players):
        print(f"Computed average game value for player {player}: {utils[i] :.3f}")

    policy = cfr_trainer.get_root_policy_for_player(my_role)
    print(f"policy = {policy}")
    model.print_eval(propnet.get_state(cur.data))
    choice = random.random()
    for i, p in enumerate(policy):
        if choice < p:
            print("chosen", i)
            taken_moves[my_role] = legal[my_role][i]
            break
        else:
            choice -= p

    moves = [taken_moves[role].id for role in propnet.roles]

    data = data.copy()
    propnet.do_sees_step(data, tuple(moves))
    visible = propnet.visible_dict(data)
    data = data.copy()
    propnet.do_non_sees_step(data, tuple(moves))
    print('Moves were:')
    for move in propnet.legal:
        if move.id in moves and move.move_gdl.strip() != 'noop':
            print(move.move_role, move.move_gdl)
    game_printer.make_moves(moves, propnet)
    game_printer.print_moves()
    # print('Play took %.4f seconds' % (time.time() - start))
    if propnet.is_terminal(data):
        break

print("Terminal state reaced")
for role in propnet.roles:
    if role != "random":
        print(f"visible for {role}: ", [x.gdl for x in visible[role]])
for role, score in propnet.scores(data).items():
    print(role, 'got', score)
