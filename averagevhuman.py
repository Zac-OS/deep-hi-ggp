# from imperfectNode_fast_valids import ImperfectNode
# from imperfectNode_Model import ImperfectNode
# from imperfectNode_fast_invalids import ImperfectNode
from imperfectNode import ImperfectNode
from perfectNode import PerfectNode
from CFRTrainer_imperfect import CFRTrainerImperfect
from print_conect4 import PrintConect4
import random
from propnet import load_propnet
import time
import numpy as np
import sys
from lru import LRU


use_tf = False
from model_pytorch import Model


game = sys.argv[1]
my_role = sys.argv[2]

print("We're playing", game)
print('I am', my_role)

num_iterations = 80

game_printer = PrintConect4(game)

data, propnet = load_propnet(game)
model = Model(propnet)
# model.load_most_recent(game)
model.load(f"models/{game}/step-000109.ckpt")
state = propnet.get_state(data)
# print(state)
# print(propnet.data2num(data))
# print(hash(propnet.data2num(data)))
# print(model.eval(propnet.get_state(data)))
# print(model.eval(propnet.get_state(data)))
# exit()

# myNode = ImperfectNode(propnet, data.copy(), my_role, model, LRU(2000))
myNode = ImperfectNode(propnet, data.copy(), my_role)
cur = PerfectNode(propnet, data.copy())
visible = propnet.visible_dict(data)
for step in range(1000):
    legal = propnet.legal_moves_dict(data)
    taken_moves = {}
    for role in propnet.roles:
        if role != "random":
            print(f"visible for {role}: ", [x.gdl for x in visible[role]])

    # start = time.time()
    # states = set()
    # for i, state in enumerate(myNode.generate_posible_games()):
    #     states.add(int("".join(str(int(x)) for x in state), 2))
    #     if i > 100:
    #         break
    # print(f"num states: {len(states)}")
    # print(time.time() - start)

    for role in propnet.roles:
        moves = legal[role]
        if len(moves) == 1:
            taken_moves[role] = moves[0]
        elif role == "random":
            # taken_moves[role] = moves[step % len(moves)]
            taken_moves[role] = random.choice(moves)
            # taken_moves[role] = moves[1]
        elif role != my_role:
            print(f'Valid moves for {role}: {[move.move_gdl for move in moves]}')
            m = input('Enter move: ')
            matching = [move for move in moves if m in move.move_gdl]
            while not matching:
                print('No moves containing %r' % m)
                m = input('Enter move: ')
                matching = [move for move in moves if m in move.move_gdl]
            print('Making move', matching[0].move_gdl)
            taken_moves[role] = matching[0]

    if my_role not in taken_moves:
        start = time.time()
        depth = 0
        while time.time() - start < 1:
            myNode.data = myNode.data.copy()
            depth += 1
            print("depth: ", depth)
            # cfr_trainer = CFRTrainerImperfect(myNode)
            cfr_trainer = CFRTrainerImperfect(myNode, depth, model, 2)
            utils = cfr_trainer.train(num_iterations)
            # break
            if depth > 5:
                break
        for i, player in enumerate(cfr_trainer.players):
            print(f"Computed average game value for player {player}: {utils[i] :.3f}")

        policy = cfr_trainer.get_root_policy_for_player(my_role, num_iterations*2, True)
        # policy = cfr_trainer.get_root_policy_for_player(my_role, num_iterations*2, False)
        assert 0.99 < sum(policy) < 1.01, (sum(policy), my_role, policy)

        print(f"policy = {policy}")
        # model.print_eval(propnet.get_state(data))
        out = model.eval(propnet.get_state(data))
        print(out[1])
        out = out[0]
        for role in out:
            if role != "random":
                print(role)
                print(out[role])
        choice = random.random()
        print(choice, end=" ")
        for i, p in enumerate(policy):
            if choice < p:
                print(p)
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
        break
# game_printer.print_moves()

print("Terminal state reaced")
for role in propnet.roles:
    if role != "random":
        print(f"visible for {role}: ", [x.gdl for x in visible[role]])
for role, score in propnet.scores(data).items():
    print(role, 'got', score)
