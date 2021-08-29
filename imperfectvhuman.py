from imperfectNode_fast_invalids import ImperfectNode
from perfectNode import PerfectNode
# from CFRTrainer import CFRTrainer
from CFRTrainer_imperfect import CFRTrainerImperfect
from CFRTrainer_perfect import CFRTrainerPerfect
import random
from propnet import load_propnet
from model import Model
import time
import numpy as np
import sys

size = 3
board = [[' ' for i in range(size)] for j in range(size)]
def print_board(board):
    for line in board:
        print("|".join(line))

def print_moves(moves, propnet):
    printing_moves = {}
    for role in propnet.roles:
        for move in propnet.legal_for[role]:
            if move.id in moves:
                printing_moves[role] = move.gdl
    # print(printing_moves)
    if "oplayer" in printing_moves:
        printing_moves["o"] = printing_moves["oplayer"].replace("oplayer", "o")
        printing_moves["x"] = printing_moves["xplayer"].replace("xplayer", "x")
        # printing_moves["x"]
        # printing_moves["o"]
    if "o" in printing_moves:
        # for i, ch in enumerate(printing_moves["o"]):
        #     print(i, ch)
        print(printing_moves["o"])
        x0, x1 = int(printing_moves["x"][17])-1, int(printing_moves["x"][19])-1
        o0, o1 = int(printing_moves["o"][17])-1, int(printing_moves["o"][19])-1
        if (x0, x1) == (o0, o1):
            if "random" in printing_moves:
                if "x" in printing_moves["random"]:
                    board[x0][x1] = "x"
                else:
                    board[o0][o1] = "o"
        else:
            if board[x0][x1] == " ":
                board[x0][x1] = "x"
            if board[o0][o1] == " ":
                board[o0][o1] = "o"
    print_board(board)



game = sys.argv[1]
my_role = sys.argv[2]

print("We're playing", game)
print('I am', my_role)

data, propnet = load_propnet(game)
model = Model(propnet)
model.load_most_recent(game)


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
        # if step > 5:
        #     cfr_trainer = CFRTrainerImperfect(myNode)
        # else:
        cur.data = data.copy()
        cfr_trainer = CFRTrainerPerfect(cur, 1, model)
        num_iterations = 10
        utils = cfr_trainer.train(num_iterations)
        for i, player in enumerate(cfr_trainer.players):
            print(f"Computed average game value for player {player}: {utils[i] :.3f}")

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
        # else:
        #     # taken_moves[my_role] = random.choice(legal[my_role])
        #     taken_moves[my_role] = legal[my_role][0]

        if step > -6:
            states = set()
            for i, state in enumerate(myNode.generate_posible_games()):
                states.add(int("".join(str(int(x)) for x in state), 2))
                if i > 100:
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
    print_moves(moves, propnet)
    # print('Play took %.4f seconds' % (time.time() - start))
    if propnet.is_terminal(data):
        break

print("Terminal state reaced")
for role in propnet.roles:
    if role != "random":
        print(f"visible for {role}: ", [x.gdl for x in visible[role]])
for role, score in propnet.scores(data).items():
    print(role, 'got', score)
