from CFRTrainer_perfect import CFRTrainerPerfect
from perfectNode import PerfectNode
from propnet import load_propnet
from model import Model
import time
import sys
import random

game = sys.argv[1]

size = 5
if game == "blindtictactoe":
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
    print(printing_moves)
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


data, propnet = load_propnet(game)
model = Model(propnet)
# model.load_most_recent(game)
# model.save(game, 0)

def do_game(cur, propnet, model):
    global board
    board = [[' ' for i in range(size)] for j in range(size)]

    depth = 1
    # cfr_trainer = CFRTrainerPerfect(cur)
    num_iterations = 20
    # utils = cfr_trainer.train(num_iterations)
    # for i, player in enumerate(cfr_trainer.players):
    #     print(f"Computed average game value for player {player}: {(utils[i] / num_iterations):.3f}")
    #     policy = cfr_trainer.get_root_policy_for_player(player)
    #     print(f"policy = {policy}")

    states = []
    for step in range(1000):
        # model.print_eval(propnet.get_state(cur.data))
        start = time.time()
        # for i in range(N):
        #     simulation(cur)
        cfr_trainer = CFRTrainerPerfect(cur, depth, model)
        # utils = cfr_trainer.train(num_iterations)
        utils = cfr_trainer.train(random.randint(4, 10))
        # print('Counts were:')
        # for role, counts in probs.items():
        #     print('For', role)
        #     for id, count in counts.items():
        #         print(propnet.id_to_move[id].move_role, propnet.id_to_move[id].move_gdl, count)
        #     print('New expected return:', cur.q[role]/cur.count)
        # if any(sum(x.values()) < 10 for x in probs.values()):
            # import pdb; pdb.set_trace()
        formatted_probs = {}
        valid_probs = {}
        # print('Probs were:')
        moves_dict = propnet.legal_moves_dict(cur.data)
        moves_dict_ids = {role:[x.id for x in moves_dict[role]] for role in moves_dict}
        for role in propnet.roles:
            # print("here----", cfr_trainer.get_root_policy_for_player(role))
            policy = iter(cfr_trainer.get_root_policy_for_player(role))
            formatted_probs[role] = [0] * len(propnet.legal_for[role])
            valid_probs[role] = []
            # print([x.id for x in propnet.legal_for[role]], moves_dict_ids[role])
            for i, legal in enumerate(propnet.legal_for[role]):
                if legal.id in moves_dict_ids[role]:
                    valid_probs[role].append(next(policy))
                    formatted_probs[role][i] = valid_probs[role][-1]
            # if role != "random":
            #     print("policy for: ", role)
            #     print(formatted_probs[role])
            total = sum(formatted_probs[role])
            assert total > 0.99, (total, role, formatted_probs[role], formatted_probs)

            # if total == 0:
            #     total = 1
            # for i, prob in enumerate(formatted_probs[role]):
            #     # print(propnet.legal[i].move_gdl, prob/total)
            #     formatted_probs[role][i] = prob/total
        state = propnet.get_state(cur.data)
        # qs = {role: q/cur.count for role, q in cur.q.items()}
        qs = {role: utils[i] for i, role in enumerate(propnet.roles)}
        states.append((state, formatted_probs, qs))
        moves = []
        for player in propnet.roles:
            choice = random.random()
            # print(choice, end='   -------   ')
            for i, p in enumerate(valid_probs[player]):
                if choice < p:
                    # print(i, p)
                    moves.append(moves_dict_ids[player][i])
                    break
                else:
                    choice -= p

        print_moves(moves, propnet)

        cur = cur.get_or_make_child(tuple(moves))
        print('Play took %.4f seconds' % (time.time() - start))
        if cur.terminal:
            break
    scores = cur.scores
    for s, p, q in states:
        model.add_sample(s, p,q)


start = time.time()
for i in range(50000):
    cur = PerfectNode(propnet, data.copy())
    # cur[0] = B1Node(propnet, data, model=model)
    print('Game number', i)
    do_game(cur, propnet, model)
    # break
    model.train(epochs=10)
    if i and i % 50 == 0:
        model.save(game, i)
        with open(f'models/times-{game}', 'a') as f:
            f.write(f'{i} {time.time()-start}\n')
