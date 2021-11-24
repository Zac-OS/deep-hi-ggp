# from imperfectNode_fast_invalids import ImperfectNode
# from imperfectNode_fast_valids import ImperfectNode
from imperfectNode_Model import ImperfectNode as ImperfectNodeModel
# from imperfectNode import ImperfectNode
from CFRTrainer_imperfect import CFRTrainerImperfect
from CFRTrainer_perfect import CFRTrainerPerfect
from perfectNode import PerfectNode
from propnet import load_propnet
import time
import sys
import random
from print_conect4 import PrintConect4
import torch
import wandb
from model_pytorch import Model
from lru import LRU

game = sys.argv[1]
num_iterations = 100
base_depth = 0
replay_size = 2500
load_model = False
log_data = False

game_printer = PrintConect4(game)

data, propnet = load_propnet(game)
model = Model(propnet, replay_size)
if load_model:
    # model.load_most_recent(game)
    model.load(f"models/{game}/step-000200.ckpt")
    # model.load(f"models/{game}/step-000201.ckpt")

num_perfect_iterations = [num_iterations//2 + 1, num_iterations*2]
num_imperfect_iterations = [num_iterations//2, num_iterations]
if log_data:
    run = wandb.init(
      project=game,
      entity="zestypuffin",
      config={
          "base_depth": base_depth,
          "replay_size": replay_size,
          "from_scratch": not load_model,
          "num_perfect_iterations": f"{num_perfect_iterations[0]} -> {num_perfect_iterations[1]}",
          "num_imperfect_iterations": f"{num_imperfect_iterations[0]} -> {num_imperfect_iterations[1]}",
      })


def tonum(data):
    return str(str(int("".join(str(int(x)) for x in data), 2)).__hash__())[:5]


def get_perfect_outputs_pytorch(propnet, cur, data, model, start):
    depth = base_depth
    while time.time() - start < 0.7:
        depth += 1
        print("depth", depth)
        cfr_trainer = CFRTrainerPerfect(cur, depth, model, 8)
        utils = cfr_trainer.train(random.randint(*num_perfect_iterations))
        if depth > 6:
            break
        # break

    print(model.eval(propnet.get_state(data)))
    moves_dict = propnet.legal_moves_dict(data)
    moves_dict_ids = {role: [x.id for x in moves_dict[role]] for role in moves_dict}
    probs = torch.zeros((len(propnet.roles), max(len(propnet.legal_for[role]) for role in propnet.roles)))
    for i, role in enumerate(propnet.roles):
        policy = cfr_trainer.get_root_policy_for_player(role)
        print("first", role, policy)
        policy = iter(policy)
        for j, legal in enumerate(propnet.legal_for[role]):
            if legal.id in moves_dict_ids[role]:
                probs[i][j] = next(policy)

        total = sum(probs[i])
        assert 0.99 < total < 1.01, (total, role, probs[i])
    return probs, [utils[i] for i, _ in enumerate(propnet.roles)]

def get_moves_cfr(propnet, imperfectNodes, data, model, step):
    moves_dict = propnet.legal_moves_dict(data)
    moves_dict_ids = {role: [x.id for x in moves_dict[role]] for role in moves_dict}

    def no_train(role):
        if role == "random":
            return True
        if len(moves_dict_ids[role]) < 2:
            return True
        # if step > 10:
        #     return True
        return False

    depth = base_depth
    cfr_trainers = {}
    start = time.time()
    while time.time() - start < 0.3:
        depth += 1
        print("Depth: ", depth)
        for role, node in imperfectNodes.items():
            if no_train(role):
                continue
            cfr_trainers[role] = CFRTrainerImperfect(node, depth, model, 1)
            cfr_trainers[role].train(random.randint(*num_imperfect_iterations))
        if depth > 6:
            break

    valid_probs = {}
    for role in propnet.roles:
        if no_train(role):
            continue
        policy = cfr_trainers[role].get_root_policy_for_player(role, num_iterations, False)
        print("second", role, policy)
        policy = iter(policy)
        valid_probs[role] = []
        for i, legal in enumerate(propnet.legal_for[role]):
            if legal.id in moves_dict_ids[role]:
                valid_probs[role].append(next(policy))

        total = sum(valid_probs[role])
        # assert 0.99 < total < 1.01, (total, role, len(list(policy)), sum(cfr_trainers[role].get_root_policy_for_player(role, num_iterations)), valid_probs)
        if not 0.99 < total < 1.01:
            valid_probs[role] = None

    moves = []
    for player in propnet.roles:
        if no_train(player) or valid_probs[player] is None:
            print(f"random move for player {player}")
            moves.append(random.choice(moves_dict_ids[player]))
        else:
            choice = random.random()
            for i, p in enumerate(valid_probs[player]):
                if choice < p:
                    print(i, p, choice)
                    moves.append(moves_dict_ids[player][i])
                    break
                else:
                    choice -= p
    return moves

# @profile
def do_game(cur, propnet, model):
    game_printer.reset()

    data = cur.data
    states = []
    # imperfectNodes = {role: ImperfectNode(propnet, data.copy(), role) for role in propnet.roles}
    # cache = LRU(2000)
    imperfectNodes_model = {role: ImperfectNodeModel(propnet, data.copy(), role, model, LRU(2000)) for role in propnet.roles}
    for step in range(1000):
        start = time.time()

        state = propnet.get_state(data)
        probs, qs = get_perfect_outputs_pytorch(propnet, cur, data, model, start)
        states.append((state, probs, qs))

        start2 = time.time()
        moves = get_moves_cfr(propnet, imperfectNodes_model, data, model, step)


        propnet.do_sees_step(data, tuple(moves))
        for i, role in enumerate(propnet.roles):
            # imperfectNodes[role].add_history((moves[i], propnet.sees_ids_for(role, data)))
            imperfectNodes_model[role].add_history((moves[i], propnet.sees_ids_for(role, data)))
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
    loss = model.train(epochs=20)
    if loss > 0:
        root_out = model.eval(propnet.get_state(data))
        if log_data:
            wandb.log({"loss": loss*35, "root policy": root_out[0], "root values": root_out[1]})

    if i % 50 == 0:
        model.save(f"{game}", i//50)
        with open(f'models/times-{game}', 'a') as f:
            f.write(f'{i} {time.time()-start}\n')
