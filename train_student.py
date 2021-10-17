from imperfectNode_fast_invalids import ImperfectNode
# from imperfectNode import ImperfectNode
from CFRTrainer_imperfect import CFRTrainerImperfect
from CFRTrainer_perfect import CFRTrainerPerfect
from perfectNode import PerfectNode
from propnet import load_propnet
import time
import sys
import random
import numpy as np
from print_conect4 import PrintConect4
import torch
import wandb

play_using_cfr = True
use_tf = False
if use_tf:
    from model import Model
else:
    from model_pytorch import Model


game = sys.argv[1]

num_iterations = 80
base_depth = 1
replay_size = 3200
load_model = False

game_printer = PrintConect4(game)

data, propnet = load_propnet(game)
model = Model(propnet, replay_size)
if load_model:
    model.load_most_recent(game)
model.save(game, 0)


run = wandb.init(
  project=game,
  entity="zestypuffin",
  config={
      "base_depth": base_depth,
      "replay_size": replay_size,
      "from_scratch": not load_model,
      "framework": "tensorflow" if use_tf else "pytorch",
      "playout mode": "cfr" if play_using_cfr else "raw network",
      "num_perfect_iterations": f"{num_iterations//2 + 1} -> {num_iterations*2}",
      "num_imperfect_iterations": f"{num_iterations//4} -> {num_iterations//2-1}",
  })


def tonum(data):
    return str(str(int("".join(str(int(x)) for x in data), 2)).__hash__())[:5]

def get_perfect_outputs_tf(propnet, cur, data, model, start):
    depth = base_depth
    while time.time() - start < 0.2:
        depth += 1
        cfr_trainer = CFRTrainerPerfect(cur, depth, model)
        utils = cfr_trainer.train(random.randint(num_iterations//2 + 1, num_iterations*2))
        # break

    print(model.eval(propnet.get_state(data)))
    formatted_probs = {}
    valid_probs = {}
    moves_dict = propnet.legal_moves_dict(data)
    moves_dict_ids = {role: [x.id for x in moves_dict[role]] for role in moves_dict}
    for role in propnet.roles:
        policy = cfr_trainer.get_root_policy_for_player(role)
        print("first", role, policy)
        policy = iter(policy)
        formatted_probs[role] = [0] * len(propnet.legal_for[role])
        valid_probs[role] = []
        for i, legal in enumerate(propnet.legal_for[role]):
            if legal.id in moves_dict_ids[role]:
                valid_probs[role].append(next(policy))
                formatted_probs[role][i] = valid_probs[role][-1]

        total = sum(formatted_probs[role])
        assert 0.99 < total < 1.01, (total, role, formatted_probs[role], formatted_probs, list(policy))

    return formatted_probs, {role: utils[i] for i, role in enumerate(propnet.roles)}

def get_perfect_outputs_pytorch(propnet, cur, data, model, start):
    depth = base_depth
    while time.time() - start < 0.2:
        depth += 1
        cfr_trainer = CFRTrainerPerfect(cur, depth, model)
        utils = cfr_trainer.train(random.randint(num_iterations//2 + 1, num_iterations*2))
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
    return probs, [utils[i] for i, role in enumerate(propnet.roles)]

def get_moves_cfr(propnet, imperfectNodes, data, model, step):
    moves_dict = propnet.legal_moves_dict(data)
    moves_dict_ids = {role: [x.id for x in moves_dict[role]] for role in moves_dict}

    def no_train(role):
        if role == "random":
            return True
        if len(moves_dict_ids[role]) < 2:
            return True
        if step > 10:
            return True
        return False

    depth = base_depth
    cfr_trainers = {}
    start = time.time()
    while time.time() - start < 0.1:
        depth += 1
        for role, node in imperfectNodes.items():
            if no_train(role):
                continue
            cfr_trainers[role] = CFRTrainerImperfect(node, depth, model, 0.5)
            cfr_trainers[role].train(random.randint(num_iterations//4, num_iterations//2-1))
        # break

    valid_probs = {}
    for role in propnet.roles:
        if no_train(role):
            continue
        policy = cfr_trainers[role].get_root_policy_for_player(role, num_iterations)
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


def get_moves_net(propnet, imperfectNodes, data, model):
    moves_dict = propnet.legal_moves_dict(data)
    moves_dict_ids = {role: [x.id for x in moves_dict[role]] for role in moves_dict}

    def no_train(role):
        if role == "random":
            return True
        if len(moves_dict_ids[role]) < 2:
            return True
        return False

    valid_probs = {}
    for role in propnet.roles:
        if no_train(role):
            continue
        node = imperfectNodes[role]
        legal_moves = list(propnet.legal_moves_for(role, data))
        policy = np.zeros(len(legal_moves))
        start = time.time()
        for i, data in enumerate(node.generate_posible_games()):
            probs_dict = model.eval(propnet.get_state(data))[0][role]
            probs_array = np.array([probs_dict[move.id] for move in legal_moves])
            policy += probs_array
            if i > num_iterations or time.time() - start > 0.5:
                break
        policy /= sum(policy)

        # print("second 2", role, [val**2 for val in policy])
        print("second", role, policy)
        policy = iter(policy)
        valid_probs[role] = []
        for i, legal in enumerate(propnet.legal_for[role]):
            if legal.id in moves_dict_ids[role]:
                valid_probs[role].append(next(policy))

        total = sum(valid_probs[role])
        valid_probs[role] = [val / total for val in valid_probs[role]]
        assert 0.99 < total < 1.01, (total, role, len(list(policy)))

    moves = []
    for player in propnet.roles:
        if no_train(player):
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
    imperfectNodes = {role: ImperfectNode(propnet, data.copy(), role) for role in propnet.roles}
    for step in range(1000):
        start = time.time()

        state = propnet.get_state(data)
        # print(state)
        # print(propnet.data2num(data))
        # print(hash(propnet.data2num(data)))
        # print(model.eval(propnet.get_state(data)))
        # print(model.eval(propnet.get_state(data)))
        # exit()
        if use_tf:
            probs, qs = get_perfect_outputs_tf(propnet, cur, data, model, start)
        else:
            probs, qs = get_perfect_outputs_pytorch(propnet, cur, data, model, start)
        states.append((state, probs, qs))

        start2 = time.time()
        if play_using_cfr:
            moves = get_moves_cfr(propnet, imperfectNodes, data, model, step)
        else:
            moves = get_moves_net(propnet, imperfectNodes, data, model)

        propnet.do_sees_step(data, tuple(moves))
        visible = propnet.visible_dict(data)
        for i, role in enumerate(propnet.roles):
            imperfectNodes[role].add_history((moves[i], propnet.sees_ids_for(role, data)))
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
    loss = model.train(epochs=20)
    if not use_tf:
        loss *= 35
    if loss > 0:
        root_out = model.eval(propnet.get_state(data))
        wandb.log({"loss": loss, "root policy": root_out[0], "root values": root_out[1]})
    do_game(cur, propnet, model)
    if i % 50 == 0:
        model.save(f"{game}", 200)
        # model.save(f"{game}_{play_using_cfr}_{use_tf}", 200)
        # model_artifact = wandb.Artifact(game, type="model")
        # model_artifact.add_file(f"./{model.save(game, i)}.")
        # run.log_artifact(model_artifact, aliases=['latest', game])
        with open(f'models/times-{game}', 'a') as f:
            f.write(f'{i} {time.time()-start}\n')
