from imperfectNode_fast_and_approx import ImperfectNode
from CFRTrainer import CFRTrainer
from perfectNode import PerfectNode
from propnet import load_propnet
from model import Model
import time
import sys

game = sys.argv[1]

data, propnet = load_propnet(game)
model = Model(propnet)
model.save(game, 0)

start = time.time()
for i in range(50000):
    playerNodes = {role: ImperfectNode(propnet, data, role) for role in propnet.players if role != "random"}
    cur = PerfectNode(propnet, data)
    # cur[0] = B1Node(propnet, data, model=model)
    print('Game number', i)
    do_game(cur, propnet, model)
    model.train(epochs=10)
    if i and i % 50 == 0:
        model.save(game, i)
        with open(f'models/times-{game}', 'a') as f:
            f.write(f'{i} {time.time()-start}\n')
