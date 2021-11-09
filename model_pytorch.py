import torch
from torch import nn
import collections
import random
import pathlib
import time
import os
import numpy as np

NUM_PRE_LAYERS = 2
MIN_PRE_SIZE = 50
NUM_POST_LAYERS = 2
MIN_POST_SIZE = 100
REPLAY_SIZE = 20000
REPLAY_SIZE = 3200


class Network(nn.Module):
    def __init__(self, num_inputs, num_actions, roles, head_depth=3):
        super().__init__()
        self.dropout = nn.Dropout(p=0.2)
        self.activ = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
        self.head = nn.Sequential(*[
            nn.Sequential(nn.Linear(num_inputs, num_inputs), self.activ, self.dropout)
        for _ in range(head_depth)])
        self.val = nn.Linear(num_inputs, len(roles))
        self.policy_head = nn.Sequential(nn.Linear(num_inputs, num_inputs), self.activ, self.dropout)
        self.policy = nn.Sequential(*[nn.Linear(num_inputs, num_actions[role]) for role in roles])
        self.max_actions = max(num_actions[role] for role in roles)

    def forward(self, x):
        x = self.head(x)

        val = self.sigmoid(self.val(x))
        x = self.policy_head(x)
        if len(x.shape) > 1:
            policies = torch.zeros((x.shape[0], val.shape[-1], self.max_actions))
            for i, policy_layer in enumerate(self.policy):
                out = self.softmax(policy_layer(x))
                policies[:,i, :out.shape[-1]] = out
        else:
            policies = [self.softmax(policy_layer(x)) for policy_layer in self.policy]

        return policies, val

class Model:
    def __init__(self, propnet, replay_size=REPLAY_SIZE):
        self.roles = propnet.roles
        self.legal_for = propnet.legal_for
        self.id_to_move = propnet.id_to_move
        self.num_actions = {role: len(actions)
                            for role, actions in propnet.legal_for.items()}
        self.num_inputs = len(propnet.propositions)
        print(f"num_inputs: {self.num_inputs}")
        self.replay_buffer = collections.deque(maxlen=replay_size)
        self.net = Network(self.num_inputs, self.num_actions, self.roles)
        self.optimiser = torch.optim.Adam(self.net.parameters())
        self.loss_func = nn.MSELoss()
        self.cache = {}


    def add_sample(self, state, probs, scores):
        self.replay_buffer.append((state, probs, scores))

    # @profile
    def eval(self, state):
        if type(state) is not tuple:
            state = tuple(state)
        if state in self.cache:
            return self.cache[state]
        all_qs = {}
        all_probs = {}
        self.net.eval()
        with torch.no_grad():
            output = self.net(torch.tensor(state, dtype=torch.float32))
            for i, role in enumerate(self.roles):
                all_qs[role] = output[1][i].item()
                all_probs[role] = {}
                for prob, inp in zip(output[0][i], self.legal_for[role]):
                    all_probs[role][inp.id] = prob.item()
        # print(all_probs)
        self.cache[state] = (all_probs, all_qs)
        return all_probs, all_qs

    def print_eval(self, state):
        probs, qs = self.eval(state)
        for role in self.roles:
            if role == "random":
                continue
            print('Role', role, 'expected return:', qs[role])
            for i, pr in probs[role].items():
                print(self.id_to_move[i].move_gdl, '%.3f' % pr)


    def train(self, epochs=5, batchsize=128):
        # Sample from replay buffer and train
        if batchsize//2 > len(self.replay_buffer):
            print('Skipping as replay buffer too small')
            return 0
        elif batchsize > len(self.replay_buffer):
            batchsize //= 2

        self.net.train()
        self.cache = {}
        sum_loss = 0
        for i in range(epochs):
            sample = random.sample(self.replay_buffer, batchsize)
            # print(sample)
            # sample = np.array(sample)
            inputs = torch.tensor([x[0] for x in sample]).float()
            # print(inputs)
            # print(inputs.shape)
            probs = torch.stack([x[1] for x in sample]).float()
            # print(probs.shape)
            # probs = torch.zeros((batchsize, len(self.roles), max(len(actions) for actions in sample[0][1])))
            # print(probs.shape)
            # for i, x in enumerate(sample):
            #     for j, role in enumerate(self.roles):
            #         for k, val in enumerate(x[1][j]):
            #             probs[i,j, k] = val
            # probs = [[torch.tensor(x[1][i]).float() for i, role in enumerate(self.roles)] for x in sample]
            # print(probs[0][0])
            values = torch.tensor([x[2] for x in sample]).float()
            # print(values.shape)
            # print(sample)
            # print(sample.shape)
            # inputs = torch.tensor(sample[:,0])
            # print(inputs.shape)
            # probs = torch.tensor(sample[:,1])
            # print(probs.shape)
            # values = torch.tensor(sample[:,2])
            # print(values.shape)

            self.optimiser.zero_grad()
            output = self.net(inputs)
            # print(probs.shape, output[0].shape)
            # print(values.shape, output[1].shape)
            loss = self.loss_func(output[0], probs) + self.loss_func(output[1], values)
            sum_loss += loss
            loss.backward()
            self.optimiser.step()

            print('Loss is', loss.item())
        # self.losses.append(sum_loss/epochs)
        return sum_loss/epochs


    def save(self, game, i):
        path = os.path.join('models', game)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        # save_path = self.saver.save(self.sess, path + '/step-%06d.ckpt' % i)
        save_path = path + f'/step-{i:06d}.ckpt'
        torch.save(self.net.state_dict(), save_path)
        print("Saved model as", save_path)
        return save_path

    def load(self, path):
        try:
            self.net.load_state_dict(torch.load(path))
            print("Loaded", path)
        except Exception as e:
            print("starting from new state", e)
        # print('Loaded model from', path)

    def load_most_recent(self, game):
        models = os.path.join(pathlib.Path(__file__).parent, 'models')
        path = os.path.join(models, game)
        newest = max(os.listdir(path))[:-5]
        self.load(os.path.join(path, newest))
