"""Train a small player brain to imitate perfect play. Usage: python train.py --n 3 --k 3 --hidden 32"""
import argparse
import json
import os

import numpy as np
import torch

from game import Game, encode
from net import Brain, evaluate

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=3)
p.add_argument("--k", type=int, default=3)
p.add_argument("--hidden", type=int, default=32)
p.add_argument("--epochs", type=int, default=3000)
p.add_argument("--out", default="brains")
a = p.parse_args()

torch.manual_seed(0)
g = Game(a.n, a.k)
pos = g.positions()
X = torch.tensor([encode(b, g) for b in pos])
Y = torch.zeros(len(pos), a.n * a.n)
for i, b in enumerate(pos):
    best = g.best_moves(b)
    Y[i, best] = 1.0 / len(best)  # spread target over all equally good moves
legal = X[:, 2 * a.n * a.n:] > 0
print(f"{len(pos)} positions")

net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], a.hidden), torch.nn.ReLU(), torch.nn.Linear(a.hidden, a.n * a.n))
opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
for ep in range(a.epochs):
    logits = net(X).masked_fill(~legal, -1e9)
    loss = -(Y * torch.log_softmax(logits, 1)).sum(1).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if ep % 500 == 0:
        print(ep, round(loss.item(), 4))

l1, l2 = net[0], net[2]
brain =Brain(l1.weight.detach().numpy(), l1.bias.detach().numpy(), l2.weight.detach().numpy(), l2.bias.detach().numpy())
os.makedirs(a.out, exist_ok=True)
name = f"{a.out}/ttt_{a.n}x{a.n}_k{a.k}_h{a.hidden}"
brain.to_json(name + ".json")
brain = Brain.from_json(name + ".json")  # score the rounded weights we actually ship
r = evaluate(brain, g)
r["params"] = int(sum(x.numel() for x in net.parameters()))
print(json.dumps(r, indent=1))
with open(name + ".eval.json", "w") as f:
    json.dump(r, f, indent=1)
