"""Train a brain to copy perfect play on every position. Usage: python train.py --n 3 --k 3 --hidden 32"""
import argparse
import json
import os
import time

import numpy as np
import torch

from net import Brain
from score import Table, features, score
from solver import digits

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=3)
p.add_argument("--k", type=int, default=3)
p.add_argument("--variant", default="standard")
p.add_argument("--hidden", type=int, default=32)
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--batch", type=int, default=0, help="0 = whole set every step")
p.add_argument("--out", default="brains")
a = p.parse_args()

torch.manual_seed(0)
n2 = a.n * a.n
t = Table(a.n, a.k, a.variant)
N = len(t.idx)
print(f"{N:,} positions")


def batch(rows):
    D = digits(t.idx[rows], n2)
    X = torch.from_numpy(features(D))
    bits = torch.from_numpy(((t.best[rows][:, None] >> np.arange(n2)) & 1).astype(np.float32))
    return X, bits / bits.sum(1, keepdim=True), torch.from_numpy(D == 0)


full = batch(np.arange(N)) if a.batch == 0 else None
net = torch.nn.Sequential(torch.nn.Linear(3 * n2, a.hidden), torch.nn.ReLU(), torch.nn.Linear(a.hidden, n2))
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
rng = np.random.default_rng(0)
t0 = time.time()
for step in range(a.steps):
    X, Y, legal = full or batch(rng.integers(0, N, a.batch))
    logits = net(X).masked_fill(~legal, -1e9)
    loss = -(Y * torch.log_softmax(logits, 1)).sum(1).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    if step % max(1, a.steps // 10) == 0:
        print(step, round(loss.item(), 4), f"{time.time() - t0:.0f}s")

l1, l2 = net[0], net[2]
brain = Brain(*(x.detach().numpy() for x in (l1.weight, l1.bias, l2.weight, l2.bias)))
os.makedirs(a.out, exist_ok=True)
tag = "" if a.variant == "standard" else "_" + a.variant
name = f"{a.out}/ttt_{a.n}x{a.n}_k{a.k}{tag}_h{a.hidden}"
brain.to_json(name + ".json")
r = score(Brain.from_json(name + ".json"), a.n, a.k, table=t)  # score the rounded weights we ship
r["params"] = int(sum(x.numel() for x in net.parameters()))
print(json.dumps(r, indent=1))
with open(name + ".eval.json", "w") as f:
    json.dump(r, f, indent=1)
