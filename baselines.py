"""Reference points for each 4x4 task, so an editor's score means something.

  random                 random weights, the floor
  transplant             the 3x3 brain copied into the top-left corner of the 4x4 board, nothing else changed
  handcoded_plain_rules  a brain written by a short script from the normal rules: for each line, neurons that
                         count my pieces and theirs (while the line is open) and vote for its empty cells. Roughly
                         what a model writes from scratch without reading the old brain. Aces `told`; on the
                         secret tasks it shows what you get if you skip reading the weights.
  handcoded_knows_rule   the same script given the secret rule (no-diagonals only), what reading pays off as
  trained_h64            trained the normal way on all 9M positions, the ceiling
Usage: python baselines.py
"""
import json
import os

import numpy as np

from bench import TASKS, reuse
from game import lines
from net import Brain
from score import Table, score


def random_brain(h=32, seed=0):
    r = np.random.default_rng(seed)
    return Brain(r.normal(0, 0.5, (h, 48)), r.normal(0, 0.5, h), r.normal(0, 0.5, (16, h)), r.normal(0, 0.5, 16))


def transplant(patient):
    old = Brain.from_json(patient)
    cells = [r * 4 + c for r in range(3) for c in range(3)]
    W1 = np.zeros((32, 48))
    for block in range(3):
        W1[:, [16 * block + i for i in cells]] = old.W1[:, 9 * block:9 * block + 9]
    W2 = np.zeros((16, 32))
    b2 = np.full(16, -5.0)  # the extra row and column are never preferred
    W2[cells], b2[cells] = old.W2, old.b2
    return Brain(W1, old.b1, W2, b2)


def handcoded(variant="standard"):
    rows, b1, votes = [], [], []
    all_lines = lines(4, 4, variant)
    L = len(all_lines[0])  # 4 for straight lines, 3 for shapes
    # (whose pieces, how many already on the line, vote for each empty cell on it)
    plan = [("mine", L - 1, 1000), ("theirs", L - 1, 300), ("mine", L - 2, 6), ("theirs", L - 2, 4), ("mine", 1, 1)]
    for who, need, vote in plan:
        for ln in all_lines:
            w = np.zeros(48)
            own, other = (0, 16) if who == "mine" else (16, 0)
            w[[own + i for i in ln]] = 1.0     # count pieces of this side on the line
            w[[other + i for i in ln]] = -10.0  # any blocker shuts the neuron off
            rows.append(w)
            b1.append(-(need - 1))  # relu(count - need + 1): fires only at `need` or more
            v = np.zeros(16)
            v[list(ln)] = vote
            votes.append(v)
    b2 = np.array([1.0 if (r in (1, 2) and c in (1, 2)) else 0.0 for r in range(4) for c in range(4)])  # like the centre
    return Brain(np.array(rows), np.array(b1, float), np.array(votes).T, b2)


if __name__ == "__main__":
    out = {}
    for task, (patient, variant, _) in TASKS.items():
        t = Table(4, 4, variant)
        refs = {"random": random_brain(), "transplant": transplant(patient), "handcoded_plain_rules": handcoded("standard")}
        if variant in ("nodiag",) or variant.startswith("shapes_"):
            refs["handcoded_knows_rule"] = handcoded(variant)
        tag = "" if variant == "standard" else "_" + variant
        f = f"brains/ttt_4x4_k4{tag}_h64.json"
        if os.path.exists(f):
            refs["trained_h64"] = Brain.from_json(f)
        out[task] = {}
        for name, b in refs.items():
            s = score(b, 4, 4, table=t)
            s["hidden"], s["reuse"] = len(b.b1), reuse(b, patient)
            out[task][name] = s
            print(f"{task:7s} {name:22s} keeps {s['keeps_result']:.3f}  loss vs perfect {s['vs_perfect']['loss']:.3f}  "
                  f"win vs random {s['vs_random']['win']:.2f}  H={s['hidden']}  reuse {s['reuse']}", flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/baselines.json", "w"), indent=1)
