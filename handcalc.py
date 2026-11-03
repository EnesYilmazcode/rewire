"""Bookkeeping test: can a model run a forward pass of a small network by hand, with no code?

Pure arithmetic, no understanding needed. If this tracks the weight-writing cliff across models, the cliff is
mostly about exact bookkeeping. Knobs make the arithmetic easier: fewer decimals, fewer neurons.
Usage: python handcalc.py --models claude-cli/claude-opus-5-5 ... --decimals 3 1 --hidden 32 16
"""
import argparse
import concurrent.futures as cf
import json
import os
import re

import numpy as np

from bench import call
from game import Game, encode
from net import Brain

BOARDS = [  # 5 mid-game positions, fixed so every model sees the same work
    (1, 0, 0, 0, 2, 0, 0, 0, 0),
    (1, 2, 0, 0, 1, 0, 0, 0, 2),
    (0, 1, 0, 2, 1, 0, 0, 2, 0),
    (2, 0, 1, 0, 1, 0, 2, 0, 0),
    (1, 2, 1, 0, 2, 0, 0, 1, 0),
]

PROMPT = """Compute the output of this small neural network BY HAND for 5 inputs. Do not write or run code.

h = relu(W1 @ x + b1)   W1 is H x 27, b1 has H numbers
s = W2 @ h + b2         W2 is 9 x H, b2 has 9 numbers

WEIGHTS
{weights}

INPUTS (each x has 27 numbers)
{inputs}

End your reply with one JSON object and nothing after it: {{"s": [[9 numbers], [9 numbers], [9 numbers], [9 numbers], [9 numbers]]}}
Round each number to 2 decimals."""


def make_brain(hidden, decimals):
    b = Brain.from_json(f"brains/ttt_3x3_k3_h{hidden}.json")
    return Brain(*(np.round(getattr(b, k), decimals) for k in ("W1", "b1", "W2", "b2")))


def run(model, hidden, decimals):
    g = Game(3, 3)
    b = make_brain(hidden, decimals)
    xs = [encode(bd, g) for bd in BOARDS]
    truth = np.array([b.scores(np.array(x)) for x in xs])
    w = {k: getattr(b, k).round(decimals).tolist() for k in ("W1", "b1", "W2", "b2")}
    text = PROMPT.format(weights=json.dumps(w), inputs="\n".join(json.dumps([int(v) for v in x]) for x in xs))
    out = {"model": model, "hidden": hidden, "decimals": decimals, "truth": truth.round(3).tolist()}
    try:
        out.update(call(model, text))
        m = list(re.finditer(r'\{\s*"s"\s*:', out["reply"]))
        got = np.array(json.JSONDecoder().raw_decode(out["reply"][m[-1].start():])[0]["s"], dtype=float)
        err = np.abs(got - truth)
        out["got"] = got.tolist()
        out["within_0.1"] = round(float((err < 0.1).mean()), 3)  # share of the 45 numbers it got right
        out["best_move_right"] = round(float(np.mean([np.argmax(got[i] - 1e9 * (np.array(BOARDS[i]) != 0)) ==
                                                      np.argmax(truth[i] - 1e9 * (np.array(BOARDS[i]) != 0))
                                                      for i in range(5)])), 3)
    except Exception as e:
        out["error"] = repr(e)[:300]
    d = f"results/handcalc/{model.replace('/', '__')}"
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(f"{d}/h{hidden}_d{decimals}.json", "w"), indent=1)
    print(f"{model:36s} H={hidden:2d} dec={decimals}  " +
          (f"numbers right {out['within_0.1']:.0%}  best move {out['best_move_right']:.0%}" if "error" not in out
           else f"FAILED {out['error'][:120]}"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+")
    p.add_argument("--hidden", nargs="+", type=int, default=[32, 16])
    p.add_argument("--decimals", nargs="+", type=int, default=[3, 1])
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    jobs = [(m, h, d) for m in a.models for h in a.hidden for d in a.decimals
            if not os.path.exists(f"results/handcalc/{m.replace('/', '__')}/h{h}_d{d}.json")]
    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda j: run(*j), jobs))
