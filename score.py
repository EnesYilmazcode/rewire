"""Score a brain on n x n tic-tac-toe against the solved game.

  keeps_result  share of all positions in play where its move keeps the best outcome (main score)
  best_move     share where its move is a best move outright
  vs_perfect    results over many games against a perfect player that picks randomly among best moves
  vs_random     results against a random player
Usage: python score.py brains/ttt_3x3_k3_h32.json --n 3 --k 3
"""
import argparse
import json

import numpy as np

from game import Game
from net import Brain
from solver import digits, solve


def features(D):
    """Board digits (N, n2) -> mover's view [mine, theirs, empty] (N, 3*n2)."""
    count = (D != 0).sum(1)
    me = np.where(count % 2 == 0, 1, 2)[:, None]
    mine = D == me
    empty = D == 0
    theirs = ~mine & ~empty
    return np.concatenate([mine, theirs, empty], 1).astype(np.float32)


def choose(brain, D):
    s = np.maximum(0, features(D) @ brain.W1.T.astype(np.float32) + brain.b1) @ brain.W2.T.astype(np.float32) + brain.b2
    s = np.where(D == 0, s, -np.inf)
    return s.argmax(1)


class Table:
    def __init__(self, n, k, variant="standard"):
        self.n, self.k, self.variant = n, k, variant
        self.idx, self.best, self.good, _ = solve(n, k, variant)
        order = np.argsort(self.idx)
        self.sorted_idx, self.order = self.idx[order], order

    def lookup(self, board_ints):
        return self.order[np.searchsorted(self.sorted_idx, board_ints)]


def score(brain, n, k, games=2000, seed=0, table=None, variant="standard"):
    t = table or Table(n, k, variant)
    n2 = n * n
    out = {}
    chunk = 500_000
    keep = top = 0
    for s in range(0, len(t.idx), chunk):
        D = digits(t.idx[s:s + chunk], n2)
        m = choose(brain, D)
        keep += ((t.good[s:s + chunk] >> m) & 1).sum()
        top += ((t.best[s:s + chunk] >> m) & 1).sum()
    out["keeps_result"] = round(float(keep / len(t.idx)), 4)
    out["best_move"] = round(float(top / len(t.idx)), 4)

    g = Game(n, k, t.variant)
    pw = 3 ** np.arange(n2, dtype=np.int64)
    rng = np.random.default_rng(seed)

    def match(perfect):
        """Play all games in lockstep. Brain is X in even games, O in odd ones."""
        D = np.zeros((games, n2), np.int64)
        brain_side = np.where(np.arange(games) % 2 == 0, 1, 2)
        done = np.zeros(games, bool)
        result = np.zeros(games, int)  # +1 brain win, 0 draw, -1 loss
        for ply in range(n2):
            mover = 1 if ply % 2 == 0 else 2
            act = ~done
            if not act.any():
                break
            bm = choose(brain, D[act])
            if perfect:
                masks = t.best[t.lookup(D[act] @ pw)]
            else:
                masks = ((D[act] == 0) * (1 << np.arange(n2))).sum(1)
            opts = (masks[:, None] >> np.arange(n2)) & 1
            r = rng.random(opts.shape) * opts
            om = r.argmax(1)
            mv = np.where(brain_side[act] == mover, bm, om)
            rows = np.where(act)[0]
            D[rows, mv] = mover
            for i in rows:
                w = g.winner(tuple(D[i]))
                if w:
                    done[i] = True
                    result[i] = 1 if w == brain_side[i] else -1
                elif not (D[i] == 0).any():
                    done[i] = True
        return {"win": round(float((result == 1).mean()), 3), "draw": round(float((result == 0).mean()), 3),
                "loss": round(float((result == -1).mean()), 3)}

    out["vs_perfect"] = match(True)
    out["vs_random"] = match(False)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("brain")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--variant", default="standard")
    a = p.parse_args()
    print(json.dumps(score(Brain.from_json(a.brain), a.n, a.k, variant=a.variant), indent=1))
