"""The player brain: input (mine/theirs/empty per cell) -> hidden ReLU -> one score per cell."""
import json
import random

import numpy as np

from game import Game, encode


class Brain:
    def __init__(self, W1, b1, W2, b2):
        self.W1, self.b1, self.W2, self.b2 = (np.asarray(x, dtype=np.float64) for x in (W1, b1, W2, b2))

    def scores(self, x):
        h = np.maximum(0, self.W1 @ x + self.b1)
        return self.W2 @ h + self.b2

    def move(self, b, g):
        s = self.scores(np.array(encode(b, g)))
        legal = g.moves(b)
        return max(legal, key=lambda i: s[i])  # illegal cells are masked out

    def to_json(self, path):
        d = {k: np.round(getattr(self, k), 3).tolist() for k in ("W1", "b1", "W2", "b2")}
        with open(path, "w") as f:
            json.dump(d, f)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["W1"], d["b1"], d["W2"], d["b2"])


def evaluate(brain, g, games=500, seed=0):
    """Optimal-move accuracy on every position, and results vs a perfect and a random opponent."""
    rng = random.Random(seed)
    pos = g.positions()
    acc = sum(brain.move(b, g) in g.best_moves(b) for b in pos) / len(pos)

    def match(opponent):
        res = {"win": 0, "draw": 0, "loss": 0}
        for i in range(games):
            brain_side = 1 if i % 2 == 0 else 2
            b = tuple([0] * g.n * g.n)
            while not g.winner(b) and g.moves(b):
                if g.to_move(b) == brain_side:
                    m = brain.move(b, g)
                else:
                    m = opponent(b)
                b = g.play(b, m)
            w = g.winner(b)
            res["draw" if w == 0 else "win" if w == brain_side else "loss"] += 1
        return {k: v / games for k, v in res.items()}

    return {
        "optimal_move_acc": round(acc, 4),
        "vs_perfect": match(lambda b: rng.choice(g.best_moves(b))),
        "vs_random": match(lambda b: rng.choice(g.moves(b))),
    }
