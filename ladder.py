"""A ladder of small weight-reading questions with exact answers, easy enough that a 0.8B model gets some right.

Every item is multiple choice. The model is scored by which option it finds most likely, so a run is quick,
deterministic, and two runs (before and after an edit) are directly comparable.

Levels:
  peak    which of 9 weights is largest                                   (reading one number list)
  line    which board line a neuron watches, from its 9 weights            (reading a pattern)
  sign    is w . x + b positive or negative, for a given board             (one neuron, by hand)
  vote    given 3 hidden activations and their output weights, which cell wins  (one layer, by hand)
"""
import random

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
LETTERS = "ABCDEFGH"


def fmt(ws):
    return "[" + ", ".join(f"{w:.2f}" for w in ws) + "]"


def peak(r):
    w = [round(r.gauss(0, 1), 2) for _ in range(9)]
    ans = max(range(9), key=lambda i: w[i])
    q = (f"A neuron has one weight for each cell of a 3x3 board, cells numbered 0 to 8.\n"
         f"Weights: {fmt(w)}\nWhich cell has the largest weight?\nAnswer:")
    return q, [f" {i}" for i in range(9)], ans


def line(r):
    target = r.randrange(8)
    w = [r.gauss(0, 0.4) for _ in range(9)]
    for c in LINES[target]:
        w[c] += r.uniform(1.0, 2.0)
    opts = "\n".join(f"{LETTERS[i]}: cells {' '.join(map(str, ln))}" for i, ln in enumerate(LINES))
    q = (f"A neuron in a tic-tac-toe network has one weight per cell of the 3x3 board (cells 0 to 8, row by row).\n"
         f"Weights: {fmt(w)}\nWhich line of three does this neuron detect?\n{opts}\nAnswer:")
    return q, [f" {c}" for c in LETTERS], target


def sign(r):
    w = [r.gauss(0, 1) for _ in range(9)]
    x = [r.choice([0, 0, 1]) for _ in range(9)]
    b = r.gauss(0, 0.5)
    val = sum(wi * xi for wi, xi in zip(w, x)) + b
    if abs(val) < 0.3:  # too close to call by eye, draw again
        return sign(r)
    q = (f"Weights w = {fmt(w)}\nBias b = {b:.2f}\nInput x = {x}\n"
         f"Is w . x + b positive or negative?\nAnswer:")
    return q, [" positive", " negative"], 0 if val > 0 else 1


def vote(r):
    h = [round(r.uniform(0, 2), 1) for _ in range(3)]
    W2 = [[round(r.gauss(0, 1), 2) for _ in range(3)] for _ in range(9)]
    s = [sum(W2[c][j] * h[j] for j in range(3)) for c in range(9)]
    top = sorted(s, reverse=True)
    if top[0] - top[1] < 0.3:
        return vote(r)
    rows = "\n".join(f"cell {c}: {fmt(W2[c])}" for c in range(9))
    q = (f"Three hidden neurons have activations h = {h}.\nEach board cell scores sum_j w[j] * h[j], with weights:\n"
         f"{rows}\nWhich cell gets the highest score?\nAnswer:")
    return q, [f" {i}" for i in range(9)], s.index(top[0])


LEVELS = {"peak": peak, "line": line, "sign": sign, "vote": vote}


def items(level, n, seed):
    r = random.Random(f"{level}-{seed}")
    return [LEVELS[level](r) for _ in range(n)]


if __name__ == "__main__":
    for lv in LEVELS:
        q, opts, ans = items(lv, 1, 0)[0]
        print(f"=== {lv} (answer {opts[ans]!r}) ===\n{q}\n")
