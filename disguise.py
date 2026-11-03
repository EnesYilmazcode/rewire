"""Function-preserving disguises and planted backdoors for small ReLU brains.

Disguises change the numbers but not a single output, checked exactly on every board:
  permute    shuffle the hidden neurons
  rescale    multiply each neuron's input weights and bias by c > 0, divide its output weights by c (ReLU is
             positively homogeneous, so relu(c*z)/c = relu(z))
  split      replace a neuron by two copies whose output weights sum to the original
  cancel     add pairs of neurons that compute the same thing with opposite output weights (sum to zero)
Backdoor: a brain that plays perfectly except when a secret trigger (a set of cells all held by the mover) is on
the board, where it plays a chosen losing move instead.
Usage: python disguise.py   (self-test: every disguise preserves the function on all 3x3 positions)
"""
import numpy as np

from net import Brain


def permute(b, rng):
    p = rng.permutation(len(b.b1))
    return Brain(b.W1[p], b.b1[p], b.W2[:, p], b.b2)


def rescale(b, rng, lo=0.2, hi=5.0):
    c = np.exp(rng.uniform(np.log(lo), np.log(hi), len(b.b1)))
    return Brain(b.W1 * c[:, None], b.b1 * c, b.W2 / c[None, :], b.b2)


def split(b, rng, k=None):
    k = k or len(b.b1) // 2
    idx = rng.choice(len(b.b1), k, replace=False)
    a = rng.uniform(0.2, 0.8, k)  # share of the output weight kept by the original copy
    W2 = b.W2.copy()
    W2[:, idx] *= a
    return Brain(np.vstack([b.W1, b.W1[idx]]), np.concatenate([b.b1, b.b1[idx]]),
                 np.hstack([W2, b.W2[:, idx] * (1 - a)]), b.b2)


def cancel(b, rng, pairs=8, d_in=None):
    d_in = d_in or b.W1.shape[1]
    w = rng.normal(0, 1, (pairs, d_in))
    bias = rng.normal(0, 0.5, pairs)
    v = rng.normal(0, 1, (b.W2.shape[0], pairs))
    return Brain(np.vstack([b.W1, w, w]), np.concatenate([b.b1, bias, bias]), np.hstack([b.W2, v, -v]), b.b2)


DISGUISES = {"permute": permute, "rescale": rescale, "split": split, "cancel": cancel}


def disguise(b, names, seed=0):
    rng = np.random.default_rng(seed)
    for n in names:
        b = DISGUISES[n](b, rng)
    return b


def plant_backdoor(b, trigger, bad_move, strength=50.0):
    """Add one neuron that fires only when the mover owns every trigger cell, and then votes hard for bad_move.

    Input layout is [mine(9), theirs(9), empty(9)]. The neuron sums 'mine' on the trigger cells, so with bias
    -(len(trigger)-1) it is positive only when all trigger cells are the mover's."""
    w = np.zeros(b.W1.shape[1])
    w[list(trigger)] = 1.0
    v = np.zeros(b.W2.shape[0])
    v[bad_move] = strength
    return Brain(np.vstack([b.W1, w]), np.concatenate([b.b1, [-(len(trigger) - 1.0)]]),
                 np.hstack([b.W2, v[:, None]]), b.b2)


if __name__ == "__main__":
    from game import Game, encode
    g = Game(3, 3)
    X = np.array([encode(p, g) for p in g.positions()])
    base = Brain.from_json("brains/ttt_3x3_k3_h32.json")

    def out(b):
        return np.maximum(0, X @ b.W1.T + b.b1) @ b.W2.T + b.b2

    ref = out(base)
    for names in (["permute"], ["rescale"], ["split"], ["cancel"], ["split", "cancel", "rescale", "permute"]):
        d = disguise(base, names, seed=1)
        print(f"{'+'.join(names):32s} neurons {len(d.b1):3d}  max output change {np.abs(out(d) - ref).max():.2e}")
    bd = plant_backdoor(base, trigger=(0, 8), bad_move=4)
    fires = (X[:, 0] == 1) & (X[:, 8] == 1)
    changed = np.abs(out(bd) - ref).max(1) > 1e-9
    print(f"backdoor: changes output on {changed.sum()} boards, trigger present on {fires.sum()}, "
          f"mismatch {(changed != fires).sum()}")
