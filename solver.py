"""Solve n x n tic-tac-toe (k in a row) for every position at once, with numpy.

A board is one integer: sum(cell_i * 3**i), cell 0 empty, 1 X, 2 O. X moves first.
Solves layer by layer from full boards back to the empty board, then caches to solved/.

Saves, for every position that is still in play (no winner, not full):
  idx   board integer
  best  bitmask of the moves that keep the best score (fastest win, slowest loss)
  good  bitmask of the moves that keep the best outcome (still a win, still a draw)
  value score for the player to move: +1 win, 0 draw, -1 loss, shrunk by 0.99 per move
Usage: python solver.py --n 4 --k 4
"""
import argparse
import itertools
import os
import time

import numpy as np

from game import lines

DISCOUNT = 0.99


def layer_states(n2, c):
    """Every board with c pieces placed in legal turn order (X count = ceil(c/2))."""
    if c == 0:
        return np.zeros(1, np.int64)
    nx = (c + 1) // 2
    pw = 3 ** np.arange(n2, dtype=np.int64)
    out = []
    for occ in itertools.combinations(range(n2), c):
        occ = np.array(occ, dtype=np.int64)
        xs = np.array(list(itertools.combinations(range(c), nx)), dtype=np.int64).reshape(-1, nx)
        base = (2 * pw[occ]).sum()  # everything O, then flip the X subset to 1
        out.append(base - pw[occ][xs].sum(1))
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def digits(idx, n2):
    return (idx[:, None] // 3 ** np.arange(n2, dtype=np.int64)) % 3


def solve(n, k, variant="standard", cache_dir="solved"):
    path = f"{cache_dir}/ttt_{n}x{n}_k{k}{'' if variant == 'standard' else '_' + variant}.npz"
    if os.path.exists(path):
        d = np.load(path)
        return d["idx"], d["best"], d["good"], d["value"]
    n2 = n * n
    ln = np.array(lines(n, k, variant))
    pw = 3 ** np.arange(n2, dtype=np.int64)
    V = np.zeros(3 ** n2, dtype=np.float32)
    keep_idx, keep_best, keep_good, keep_val = [], [], [], []
    t0 = time.time()
    for c in range(n2, -1, -1):
        S = layer_states(n2, c)
        D = digits(S, n2).astype(np.int8)
        won = np.zeros(len(S), bool)
        for v in (1, 2):
            won |= (D[:, ln] == v).all(2).any(1)
        val = np.zeros(len(S), np.float32)
        val[won] = 1.0 if variant == "misere" else -1.0  # the player who just moved completed a line
        live = ~won & (c < n2)
        if live.any():
            Sl, Dl = S[live], D[live]
            mover = 1 if c % 2 == 0 else 2
            child = np.full((len(Sl), n2), -np.inf, np.float32)
            for i in range(n2):
                e = Dl[:, i] == 0
                child[e, i] = -DISCOUNT * V[Sl[e] + mover * pw[i]]
            best_val = child.max(1)
            val[live] = best_val
            best = (np.abs(child - best_val[:, None]) < 1e-6) & np.isfinite(child)
            outcome = np.sign(np.round(child, 4))
            good = (outcome == np.sign(np.round(best_val, 4))[:, None]) & np.isfinite(child)
            bits = 1 << np.arange(n2)
            keep_idx.append(Sl)
            keep_best.append((best * bits).sum(1).astype(np.int64))
            keep_good.append((good * bits).sum(1).astype(np.int64))
            keep_val.append(best_val)
        V[S] = val
        print(f"  {c:2d} pieces: {len(S):>9,} boards, {live.sum():>9,} in play  ({time.time() - t0:.0f}s)")
    idx = np.concatenate(keep_idx)
    best = np.concatenate(keep_best)
    good = np.concatenate(keep_good)
    value = np.concatenate(keep_val)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(path, idx=idx, best=best, good=good, value=value)
    return idx, best, good, value


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--variant", default="standard")
    a = p.parse_args()
    idx, best, good, value = solve(a.n, a.k, a.variant)
    empty = np.where(idx == 0)[0][0]
    print(f"{len(idx):,} positions in play. Empty board value: {value[empty]:+.3f}, best first moves mask {best[empty]:b}")
