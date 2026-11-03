"""Tic-tac-toe on an n x n board with k in a row, plus rule variants. Perfect play by memoized minimax.

Variants: standard; nodiag (only rows and columns count); misere (completing a line loses)."""
from functools import lru_cache
import itertools


def lines(n, k, variant="standard"):
    dirs = ((0, 1), (1, 0)) if variant == "nodiag" else ((0, 1), (1, 0), (1, 1), (1, -1))
    out = []
    for r, c in itertools.product(range(n), range(n)):
        for dr, dc in dirs:
            cells = [(r + i * dr, c + i * dc) for i in range(k)]
            if all(0 <= rr < n and 0 <= cc < n for rr, cc in cells):
                out.append(tuple(rr * n + cc for rr, cc in cells))
    return out


class Game:
    """Board is a tuple of n*n ints: 0 empty, 1 X, 2 O. X moves first."""

    def __init__(self, n=3, k=3, variant="standard"):
        self.n, self.k, self.variant = n, k, variant
        self.lines = lines(n, k, variant)
        self.value = lru_cache(maxsize=None)(self._value)

    def winner(self, b):
        """Who won the game (misere: the player who did NOT complete the line)."""
        for ln in self.lines:
            v = b[ln[0]]
            if v and all(b[i] == v for i in ln):
                return 3 - v if self.variant == "misere" else v
        return 0

    def to_move(self, b):
        return 1 if b.count(1) == b.count(2) else 2

    def moves(self, b):
        return [i for i, v in enumerate(b) if v == 0]

    def play(self, b, i):
        b = list(b)
        b[i] = self.to_move(b)
        return tuple(b)

    def _value(self, b):
        """Score for the player to move: +1 win, 0 draw, -1 loss (faster wins preferred)."""
        if self.winner(b):
            return -1.0 if self.variant != "misere" else 1.0  # previous player just completed a line
        ms = self.moves(b)
        if not ms:
            return 0.0
        return max(-0.99 * self.value(self.play(b, m)) for m in ms)

    def best_moves(self, b):
        scores = {m: -self.value(self.play(b, m)) for m in self.moves(b)}
        top = max(scores.values())
        return [m for m, s in scores.items() if abs(s - top) < 1e-9]

    def positions(self):
        """Every reachable, non-terminal position."""
        seen, stack = set(), [tuple([0] * self.n * self.n)]
        while stack:
            b = stack.pop()
            if b in seen or self.winner(b) or not self.moves(b):
                continue
            seen.add(b)
            stack.extend(self.play(b, m) for m in self.moves(b))
        return sorted(seen)


def encode(b, g):
    """Input from the mover's view: [mine x n*n, theirs x n*n, empty x n*n]."""
    me = g.to_move(b)
    return [float(v == me) for v in b] + [float(v not in (0, me)) for v in b] + [float(v == 0) for v in b]
