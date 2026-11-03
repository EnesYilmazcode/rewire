"""Draw a brain: one card per hidden neuron, showing what it looks at and which moves it votes for.
Usage: python draw.py brains/ttt_3x3_k3_h32.json out.png"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from net import Brain

brain = Brain.from_json(sys.argv[1])
H, D = brain.W1.shape
n = int(round((D // 3) ** 0.5))
order = np.argsort(-np.abs(brain.W2).sum(0))  # most influential neurons first
cols, rows = 4, int(np.ceil(H / 4))
lim, olim = np.abs(brain.W1).max(), np.abs(brain.W2).max()
titles = ["mine", "theirs", "empty", "votes"]

fig, axes = plt.subplots(rows, cols * 5, figsize=(cols * 5 * 0.9, rows * 1.05 + 1.2), facecolor="white",
                         gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.35] * cols})
fig.suptitle("Tic-tac-toe brain, 32 neurons, 1,193 numbers.  Red = positive, blue = negative.\n"
             "mine / theirs / empty: which squares switch the neuron on.   votes: which square it pushes the player to take.",
             fontsize=12)
for ax in axes.flat:
    ax.axis("off")
for idx, j in enumerate(order):
    r, c = divmod(idx, cols)
    tiles = [brain.W1[j, k * n * n:(k + 1) * n * n].reshape(n, n) for k in range(3)] + [brain.W2[:, j].reshape(n, n)]
    for t, tile in enumerate(tiles):
        ax = axes[r, c * 5 + t]
        ax.axis("on")
        v = olim if t == 3 else lim
        ax.imshow(tile, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#333" if t == 3 else "#bbb")
            s.set_linewidth(1.4 if t == 3 else 0.8)
        ax.set_title(f"neuron {j}" if t == 0 else (titles[t] if r == 0 else ""), fontsize=8, loc="left" if t == 0 else "center")
        if r == 0 and t == 0:
            ax.set_title(f"mine\nneuron {j}", fontsize=8, loc="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(sys.argv[2], dpi=110)
print(sys.argv[2])
