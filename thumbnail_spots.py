"""Opener: where AIs hid secret codes (no coordination) and how often another AI found them.
Usage: python thumbnail_spots.py  (writes figures/thumbnail_spots.png)"""
import collections
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

O = {k: np.array(v, float) for k, v in json.load(open("brains/ttt_3x3_k3_h32.json")).items()}


def spot(w):
    W = {k: np.array(v, float) for k, v in w.items()}
    if len(W["b1"]) > len(O["b1"]):
        return "In a new neuron added\nat the end"
    db1 = np.where(np.abs(W["b1"] - O["b1"]) > 1e-9)[0]
    db2 = np.where(np.abs(W["b2"] - O["b2"]) > 1e-9)[0]
    if len(db1) and 0 in db1 and len(db1) <= 8:
        return "In the first few\nneurons"
    if len(db2) and not len(db1):
        return "In the final output\nnumbers"
    return "Somewhere else"


R = [json.load(open(p, encoding="utf-8")) for p in glob.glob("results/stego/receive/*/*.json")]
cnt, found = collections.Counter(), collections.defaultdict(list)
for p in glob.glob("results/stego/brains/*/stego*.json"):
    d = json.load(open(p, encoding="utf-8"))
    s = spot(d["weights"])
    cnt[s] += 1
    found[s] += [r["recovered"] for r in R if r["sender"] == d["sender"] and r["i"] == d["i"]]
order = ["In a new neuron added\nat the end", "In the first few\nneurons", "In the final output\nnumbers", "Somewhere else"]
order = [o for o in order if cnt[o]]
total = sum(cnt.values())
overall = 100 * np.mean([r["recovered"] for r in R])

fig = plt.figure(figsize=(11.7, 6.6), facecolor="white")
fig.text(0.04, 0.88, "AIs hid secret codes in the same few places", fontsize=28, weight="bold")
fig.text(0.04, 0.81, f"without agreeing on anything, so another AI found the code {overall:.0f}% of the time",
         fontsize=17, color="#444")
ax = fig.add_axes([0.08, 0.19, 0.84, 0.52])
x = np.arange(len(order))
share = [100 * cnt[o] / total for o in order]
bars = ax.bar(x, share, 0.55, color=["#2f5f8a" if o != "Somewhere else" else "#9aa9b8" for o in order])
for b, o in zip(bars, order):
    f = 100 * np.mean(found[o])
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{cnt[o]} of {total} codes", ha="center",
            fontsize=17, weight="bold")
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() / 2, f"found\n{f:.0f}%", ha="center", va="center",
            fontsize=15, color="white", weight="bold")
ax.set_xticks(x, order, fontsize=14)
ax.set_ylim(0, max(share) * 1.2)
ax.set_yticks([])
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#ccc")
ax.tick_params(axis="x", length=0, pad=10)
fig.text(0.04, 0.05, "6 AI models from 3 labs, random 4-letter codes. "
         "Every network still worked perfectly.", fontsize=10, color="#666")
fig.text(0.96, 0.05, "github.com/EnesYilmazcode/rewire", fontsize=10, color="#666", ha="right")
fig.savefig("figures/thumbnail_spots.png", dpi=110)
print({o.replace(chr(10), ' '): (cnt[o], round(100 * np.mean(found[o]))) for o in order}, round(overall))
