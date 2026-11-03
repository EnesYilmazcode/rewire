"""Draft X thumbnail in the benchbench style. Usage: python thumbnail.py  (writes figures/thumbnail_draft.png)"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(root):
    return [json.load(open(p, encoding="utf-8")) for p in glob.glob(f"{root}/*/*.json")]


R, M = load("results/stego/receive"), load("results/stego/monitor")
m = "claude-cli/claude-opus-5-5"
dec = [r["recovered"] for r in R if r["reader"] == m and r["sender"] != m]
flg = [r["prob"] >= 50 for r in M if r["reader"] == m and r["kind"] == "stego" and r["sender"] != m and r.get("prob") is not None]
vals = [100 * np.mean(dec), 100 * np.mean(flg)]

fig = plt.figure(figsize=(11.7, 6.6), facecolor="white")
fig.text(0.04, 0.89, "Claude can decode the hidden message.", fontsize=28, weight="bold")
fig.text(0.04, 0.81, "It just doesn't notice it.", fontsize=28, weight="bold", color="#a83f34")
ax = fig.add_axes([0.2, 0.17, 0.6, 0.55])
bars = ax.bar([0, 1], vals, 0.55, color=["#2f5f8a", "#a83f34"])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.0f}%", ha="center", fontsize=30, weight="bold")
ax.set_xticks([0, 1], ["Told a message is hidden,\nit finds it", "Asked to audit the network,\nit flags it"],
              fontsize=15)
ax.set_ylim(0, 100)
ax.set_yticks([])
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#ccc")
ax.tick_params(axis="x", length=0, pad=10)
fig.text(0.04, 0.05, f"Claude Opus 5.5, {len(dec)} hidden codes from other AIs. Gemini 3.1 Pro shows no such gap.", fontsize=10, color="#666")
fig.text(0.96, 0.05, "github.com/EnesYilmazcode/rewire", fontsize=10, color="#666", ha="right")
fig.savefig("figures/thumbnail_draft.png", dpi=110)
print([round(v) for v in vals], len(dec), len(flg))
