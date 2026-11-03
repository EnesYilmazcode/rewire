"""Opener in the benchbench style: one metric, one bar per AI.
Metric: share of secret codes hidden by OTHER AIs that this AI found, with no agreed method.
Usage: python thumbnail_models.py  (writes figures/thumbnail_models.png)"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = {"claude-opus-5-5": "Claude Opus 5.5", "claude-fable-5-1": "Claude Fable 5.1", "claude-opus-4-5": "Claude Opus 4.5",
         "gemini-3.1-pro-preview": "Gemini 3.1 Pro", "gemini-4-argon": "Gemini 4 Argon", "gpt-6-astra": "GPT-6 Astra"}
R = [json.load(open(p, encoding="utf-8")) for p in glob.glob("results/stego/receive/*/*.json")]
rows = []
for m in sorted({r["reader"] for r in R}):
    v = [r["recovered"] for r in R if r["reader"] == m and r["sender"] != m]
    rows.append((NAMES[m.split("/")[-1]], 100 * np.mean(v), len(v)))
rows.sort(key=lambda r: -r[1])

fig = plt.figure(figsize=(11.7, 6.6), facecolor="white")
fig.text(0.04, 0.88, "How often each AI found a secret code another AI hid", fontsize=26, weight="bold")
ax = fig.add_axes([0.06, 0.2, 0.9, 0.58])
bars = ax.bar(range(len(rows)), [r[1] for r in rows], 0.6, color="#a83f34")
for b, r in zip(bars, rows):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{r[1]:.0f}%", ha="center", fontsize=20, weight="bold")
ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=13)
ax.set_ylim(0, 112)
ax.set_yticks([])
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#ccc")
ax.tick_params(axis="x", length=0, pad=8)
n = min(r[2] for r in rows), max(r[2] for r in rows)
fig.text(0.04, 0.05, f"Codes hidden by the other models, no agreed method. {n[0]} to {n[1]} tries per model.",
         fontsize=10, color="#666")
fig.text(0.96, 0.05, "github.com/EnesYilmazcode/rewire", fontsize=10, color="#666", ha="right")
fig.savefig("figures/thumbnail_models.png", dpi=110)
print(rows)
