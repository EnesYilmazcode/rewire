"""When an AI guessed the hidden code wrong, how often was its guess a real English word?
The true codes are random letters; only about 1% of random 4-letter strings are English words.
Usage: python chart_words.py  (needs results/stego/failure_types.json from the failure analysis)"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {"claude-opus-5-5": "Claude Opus 5.5", "claude-fable-5-1": "Claude Fable 5.1", "claude-opus-4-5": "Claude Opus 4.5",
         "gemini-3.1-pro-preview": "Gemini 3.1 Pro", "gemini-4-argon": "Gemini 4 Argon", "gpt-6-astra": "GPT-6 Astra"}
t = json.load(open("results/stego/failure_types.json"))
rows = []
for m, c in t.items():
    word, rand = c.get("wrong: real English word", 0), c.get("wrong: random letters", 0)
    if word + rand:
        rows.append((NAMES[m], 100 * word / (word + rand), word, word + rand))
rows.sort(key=lambda r: -r[1])

LAB_COLOR = {"Claude": "#D97757", "GPT": "#111111", "Gemini": "#4285F4"}  # Anthropic orange, OpenAI black, Google blue


def lab_color(name):
    return next(c for k, c in LAB_COLOR.items() if name.startswith(k))

fig = plt.figure(figsize=(11.7, 6.6), facecolor="white")
fig.text(0.04, 0.88, "When AIs guessed wrong, they often saw words", fontsize=26, weight="bold")
fig.text(0.04, 0.82, "The hidden codes were random letters like XPQH. Guesses included MAZE, DUEL and IONS.",
         fontsize=14, color="#444")
ax = fig.add_axes([0.06, 0.2, 0.9, 0.52])
bars = ax.bar(range(len(rows)), [r[1] for r in rows], 0.6, color=[lab_color(r[0]) for r in rows])
for b, r in zip(bars, rows):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{r[1]:.0f}%", ha="center", fontsize=20, weight="bold")
ax.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=13)
ax.tick_params(axis="x", length=0, pad=8)
ax.set_ylim(0, 85)
ax.set_yticks([])
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#ccc")
fig.savefig("figures/5_word_bias.png", dpi=110)
print(rows)
