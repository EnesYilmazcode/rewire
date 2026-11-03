"""How a code hides in a weight: GPT-6 Astra's real XPQH example, split into ASCII groups.
Usage: python chart_ascii.py  (writes figures/6_ascii_example.png)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, GREY, INK = "#2f5f8a", "#888888", "#111111"
fig = plt.figure(figsize=(11.7, 6.6), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis("off")

ax.text(4, 53, "How an AI hid \"XPQH\" in the weights", fontsize=26, weight="bold", color=INK)
ax.text(4, 47.5, "GPT-6 Astra added this tiny number to every output, so the network's moves never change.",
        fontsize=13, color="#444")

ax.text(31, 34, "0.000", fontsize=40, family="DejaVu Sans Mono", ha="right", va="center", color=GREY)
groups, xs = ["088", "080", "081", "072"], [39, 53, 67, 81]
for g, x, letter in zip(groups, xs, "XPQH"):
    ax.add_patch(FancyBboxPatch((x - 5.6, 30.2), 11.2, 7.6, boxstyle="round,pad=0.2", fc="#eaf1f8", ec=BLUE, lw=1.5))
    ax.text(x, 34, g, fontsize=40, family="DejaVu Sans Mono", ha="center", va="center", color=BLUE, weight="bold")
    ax.annotate("", xy=(x, 17), xytext=(x, 29), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.5))
    ax.text(x + 1.2, 23, f"ASCII {int(g)}", fontsize=11, ha="left", va="center", color=GREY)
    ax.text(x, 11, letter, fontsize=54, weight="bold", ha="center", va="center", color=INK)
fig.savefig("figures/6_ascii_example.png", dpi=110)
print("figures/6_ascii_example.png")
