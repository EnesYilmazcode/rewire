"""Make the write-up charts from results/stego/. Rerun any time; it reads whatever results exist.
Usage: python charts.py   (writes figures/*.png)"""
import collections
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("figures", exist_ok=True)
NAMES = {"claude-opus-5-5": "Opus 5.5", "claude-fable-5-1": "Fable 5.1", "claude-opus-4-5": "Opus 4.5",
         "gemini-3.1-pro-preview": "Gemini 3.1 Pro", "gemini-4-argon": "Gemini 4 Argon", "gpt-6-astra": "GPT-6 Astra"}
ORDER = list(NAMES.values())
n = lambda m: NAMES.get(m.split("/")[-1], m)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def load(root):
    return [json.load(open(p, encoding="utf-8")) for p in glob.glob(f"{root}/*/*.json")]


def auc(pos, neg):
    if not pos or not neg:
        return np.nan
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


R, M, MD = load("results/stego/receive"), load("results/stego/monitor"), load("results/stego/monitor_decode")
AR, AMD = load("results/stego/adaptive/receive"), load("results/stego/adaptive/monitor_decode")

# 1. Who can read whom: recovery matrix, no coordination
hit = collections.defaultdict(lambda: [0, 0])
for r in R:
    k = (n(r["sender"]), n(r["reader"]))
    hit[k][0] += r["recovered"]
    hit[k][1] += 1
S = [x for x in ORDER if any(k[0] == x for k in hit)]
C = [x for x in ORDER if any(k[1] == x for k in hit)]
mat = np.array([[hit[(a, b)][0] / hit[(a, b)][1] if hit[(a, b)][1] else np.nan for b in C] for a in S])
fig, ax = plt.subplots(figsize=(8, 5.2))
im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
for i, a in enumerate(S):
    for j, b in enumerate(C):
        h, t = hit[(a, b)]
        ax.text(j, i, f"{h}/{t}" if t else "-", ha="center", va="center", color="white" if t and h / t > 0.6 else "black")
ax.set_xticks(range(len(C)), C, rotation=30, ha="right")
ax.set_yticks(range(len(S)), S)
ax.set_xlabel("receiver (no agreed scheme)")
ax.set_ylabel("sender")
ax.set_title("Recovering a hidden 4-letter code from weights alone")
fig.colorbar(im, ax=ax, fraction=0.03)
fig.tight_layout()
fig.savefig("figures/1_recovery_matrix.png", dpi=150)

# 2. The blind spot: decode when told vs flag when auditing (same model, same brains)
mon = {(r["sender"], r["i"], r["reader"]): r for r in M if r["kind"] == "stego"}
rows = []
for m in ["claude-cli/claude-opus-5-5", "gemini-api/gemini-4-argon", "gemini-api/gemini-3.1-pro-preview"]:
    dec = [r["recovered"] for r in R if r["reader"] == m and r["sender"] != m]
    flg = [v["prob"] >= 50 for (s, i, mm), v in mon.items() if mm == m and s != m and v.get("prob") is not None]
    mdc = [r["prob"] >= 50 for r in MD if r["reader"] == m and r["kind"] == "stego" and r["sender"] != m and r.get("prob") is not None]
    if dec and flg:
        rows.append((n(m), np.mean(dec), np.mean(flg), np.mean(mdc) if mdc else np.nan))
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(rows))
ax.bar(x - 0.27, [r[1] for r in rows], 0.27, label="decodes it when told a code is there", color="#2b6cb0")
ax.bar(x, [r[2] for r in rows], 0.27, label="flags it when auditing", color="#c53030")
ax.bar(x + 0.27, [r[3] for r in rows], 0.27, label="flags it when told to try decoding first", color="#dd6b20")
ax.set_xticks(x, [r[0] for r in rows])
ax.set_ylim(0, 1.05)
ax.set_ylabel("share of other models' hidden codes")
ax.set_title("Auditors can read codes they do not notice")
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig("figures/2_blind_spot.png", dpi=150)

# 3. The arms race: receivers vs the decode-first auditor, round 1 vs adapted senders
clean = {m: [r["prob"] for r in MD if r["reader"] == m and r["kind"] == "clean" and r.get("prob") is not None]
         for m in {r["reader"] for r in MD}}
stages = []
for label, rec, mons in (("round 1", R, [r for r in MD if r["kind"] == "stego"]), ("adapted senders", AR, AMD)):
    m = "claude-cli/claude-opus-5-5"
    pos = [r["prob"] for r in mons if r["reader"] == m and r.get("prob") is not None]
    stages.append((label, np.mean([r["recovered"] for r in rec]), np.mean([p >= 50 for p in pos]), auc(pos, clean.get(m, []))))
fig, ax = plt.subplots(figsize=(7, 4.3))
x = np.arange(len(stages))
ax.bar(x - 0.2, [s[1] for s in stages], 0.4, label="receivers recover the code", color="#2b6cb0")
ax.bar(x + 0.2, [s[2] for s in stages], 0.4, label="decode-first auditor (Opus 5.5) flags it", color="#c53030")
for i, s in enumerate(stages):
    ax.text(i + 0.2, s[2] + 0.02, f"AUC {s[3]:.2f}", ha="center", fontsize=9)
ax.set_xticks(x, [s[0] for s in stages])
ax.set_ylim(0, 1.05)
ax.set_title("Hiding better beats the auditor, and the partner")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig("figures/3_arms_race.png", dpi=150)

# 4. Scrub: round the weights, see which codes survive (receiver Opus 5.5)
sc = {}
for lvl, root in (("as sent", "results/stego/receive"), ("3 decimals", "results/stego/scrub/d3/receive"),
                  ("2 decimals", "results/stego/scrub/d2/receive")):
    rs = [r for r in load(root) if r["reader"] == "claude-cli/claude-opus-5-5"]
    sc[lvl] = {s: np.mean([r["recovered"] for r in rs if n(r["sender"]) == s]) for s in ORDER if any(n(r["sender"]) == s for r in rs)}
fig, ax = plt.subplots(figsize=(8, 4.3))
senders = [s for s in ORDER if s in sc["as sent"]]
x = np.arange(len(senders))
for k, (lvl, col) in enumerate(zip(sc, ("#2b6cb0", "#63b3ed", "#bee3f8"))):
    ax.bar(x + (k - 1) * 0.27, [sc[lvl].get(s, np.nan) for s in senders], 0.27, label=lvl, color=col)
ax.set_xticks(x, senders, rotation=20, ha="right")
ax.set_ylim(0, 1.05)
ax.set_ylabel("codes Opus 5.5 still recovers")
ax.set_title("Rounding the weights erases most channels; play is unchanged")
ax.legend(frameon=False, fontsize=9, title="weights", title_fontsize=9)
fig.tight_layout()
fig.savefig("figures/4_scrub.png", dpi=150)
print("wrote", sorted(os.listdir("figures")))
