"""Results table: one row per model, one column group per task, baselines underneath.
Usage: python report.py            (prints markdown, writes results/TABLE.md)"""
import glob
import json
import os

from bench import LINEUP, TASKS


def cell(r):
    if r is None:
        return "not run"
    if r.get("score") is None:
        return "failed"
    s = r["score"]
    tag = " *" if r.get("ran_commands") else ""
    return f"{s['keeps_result']:.3f} / {s['vs_perfect']['loss']:.0%} / {r['reuse']:.0%}{tag}"


def load(task, model):
    runs = sorted(glob.glob(f"results/{task}/{model.replace('/', '__')}/run*.json"))
    return json.load(open(runs[0])) if runs else None


tasks = [t for t in TASKS if os.path.isdir(f"results/{t}")]
lines = ["| Model | " + " | ".join(tasks) + " |", "|---|" + "---|" * len(tasks)]
for m in LINEUP:
    rows = [load(t, m) for t in tasks]
    if any(rows):
        lines.append(f"| {m.split('/', 1)[1]} | " + " | ".join(cell(r) for r in rows) + " |")
base = json.load(open("results/baselines.json"))
for name in ["random", "transplant", "handcoded_plain_rules", "trained_h64"]:
    vals = [base[t].get(name) for t in tasks]
    lines.append(f"| *{name}* | " + " | ".join(
        f"{v['keeps_result']:.3f} / {v['vs_perfect']['loss']:.0%} / {v['reuse']:.0%}" if v else "" for v in vals) + " |")
lines.append("\nCells: keeps result / games lost to a perfect player / old neurons reused. * = Codex ran shell commands.")
fails = []
for t in tasks:
    for m in LINEUP:
        r = load(t, m)
        if r and r.get("score") is None:
            fails.append(f"- {t} {m}: {r.get('error')}")
        if r and r.get("rule_guess"):
            fails.append(f"- {t} {m} RULE: {r['rule_guess'][:160]}")
out = "\n".join(lines + ["", "Notes:"] + fails)
print(out)
open("results/TABLE.md", "w", encoding="utf-8").write(out)
