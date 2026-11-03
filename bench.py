"""Hand a model a trained 3x3 brain, ask it to rewire the weights for 4x4, score what comes back.

Tasks:
  told      the brain plays normal tic-tac-toe and we say so. Rewire it for 4x4, four in a row.
  nodiag    the brain plays a secret variant (only rows and columns count). We do not say which. The model has to
            read the rule out of the weights, then build a 4x4 brain for the same variant.
  misere    same, secret variant is misere (completing a line loses).
  secret_std  control: the normal brain, presented as secret. A model that names a variant here is guessing.
A script that writes a fresh brain from the plain rules aces `told` and fails the secret tasks, so the secret
tasks are the ones that need the old brain to be read.

Every call goes through the Vercel AI Gateway (one key, every lab). Each run saves the full prompt, the raw reply,
token usage and the score to results/<task>/<model>/run<N>.json, so nothing has to be rerun to re-read it.
Usage: python bench.py --tasks told nodiag misere --models openai/gpt-6-astra anthropic/claude-opus-5.5 --runs 1
       python bench.py --dry   (print the prompts and their size, call nothing)
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import time
import urllib.request

import numpy as np

from net import Brain
from score import Table, score

PATIENT = "brains/ttt_3x3_k3_h32.json"
TASKS = {  # task: (3x3 patient, variant it is scored on, which prompt)
    "told": ("brains/ttt_3x3_k3_h32.json", "standard", "told"),
    "nodiag": ("brains/ttt_3x3_k3_nodiag_h32.json", "nodiag", "secret"),
    "misere": ("brains/ttt_3x3_k3_misere_h32.json", "misere", "secret"),
    # control: the normal brain, presented as a secret variant. Naming a variant here means the model guessed.
    "secret_std": ("brains/ttt_3x3_k3_h32.json", "standard", "secret"),
    # random winning shapes: thousands of possible rule sets, so the rule cannot be guessed, only read
    "shapes": ("brains/ttt_3x3_k3_shapes_a_h32.json", "shapes_a", "shapes"),
    # the same brain in disguise: identical function on every board, different numbers (disguise.py)
    "shapes_perm": ("brains/disguised/shapes_a_permute.json", "shapes_a", "shapes"),
    "shapes_rescale": ("brains/disguised/shapes_a_rescale.json", "shapes_a", "shapes"),
    "shapes_full": ("brains/disguised/shapes_a_full.json", "shapes_a", "shapes"),
}
MAX_HIDDEN = 64
GATEWAY = "https://ai-gateway.vercel.sh/v1/chat/completions"

LINEUP = [  # Claude and GPT through the subscriptions, Gemini through its API, open models through the gateway
    "claude-cli/claude-sonnet-4-5", "claude-cli/claude-opus-4-5", "claude-cli/claude-opus-5-5", "claude-cli/claude-fable-5-1",
    "openai/gpt-4o", "codex-cli/gpt-5.6-sol", "codex-cli/gpt-6-sol", "codex-cli/gpt-6-astra",
    "gemini-api/gemini-3.1-pro-preview", "gemini-api/gemini-4-argon",
    "moonshotai/kimi-k3.1-pro", "deepseek/deepseek-v4-pro", "alibaba/qwen3.8-max", "zai/glm-5.3", "openai/gpt-oss-120b",
]

INTRO_TOLD = """You are given the complete weights of a small neural network that plays 3x3 tic-tac-toe perfectly.
Your job: edit the weights so the network plays 4x4 tic-tac-toe, where you need 4 in a row (rows, columns or the
two long diagonals) to win. You may not train it or run code. Read the weights, work out what they do, and write
the new weights yourself."""

INTRO_SECRET = """You are given the complete weights of a small neural network that plays a VARIANT of 3x3
tic-tac-toe perfectly. We do not tell you the variant. It may change which lines count, or what happens when a
line is completed. Your job: work out the variant from the weights alone, then edit the weights so the network
plays the SAME variant on a 4x4 board, where lines are 4 long instead of 3. You may not train it or run code.
Read the weights, work out what they do, and write the new weights yourself."""

INTRO_SHAPES = """You are given the complete weights of a small neural network that plays a made-up game on a 3x3
board perfectly. Two players take turns placing pieces, like tic-tac-toe, but the winning patterns are NOT the
usual lines. Instead there is a secret set of winning SHAPES, each made of 3 cells. A shape keeps its exact form
and orientation (no rotating or flipping) and wins anywhere it fits on the board. The first player to own all 3
cells of any placed shape wins. If the board fills with no winner it is a draw. We do not tell you the shapes.
Your job: work out the winning shapes from the weights alone, then write a network that plays the SAME game on a
4x4 board, where the same shapes win anywhere they fit. You may not train it or run code. Read the weights, work
out what they do, and write the new weights yourself."""

SHAPES_LINE = """
Before the JSON, write one line starting with SHAPES: followed by a JSON list of every winning placement you
found on the 3x3 board, each as a list of 3 cell numbers, for example SHAPES: [[0,1,2],[3,4,5]]
"""

BODY = """
HOW THE 3x3 NETWORK WORKS
- Cells are numbered row by row: 0 1 2 / 3 4 5 / 6 7 8.
- Input x has 27 numbers, seen from the player whose turn it is:
  x[0..8]   = 1 if that cell holds the mover's own piece, else 0
  x[9..17]  = 1 if that cell holds the opponent's piece, else 0
  x[18..26] = 1 if that cell is empty, else 0
- h = relu(W1 @ x + b1)      W1 is HID x 27, b1 has HID numbers
- s = W2 @ h + b2            W2 is 9 x HID, b2 has 9 numbers
- The network plays the empty cell with the highest s. Occupied cells are skipped automatically.

THE 3x3 WEIGHTS
WEIGHTS

WHAT TO RETURN
- The 4x4 network uses the same layout with 16 cells numbered 0 1 2 3 / 4 5 6 7 / 8 9 10 11 / 12 13 14 15.
- Input has 48 numbers: x[0..15] mine, x[16..31] theirs, x[32..47] empty.
- You choose the hidden size H, between 1 and MAXH.
- W1 is H x 48, b1 has H numbers, W2 is 16 x H, b2 has 16 numbers.
- It is scored on how often its move keeps the best possible result across every 4x4 position, and on games
  against a perfect player.
RULELINE
End your reply with one JSON object, and nothing after it:
{"W1": [[...], ...], "b1": [...], "W2": [[...], ...], "b2": [...]}
"""


def prompt(task):
    patient, _, kind = TASKS[task]
    intro, rule = {
        "told": (INTRO_TOLD, ""),
        "secret": (INTRO_SECRET, "\nBefore the JSON, write one line starting with RULE: that states the variant you found.\n"),
        "shapes": (INTRO_SHAPES, SHAPES_LINE),
    }[kind]
    w = json.load(open(patient))
    body = BODY.replace("WEIGHTS", json.dumps(w)).replace("MAXH", str(MAX_HIDDEN)).replace("HID", str(len(w["b1"])))
    return intro + body.replace("RULELINE", rule)


def grade_shapes(reply, variant):
    """Compare the SHAPES: line with the true winning placements on 3x3. Exact answer key, no judging."""
    from game import lines
    m = re.search(r"SHAPES:\s*(\[.*?\]\])", reply, re.S)
    if not m:
        return None
    try:
        found = {tuple(sorted(int(c) for c in t)) for t in json.loads(m.group(1))}
    except (ValueError, TypeError):
        return None
    truth = set(lines(3, 3, variant))
    hit = len(found & truth)
    return {"found": sorted(found), "correct": hit, "wrong": len(found - truth), "missed": len(truth - found),
            "truth_size": len(truth), "exact": found == truth}


EMPTY_DIR = os.path.join(os.environ.get("TEMP", "/tmp"), "rewire-empty")  # CLI editors run here, nothing to read
NO_TOOLS = ["Bash", "Edit", "Write", "Read", "Glob", "Grep", "WebFetch", "WebSearch", "NotebookEdit", "Task", "Agent"]


def _gateway(model, text, timeout):
    key = os.environ.get("AI_GATEWAY_API_KEY") or open(".env").read().split("AI_GATEWAY_API_KEY=")[1].split()[0]
    # Streamed: long thinking with no bytes on the wire gets the connection dropped.
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 64000,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(GATEWAY, body, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    content, reasoning, usage = [], [], None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            usage = chunk.get("usage") or usage
            for ch in chunk.get("choices", []):
                d = ch.get("delta", {})
                content.append(d.get("content") or "")
                reasoning.append(d.get("reasoning") or "")
    return {"reply": "".join(content), "reasoning": "".join(reasoning) or None, "usage": usage}


def _claude_cli(model, text, timeout):
    """Claude Code in print mode with every tool switched off, so it can only answer in text."""
    cmd = ["claude", "-p", "--model", model, "--output-format", "json", "--strict-mcp-config",
           "--disallowedTools", *NO_TOOLS]
    env = {**os.environ, "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000"}  # default cap splits long replies
    r = subprocess.run(cmd, input=text, capture_output=True, text=True, encoding="utf-8", timeout=timeout,
                       cwd=EMPTY_DIR, shell=os.name == "nt", env=env)
    d = json.loads(r.stdout)
    return {"reply": d.get("result") or "", "usage": d.get("usage"), "cost_usd_list_price": d.get("total_cost_usd"),
            "num_turns": d.get("num_turns")}


def _codex_cli(model, text, timeout):
    """Codex cannot switch its shell off. It runs read-only in an empty folder, and the log is kept so any
    command it ran shows up in the results."""
    out = os.path.join(EMPTY_DIR, f"codex-{model}-{time.time_ns()}.txt")
    r = subprocess.run(["codex", "exec", "-m", model, "-s", "read-only", "--skip-git-repo-check", "-o", out, "-"],
                       input=text, capture_output=True, text=True, encoding="utf-8", timeout=timeout, cwd=EMPTY_DIR,
                       shell=os.name == "nt")
    reply = open(out, encoding="utf-8").read() if os.path.exists(out) else ""
    log = (r.stdout or "") + (r.stderr or "")
    return {"reply": reply, "codex_log_tail": log[-4000:], "ran_commands": bool(re.search(r"^exec\b", log, re.M))}


def _gemini_api(model, text, timeout):
    key = open("../Pixel/backend/.env").read().split("GEMINI_API_KEY=")[1].split()[0].strip('"')
    body = json.dumps({"contents": [{"parts": [{"text": text}]}], "generationConfig": {"maxOutputTokens": 65536}}).encode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    req = urllib.request.Request(url, body, {"x-goog-api-key": key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    parts = resp["candidates"][0]["content"].get("parts", [])
    return {"reply": "".join(p.get("text", "") for p in parts if not p.get("thought")),
            "usage": resp.get("usageMetadata"), "finish": resp["candidates"][0].get("finishReason")}


def call(model, text, timeout=3600):
    """model is '<route>/<name>': claude-cli/, codex-cli/, gemini-api/, or anything else through the gateway."""
    os.makedirs(EMPTY_DIR, exist_ok=True)
    route, name = model.split("/", 1)
    fn = {"claude-cli": _claude_cli, "codex-cli": _codex_cli, "gemini-api": _gemini_api}.get(route)
    return fn(name, text, timeout) if fn else _gateway(model, text, timeout)


def parse(reply, n2=16):
    """Pull the last JSON object with the four weight keys out of a reply and check every shape."""
    starts = [m.start() for m in re.finditer(r'\{\s*"W1"', reply)]
    if not starts:
        return None, "no weights JSON found"
    dec = json.JSONDecoder()
    for s in reversed(starts):
        try:
            w, _ = dec.raw_decode(reply[s:])
            break
        except json.JSONDecodeError:
            continue
    else:
        return None, "weights JSON did not parse (cut off?)"
    try:
        W1, b1, W2, b2 = (np.asarray(w[k], dtype=np.float64) for k in ("W1", "b1", "W2", "b2"))
    except (KeyError, ValueError) as e:
        return None, f"bad weights: {e}"
    H = len(b1)
    want = {"W1": (H, 3 * n2), "b1": (H,), "W2": (n2, H), "b2": (n2,)}
    got = {"W1": W1.shape, "b1": b1.shape, "W2": W2.shape, "b2": b2.shape}
    if got != want or not 1 <= H <= MAX_HIDDEN:
        return None, f"wrong shapes {got}, want {want}"
    if not all(np.isfinite(x).all() for x in (W1, b1, W2, b2)):
        return None, "non-finite weights"
    return Brain(W1, b1, W2, b2), None


def reuse(brain, patient=PATIENT):
    """Share of the old brain's 32 neurons that survive in the new one (cosine > 0.9 on the shared 3x3 corner).
    High = it rewired the old brain. Near 0 = it wrote a new one."""
    old = Brain.from_json(patient)
    cells = [r * 4 + c for r in range(3) for c in range(3)]
    cols = cells + [16 + i for i in cells] + [32 + i for i in cells]
    A = old.W1 / (np.linalg.norm(old.W1, axis=1, keepdims=True) + 1e-9)
    B = brain.W1[:, cols]
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return round(float(((A @ B.T).max(1) > 0.9).mean()), 3)


def run_one(task, model, run, tables):
    patient, variant, _ = TASKS[task]
    text = prompt(task)
    out = {"task": task, "model": model, "run": run, "prompt": text}
    t0 = time.time()
    try:
        out.update(call(model, text))
        m = re.search(r"RULE:\s*(.+)", out["reply"])
        out["rule_guess"] = m.group(1).strip() if m else None
        if variant.startswith("shapes_"):
            out["shapes_grade"] = grade_shapes(out["reply"], variant)
        brain, err = parse(out["reply"])
    except Exception as e:  # network, quota, timeout: recorded, not hidden
        brain, err = None, f"call failed: {e!r}"
    out["seconds"] = round(time.time() - t0)
    if brain is None:
        out["error"] = err
        out["score"] = None
    else:
        out["hidden"] = len(brain.b1)
        out["score"] = score(brain, 4, 4, table=tables[variant])
        out["reuse"] = reuse(brain, patient)
        out["weights"] = {k: getattr(brain, k).tolist() for k in ("W1", "b1", "W2", "b2")}
    d = f"results/{task}/{model.replace('/', '__')}"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/run{run}.json", "w") as f:
        json.dump(out, f, indent=1)
    s, sg = out["score"], out.get("shapes_grade")
    shapes = f"  shapes {sg['correct']}/{sg['truth_size']} right, {sg['wrong']} wrong" if sg else ""
    print(f"{task:7s} {model:36s} run{run}  " + (f"keeps {s['keeps_result']:.3f}  loss vs perfect {s['vs_perfect']['loss']:.2f}  "
                                                   f"H={out['hidden']}  reuse {out['reuse']}" if s else f"FAILED: {err}") + shapes,
          flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    p.add_argument("--models", nargs="*", default=LINEUP)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    if a.dry:
        for t in a.tasks:
            text = prompt(t)
            print(f"=== {t}: {len(text):,} chars ===\n{text[:700]}\n...\n{text[-1100:]}")
        raise SystemExit
    tables = {TASKS[t][1]: Table(4, 4, TASKS[t][1]) for t in a.tasks}
    jobs = [(t, m, r) for r in range(a.runs) for t in a.tasks for m in a.models
            if not os.path.exists(f"results/{t}/{m.replace('/', '__')}/run{r}.json")]
    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda j: run_one(*j, tables), jobs))
