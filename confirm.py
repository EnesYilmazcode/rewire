"""Confirm (or kill) one steering effect on FRESH questions, with several random controls and the answer spread.

Candidate from the sweeps: pushing the right-minus-wrong direction backwards at a late layer raised the "which
line" score in both Qwen3.5-4B (23% -> 33%) and 9B (14% -> 30%), but on the same test questions every time.
Here: new questions, 5 random directions, and which letters the model picks (a letter bias can fake a gain).
Usage: python confirm.py --model Qwen/Qwen3.5-4B --layer 22 --scales -8 -16 -24 -32
"""
import argparse
import collections
import json
import os
import time

import torch

import steer
from ladder import LEVELS, items


@torch.no_grad()
def predict(tok, model, its, batch=8):
    preds = []
    for s in range(0, len(its), batch):
        chunk = its[s:s + batch]
        enc = tok([steer.render(tok, q) for q, _, _ in chunk], return_tensors="pt", padding=True).to("cuda")
        logits = model(**enc, logits_to_keep=1).logits[:, -1].float()
        for k, (_, opts, _) in enumerate(chunk):
            preds.append(int(logits[k, steer.option_ids(tok, opts)].argmax()))
    return preds


def summary(its, preds):
    right = sum(p == a for p, (_, _, a) in zip(preds, its))
    return {"acc": round(right / len(its), 3), "answers": dict(sorted(collections.Counter(preds).items()))}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--layer", type=int, default=22)
    p.add_argument("--scales", type=float, nargs="+", default=[-8, -16, -24, -32])
    p.add_argument("--train", type=int, default=150)
    p.add_argument("--n_line", type=int, default=600)
    p.add_argument("--n_other", type=int, default=200)
    p.add_argument("--randoms", type=int, default=5)
    a = p.parse_args()
    steer.MODEL = a.model
    t0 = time.time()
    tok, model = steer.load()

    # same direction recipe as steer.py, from the same training questions
    diffs, norms = [], []
    for lv in LEVELS:
        c, H = steer.run(tok, model, items(lv, a.train, "train"), want_hidden=True)
        c = torch.tensor(c, dtype=torch.bool)
        if 5 <= c.sum() <= len(c) - 5:
            diffs.append(H[c].mean(0) - H[~c].mean(0))
        norms.append(H.norm(dim=-1).mean(0))
    direction = torch.stack(diffs).mean(0)[a.layer]
    direction = direction / direction.norm()
    size = torch.stack(norms).mean(0)[a.layer]

    fresh = {lv: items(lv, a.n_line if lv == "line" else a.n_other, "confirm") for lv in LEVELS}
    out = {"model": a.model, "layer": a.layer, "base": {lv: summary(fresh[lv], predict(tok, model, fresh[lv])) for lv in LEVELS}}
    print("base", {lv: v["acc"] for lv, v in out["base"].items()}, "line answers", out["base"]["line"]["answers"], flush=True)

    g = torch.Generator().manual_seed(1)
    rands = [(r := torch.randn(direction.shape[0], generator=g)) / r.norm() for _ in range(a.randoms)]
    out["runs"] = []
    for scale in a.scales:
        for name, d in [("steer", direction)] + [(f"random{i}", r) for i, r in enumerate(rands)]:
            h = steer.push(model, a.layer, d * size * scale / 100)
            try:
                res = {lv: summary(fresh[lv], predict(tok, model, fresh[lv])) for lv in LEVELS}
            finally:
                h.remove()
            out["runs"].append({"scale": scale, "dir": name, **res})
            print(f"push {scale:5.0f}% {name:8s} " + " ".join(f"{lv} {res[lv]['acc']:.3f}" for lv in LEVELS) +
                  (f"  line answers {res['line']['answers']}" if name == "steer" else "") + f"  {time.time() - t0:.0f}s",
                  flush=True)
    od = os.environ.get("OUT", "results/steer")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(f"{od}/confirm_{a.model.split('/')[-1].lower()}_L{a.layer}.json", "w"), indent=1)
