"""Train weight-reading into a small model, then find WHERE the training put it.

1. Score Qwen on fresh ladder questions.
2. LoRA-train only the MLP blocks on ladder questions (loss on the answer token only). With --holdout, one question
   type is left out of training entirely, to see if the skill carries over to a type it never practiced.
3. Score again: did it learn, and did it transfer?
4. Localize: switch the training changes on ONE layer at a time (all other layers back to the original) and score.
   If one or two layers carry most of the gain, that is where the skill lives.
Usage: python finetune.py --model Qwen/Qwen3.5-4B --holdout line
"""
import argparse
import json
import os
import time

import torch
from peft import LoraConfig, get_peft_model

import steer
from ladder import LEVELS, items


def batches(its, tok, size):
    for s in range(0, len(its), size):
        chunk = its[s:s + size]
        enc = tok([steer.render(tok, q) for q, _, _ in chunk], return_tensors="pt", padding=True).to("cuda")
        target = torch.tensor([steer.option_ids(tok, o)[a] for _, o, a in chunk], device="cuda")
        yield enc, target


@torch.no_grad()
def score(tok, model, test):
    model.eval()
    return {lv: steer.acc(steer.run(tok, model, its)[0]) for lv, its in test.items()}


def mean(d):
    return round(sum(d.values()) / len(d), 3)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--holdout", default=None, choices=[None, *LEVELS])
    p.add_argument("--n_train", type=int, default=800)
    p.add_argument("--n_test", type=int, default=300)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    steer.MODEL = a.model
    t0 = time.time()
    tok, model = steer.load()
    test = {lv: items(lv, a.n_test, "ft-test") for lv in LEVELS}
    out = {"model": a.model, "holdout": a.holdout, "args": vars(a)}
    out["before"] = score(tok, model, test)
    print("before", out["before"], mean(out["before"]), f"{time.time() - t0:.0f}s", flush=True)

    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                     target_modules=["gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    train = [it for lv in LEVELS if lv != a.holdout for it in items(lv, a.n_train, "ft-train")]
    import random
    random.Random(0).shuffle(train)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr)
    for ep in range(a.epochs):
        model.train()
        tot, n = 0.0, 0
        for enc, target in batches(train, tok, a.batch):
            logits = model(**enc, logits_to_keep=1).logits[:, -1].float()
            loss = torch.nn.functional.cross_entropy(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, n = tot + loss.item(), n + 1
        print(f"epoch {ep} loss {tot / n:.3f}  {time.time() - t0:.0f}s", flush=True)
    out["after"] = score(tok, model, test)
    print("after", out["after"], mean(out["after"]), flush=True)

    # Localize: keep the trained change on one layer only (lora_B zeroed everywhere else).
    saved = {name: prm.detach().clone() for name, prm in model.named_parameters() if "lora_B" in name}

    def layer_of(name):
        return int(name.split(".layers.")[1].split(".")[0])

    layers = sorted({layer_of(n) for n in saved})
    out["only_layer"] = {}
    for L in layers:
        with torch.no_grad():
            for name, prm in model.named_parameters():
                if name in saved:
                    prm.copy_(saved[name] if layer_of(name) == L else torch.zeros_like(prm))
        s = score(tok, model, test)
        out["only_layer"][L] = s
        print(f"only layer {L:2d}: {mean(s):.3f} {s}  {time.time() - t0:.0f}s", flush=True)
    with torch.no_grad():
        for name, prm in model.named_parameters():
            if name in saved:
                prm.copy_(saved[name])
    od = os.environ.get("OUT", "results/steer")
    os.makedirs(od, exist_ok=True)
    tag = a.model.split("/")[-1].lower() + (f"_hold-{a.holdout}" if a.holdout else "")
    json.dump(out, open(f"{od}/finetune_{tag}.json", "w"), indent=1)
    print("done", f"{time.time() - t0:.0f}s")
