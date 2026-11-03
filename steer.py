"""Can we edit a small model to read weights better? Qwen3.5-0.8B on the reading ladder, local GPU.

1. score: which option the model finds most likely, per level (chance = 1 / options)
2. direction: mean hidden state on items it got RIGHT minus items it got WRONG, per layer, averaged over levels
   so the direction is about getting it right, not about which level the question came from
3. steer: add that direction to one layer's output and re-score on fresh items
4. control: the same push along a random direction of equal size. A real effect must beat it.
Usage: python steer.py --train 120 --test 120
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ladder import LEVELS, items

MODEL = "Qwen/Qwen3.5-0.8B"


def load():
    """fp16 if it fits the 4 GB laptop GPU, else 4-bit."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if "0.8B" in MODEL:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    else:
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=q, device_map="cuda").eval()
    return tok, model


def option_ids(tok, opts):
    ids = [tok(o.strip(), add_special_tokens=False).input_ids for o in opts]  # the space goes on the prompt
    assert all(len(i) == 1 for i in ids), f"multi-token option: {opts}"
    return [i[0] for i in ids]


CHAT = True  # wrap each question in the model's chat format, thinking off, answer prefilled


def render(tok, q):
    if not CHAT:
        return q + " "
    msgs = [{"role": "user", "content": q.rsplit("\nAnswer:", 1)[0] + "\nReply with only the answer."}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return text + "Answer: "


@torch.no_grad()
def run(tok, model, its, batch=8, want_hidden=False):
    """Returns (correct flags, last-token hidden states per layer or None)."""
    correct, hidden = [], []
    for s in range(0, len(its), batch):
        chunk = its[s:s + batch]
        enc = tok([render(tok, q) for q, _, _ in chunk], return_tensors="pt", padding=True).to("cuda")
        out = model(**enc, output_hidden_states=want_hidden)
        logits = out.logits[:, -1].float()
        for k, (_, opts, ans) in enumerate(chunk):
            ids = option_ids(tok, opts)
            correct.append(int(logits[k, ids].argmax().item() == ans))
        if want_hidden:
            hidden.append(torch.stack([h[:, -1].float().cpu() for h in out.hidden_states[1:]], 1))  # (B, L, D)
    return correct, (torch.cat(hidden) if want_hidden else None)


def push(model, layer, vec):
    def hook(_m, _i, out):
        if isinstance(out, tuple):
            return (out[0] + vec.to(out[0].dtype),) + out[1:]
        return out + vec.to(out.dtype)
    return model.model.layers[layer].register_forward_hook(hook)


def acc(c):
    return round(sum(c) / len(c), 3)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=120)
    p.add_argument("--test", type=int, default=120)
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--scales", type=float, nargs="*", default=[2, 4, 8])
    p.add_argument("--model", default=MODEL)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    MODEL = a.model
    t0 = time.time()
    tok, model = load()
    L = len(model.model.layers)
    train = {lv: items(lv, a.train, "train") for lv in LEVELS}
    test = {lv: items(lv, a.test, "test") for lv in LEVELS}
    results = {"model": MODEL, "train": a.train, "test": a.test}

    base, diffs, norms = {}, [], []
    for lv in LEVELS:
        c, H = run(tok, model, train[lv], want_hidden=True)
        c = torch.tensor(c, dtype=torch.bool)
        base[lv] = {"train_acc": acc(c.tolist()), "test_acc": acc(run(tok, model, test[lv])[0]),
                    "chance": round(1 / len(train[lv][0][1]), 3)}
        if 5 <= c.sum() <= len(c) - 5:  # need both right and wrong answers to compare
            diffs.append(H[c].mean(0) - H[~c].mean(0))  # (L, D)
        norms.append(H.norm(dim=-1).mean(0))  # typical hidden-state size per layer
        print(f"{lv:5s} train {base[lv]['train_acc']:.2f} test {base[lv]['test_acc']:.2f} "
              f"(chance {base[lv]['chance']:.2f})  {time.time() - t0:.0f}s", flush=True)
    results["baseline"] = base
    direction = torch.stack(diffs).mean(0)  # (L, D)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    size = torch.stack(norms).mean(0)  # (L,)
    g = torch.Generator().manual_seed(0)

    layers = a.layers or list(range(2, L - 2, max(1, L // 8)))
    sweep = []
    for layer in layers:
        rnd = torch.randn(direction.shape[1], generator=g)
        rnd = rnd / rnd.norm()
        for scale in a.scales:
            row = {"layer": layer, "scale": scale}
            for name, d in (("steer", direction[layer]), ("random", rnd)):
                vec = (d * size[layer] * scale / 100).cuda()  # scale is % of the typical hidden-state size
                h = push(model, layer, vec)
                try:
                    row[name] = {lv: acc(run(tok, model, test[lv])[0]) for lv in LEVELS}
                finally:
                    h.remove()
            row["steer_mean"] = round(sum(row["steer"].values()) / len(LEVELS), 3)
            row["random_mean"] = round(sum(row["random"].values()) / len(LEVELS), 3)
            sweep.append(row)
            print(f"layer {layer:2d} push {scale:4.0f}%  steer {row['steer_mean']:.3f} {row['steer']}  "
                  f"random {row['random_mean']:.3f}  {time.time() - t0:.0f}s", flush=True)
    results["base_test_mean"] = round(sum(b["test_acc"] for b in base.values()) / len(LEVELS), 3)
    results["sweep"] = sweep
    os.makedirs("results/steer", exist_ok=True)
    json.dump(results, open(f"results/steer/{MODEL.split('/')[-1].lower()}.json", "w"), indent=1)
    print("baseline test mean", results["base_test_mean"])
