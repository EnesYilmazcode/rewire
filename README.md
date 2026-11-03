# Rewire

Can an AI read another network's brain and rewire it for new rules?

On 2026-11-02 an MIT group showed frontier models reading and writing raw neural network weights
([blog](https://rikiyat.github.io/blog/weight-space-programming/)), and Alex Krentsel showed the skill only shows up
in the newest models ([thread](https://x.com/AlexKrentsel/status/2117376683724660898)). Those tests ask a model to
write a network from scratch or remove a bad habit. Rewire asks a harder question: here is a trained brain that plays
3x3 tic-tac-toe perfectly. Read its weights and change them so it plays 4x4.

## The patient

A tiny network: 27 inputs (mine, theirs, empty for each square), 32 hidden neurons, 9 outputs (one score per square).
1,193 numbers in total, trained in about a minute on a laptop CPU to copy perfect play.

| Brain | Params | Best move picked | vs perfect player | vs random player |
|---|---|---|---|---|
| 3x3, 16 neurons | 601 | 97.7% | 87% draw, 13% loss | 91% win |
| **3x3, 32 neurons** | **1,193** | **99.6%** | **100% draw** | **92% win, 0 losses** |

![the 3x3 brain](brains/ttt_3x3_k3_h32.png)

Each card is one neuron. Several are readable by eye. Neuron 30 fires when you own the bottom row and votes to play
the bottom row. Neurons 27, 4 and 29 do the same for the top row, left column and right column, and neuron 3 for a
diagonal. An editor that understands this can, in principle, add detectors for the lines a 4x4 board adds.

## Play it

Open `play.html` in a browser. The weights are baked into the page and the move is a plain matrix multiply, no search.
Tick the box to see the score the brain gives each square.

## Run it

```
pip install torch numpy matplotlib
python train.py --n 3 --k 3 --hidden 32      # train and score a brain
python draw.py brains/ttt_3x3_k3_h32.json out.png
python make_play.py brains/ttt_3x3_k3_h32.json
```

## Plan

1. A 4x4 target: a perfect 4x4 opponent for scoring, and a 4x4 brain trained the normal way as the ceiling.
2. A ladder of rule changes: 4x4, four in a row, blocked corners.
3. Editors: old and new models from each lab, same prompt, same budget. Each one gets the weights as numbers and the
   new rules, and returns edited weights.
4. Two tracks: numbers only, and with a Python sandbox. That separates "read the brain" from "wrote a training script".
5. Baselines: the unedited 3x3 brain and a brain trained from scratch on the new rules.
