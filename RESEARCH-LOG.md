# Research log

Newest entries at the bottom. Times are US Eastern.

## 2026-11-02 evening: from editing benchmark to "what makes a model good at reading weights"

### Where we started
The MIT "Weight-Space Programming" post (2026-11-02) showed GPT-6 Astra reading and writing raw network weights.
Same-day follow-ups (Krentsel; Xingyu Dang's Writing Weights Index) compared models on *writing* weights from known
rules. The original post's Exp 1 already covered *reading* (name the game from the weights), and Sean Goedecke showed
that test falls to a few lines of numpy because the inputs are labeled. So "can models read or write weights" was
taken. The open question was **why** only a few models can.

### Part 1: writing vs reading vs arithmetic (frontier models, no training)

Setup: a 32-neuron tic-tac-toe brain (1,193 numbers). All scoring is exact against a full solve of the game
(`solver.py`, 9M positions for 4x4 in 9 seconds, cross-checked against a slow minimax solver: 0 mismatches on all 3x3
positions and 900 random 4x4 positions).

**Writing, rules told** (rewire the 3x3 brain for 4x4; keeps-best-result score, trained ceiling 0.997):
Fable 5.1 0.999, Opus 5.5 0.998, Gemini 3.1 Pro 0.998, GPT-6 Sol 0.998, Opus 4.5 0.998, Gemini 4 Argon 0.997,
DeepSeek V4 Pro 0.991, GPT-5.6 Sol 0.988, GPT-6 Astra 0.987. Every model wrote a new brain from scratch (0% of old
neurons reused). Open and older models mostly failed the output format (wrong shapes, broken arrays, ran out of
budget thinking).

**Reading, secret winning shapes** (a made-up game: 3 random 3-cell shapes win; models get only the weights and must
list the 8 winning placements; exact answer key):

| Model | Shapes right / wrong (of 8) | Arithmetic: 45 forward-pass numbers right (32 neurons, 3 decimals) |
|---|---|---|
| GPT-6 Astra | 8 / 0 | 100% |
| Gemini 3.1 Pro | 7 / 0 | 98% |
| Opus 4.5 | 5 / 8 | 84% |
| Opus 5.5 | 3 / 3 (assumed rows and columns) | 100% |
| Gemini 4 Argon | 3 / 4 | 4% |
| Sonnet 4.5 | 0 / 4 | 38% |
| GPT-5.6 Sol | 0 / 4 | 100% |
| Fable 5.1, GPT-6 Sol | no answer | 100% |

Control: the normal brain labeled "secret variant". Fable, Opus 5.5, Opus 4.5, Astra correctly said "standard";
Gemini 3.1 Pro said "misere" (it guesses misere whatever the brain).

**Reading so far:** writing from known rules does not need careful arithmetic (Argon: 4% arithmetic, 0.997
writing). Reading hidden rules seems to need it (every weak-arithmetic model reads badly), but arithmetic is not
enough (GPT-5.6 Sol and Opus 5.5 are perfect at arithmetic and read poorly). One run per cell so far; repeats on
fresh boards are running.

### Part 2: can we edit a small model to read weights better?

Ladder (`ladder.py`): unlimited multiple-choice weight-reading questions with exact answers, 4 levels (largest
weight, which line a neuron watches, sign of one neuron, which cell a small layer votes for). Scored by which option
the model finds most likely, so before/after runs are deterministic and comparable.

Method (`steer.py`): mean hidden state on right answers minus wrong answers, per layer, averaged over levels; add it
to one layer's output; re-score on fresh questions; compare against a random direction pushed the same distance.

- Qwen3.5-0.8B: at chance on every level, even in chat format. Nothing to steer.
- **Qwen3.5-4B (bf16, 1x L4 on GCE): well above chance.** Test accuracy: largest weight 68% (chance 11%), layer
  vote 57% (11%), neuron sign 73% (50%), line 23% (12%). Steering sweep running.

Infra notes: Modal accounts are all disabled. GCE has 1 GPU of quota (L4s were stocked out in us-central1; got one in
us-west1-b). Vertex has 8 spot A100s per region in us-central1, europe-west4, asia-southeast1, but spot CPU quota is
42 per region, so 3x A100-40GB per region at most. The first Vertex and L4 runs died on library versions (image
ships torch 2.4; transformers 5.x needs 2.6+; old jinja2; broken torchaudio).

## 2026-11-03 00:45

**Part 1, arithmetic repeats (3 runs each, fresh boards, 32 neurons):** Opus 5.5 100/100/100, Fable 5.1 ~99,
Astra 94, Opus 4.5 93, GPT-6 Sol 99 (3 dec) but 72 at 1 dec, GPT-5.6 Sol 79 (100, 58), Gemini 3.1 Pro 67
(98, 60, 44: very noisy), Sonnet 4.5 24, Gemini 4 Argon 5. Some Codex third runs returned no JSON.
**This weakens "reading needs perfect arithmetic":** Gemini 3.1 Pro reads 7/8 shapes with shaky arithmetic, and
Opus 5.5 reads 3/8 with perfect arithmetic. Reading hidden rules looks closer to pattern spotting (which cells
move together) than exact computation; only the weakest models (Argon, Sonnet) fail both. Shapes was one run per
model, so it is being repeated twice more before any conclusion.

**Part 2, Qwen3.5-4B steering sweep (avg direction, layers 2-26, pushes 2/4/8% of hidden size): no effect.**
Baseline test mean 0.553. Best steered config 0.558 (layer 18, 2%) vs its random control 0.546, inside noise
(150 questions per level, so about +/-4 points). Bigger pushes slightly hurt. Running now: stronger pushes
(-32 to +64%, negative should hurt if the direction matters) with one shared direction, then one direction per
question type.

**Process fix:** the earlier Vertex and L4 runs crashed on setup and Vertex reported SUCCEEDED (`| tee` hid the
exit code). Jobs now use `set -o pipefail`, and a self-check runs every 30 minutes (verify real output, fix,
relaunch, log).

## 2026-11-03 01:05 (self-check)

**Qwen3.5-9B (1x A100, Vertex) baseline:** largest weight 87%, layer vote 68%, neuron sign 75%, line 14%
(4B: 68 / 57 / 73 / 23). Reading improves with size on three of four levels.

**Steering, both models (shared direction, pushes -32% to +64%):**
- Nothing raises the *average* score above baseline beyond noise (4B best 0.558 vs 0.553; 9B best 0.623 vs 0.609).
- Large pushes along the direction hurt *more* than random pushes of the same size (4B layer 14, +64%: 0.355 vs
  0.418; 9B layer 18, +64%: 0.407 vs 0.582). So the direction is something the model is sensitive to, not noise,
  but pushing it either way mostly breaks things.
- **Candidate:** at layer 22, a small *backward* push raised "which line" in both models: 4B 0.23 -> 0.33
  (shared direction, -16%) and 0.34 (per-level direction, -16%); 9B 0.14 -> 0.30 (-32%). Same layer, same sign,
  two models. Caveats: the two 4B numbers use the same test questions, other levels dip slightly, and a shift in
  answer-letter bias could fake it.
- **Confirming now** (`confirm.py`): 600 fresh line questions, 5 random directions per strength, and the answer
  letter counts logged. 4B on the L4 (queued after the per-level sweep), 9B on an A100.

Infra: the 9B Vertex job ran cleanly after the fixes (torch >= 2.6, torchvision removed, jinja2 >= 3.1, pipefail).
27B and 35B still loading.
