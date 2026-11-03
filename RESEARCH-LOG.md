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
