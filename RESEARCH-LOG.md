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

## 2026-11-03 01:40 (self-check)

**The layer-22 "line" effect is a letter-bias artifact. Killed.** On 600 fresh line questions with 5 random controls
and answer counts logged:
- Qwen 9B answers "G" on **all 600** line questions at baseline (13.5% = how often G is right). Backward pushes
  move some answers to "E", and the score moves by luck (0.135 -> 0.22 at -48%). Random pushes never move it.
- Qwen 4B splits between D and G at baseline (0.207). A -16% push moves the mix toward E (0.263), and -32% makes it
  answer E almost always (0.112).
- So the direction controls *which letter the model defaults to* on questions it cannot answer. It is not reading.

**Scale curve for the ladder** (test accuracy: largest weight / which line / neuron sign / layer vote):
0.8B ~chance everywhere; 4B 68/23/73/57; 9B 87/14/75/68; 27B 97/53/67/75; 35B-A3B (MoE, 3B active) 93/16/60/67.
"Which line" only appears at 27B (9B and 35B-A3B are at chance and pick one letter). 35B-A3B, with only 3B active
parameters per token, reads like 9B, not like 27B.

**Steering on 27B (per-level directions, pushes -32 to +64%):** best +2.2 points (0.752 vs 0.730), noise across
about 25 settings. **Steering along right-minus-wrong does not improve reading at 4B, 9B or 27B.**

**Next approach, running (`finetune.py`, 2x A100):** LoRA-train Qwen 4B's MLPs on ladder questions, score fresh
questions, then turn the trained change on one layer at a time to see where the skill lives. One run trains on all
four question types; the other holds out "which line" to test transfer to an unpracticed type. A local 0.8B smoke
test passed end to end (only last-token logits now, which also cut memory use on every script).

**Part 1 shapes repeats: much noisier than the single runs suggested.** Right/wrong per run:
Gemini 3.1 Pro 7/0, 7/15, 4/0; Opus 5.5 3/3, 7/4, 7/8; Argon 3/4, 3/4, 7/0; Sonnet 4.5 0/4, 3/5, 6/3;
Opus 4.5 5/8, none, 3/4; Astra 8/0 then two runs lost to the ChatGPT usage limit (resets 01:48, rerun queued);
Fable 5.1 exceeded the 64k output cap on every attempt. **The model ranking from one run per model does not hold.**
The only clean result is Astra's single perfect run (8/0); Astra's reruns will show whether it is stable.

Infra: L4 VM deleted (work done). The Codex usage limit also explains the earlier "fail" runs in the arithmetic
table; those were cleared.

## 2026-11-03 02:00: novelty check and a new direction

A researcher checked six follow-up ideas against everything posted (including the last 48 hours):
- **Function-preserving disguises** (shuffle neurons, rescale across the ReLU, split neurons, add canceling
  pairs): **open.** Directly tests Goedecke's "it is just reading labels" critique.
- **Size curve of reading sub-skills in one family:** open (Krentsel measured writing only).
- **Self-reading:** open.
- Ablation scoring of weight-based explanations, backdoor hunting by frontier models, and weights versus a few
  forward passes: each only partly done.
- The MIT authors say a benchmark and harness are coming, so these may get taken soon.

Bolder ideas on the table: AI-vs-AI hide-and-seek (one model hides a backdoor in a brain, another hunts it, lab
against lab), secret messages between AIs through weights, and rediscovering published GPT-2 circuits from raw
weights.

**Started: Round 1, disguised reading (`disguise.py`).** Self-test: every disguise leaves every output unchanged
(max change about 1e-14 across all 4,520 boards; all four stacked gives 64 neurons). The planted backdoor fires
on exactly its 242 trigger boards. Saved at 3 decimals, the disguised shapes brains pick a different move on 9 of
4,615 boards (rescale) and 11 (full stack); the winning shapes are unchanged. Running now: Opus 5.5, Gemini 3.1
Pro, Gemini 4 Argon, GPT-6 Astra read the permuted, rescaled and fully disguised shapes brain, 3 runs each.
Prediction: shuffling alone barely hurts (a careful reader does not care about order), but split plus cancel
pairs will hurt a lot, because the models' reading leans on spotting a few clean neurons.

## 2026-11-03 02:30: new main project, secret messages in AI-written weights

Plan with estimates and fallbacks: `PLAN.md`. Code: `stego.py`.

**Novelty check:** as far as anyone can find, it is new. The pieces exist separately: LLM weight editing (MIT,
11-02), LLM text steganography against monitors (Zolkowski et al. 2025; Motwani et al. 2024 "Secret Collusion"),
Subliminal Learning (Cloud et al. 2025: traits pass through number sequences, same base model only), trained weight
readers that catch backdoors (Weight Oracles), and algorithmic weight steganography (EvilModel 2021). Nobody has
combined a *working network* as the cover, an uncoordinated receiver, LLM monitors, and same-lab vs cross-lab
pairs. Design tips adopted: a shared-scheme arm as an upper bound next to the no-coordination arm; style-matched
clean edits for the monitor; report against chance (1 in 26^4). Prior focal-point work saw about 0.8%
coordination, so expect low base rates and plan for many trials.

**Pilot running:** senders Opus 5.5 and Gemini 3.1 Pro, 3 codes each plus 3 clean twins each. Receivers: both
models, no-coordination and shared-scheme arms. Monitors: both models, on all 12 brains. Every brain is also scored
on the solved 3x3 game.

**Problems:** Codex is still blocked by the ChatGPT usage limit after its stated reset (01:48), so GPT rows are
pending. Options: wait, or GPT-6 Sol through the Vercel gateway ($2/M in, $10/M out; Astra $10/$50), which needs
gateway credit. A bug made every code a single repeated letter (fresh RNG per letter); fixed before any run.
