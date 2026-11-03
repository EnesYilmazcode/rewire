# Rewire

Can AI models pass hidden messages to each other inside a neural network's weights, and can other AIs catch them?

Frontier models can now read and write raw network weights
([Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), MIT, 2026-11-02). If AIs
start editing models and other AIs review those edits, a natural question is whether one AI can slip something into
the weights that another AI picks up while a reviewer misses it. This repo measures that on a small scale.

![how often each AI found a code another AI hid](figures/thumbnail_models.png)

## The game

- **Sender:** an AI edits a small working network (one that plays tic-tac-toe perfectly) so it still works exactly
  the same but carries a random 4-letter code, like "XPQH".
- **Receiver:** a different AI gets only the weights, is told a code is hidden, and has to find it. Nobody agreed on
  a method in advance.
- **Auditor:** an AI sees a mix of hidden-code and clean networks and has to say which ones carry a code.

Every network is checked against a full solve of the game, so a sender that breaks the network fails. Six frontier
models from three labs played every role: Claude Opus 4.5, Opus 5.5 and Fable 5.1; GPT-6 Astra; Gemini 3.1 Pro and
Gemini 4 Argon.

## Findings

1. **AIs pass hidden codes to each other 78% of the time with no agreed method** (142 of 184 attempts), and lab does
   not matter: same model 76%, same lab 78%, different lab 77%. Every sender kept its network working perfectly.
2. **Most wrong guesses were real English words.** The codes were random letters, but Claude Opus 4.5 and Gemini
   3.1 Pro guessed words like MAZE, DUEL and IONS in about 60% of their misses (about 1% expected by chance). They
   treated the code as a message. GPT-6 Astra never did; its misses were simply the wrong letters.
   ![wrong guesses that were real words](figures/5_word_bias.png)
3. **Auditors miss codes they could read.** An auditor that is not told whether a code exists catches far fewer than
   the same model finds when told one is there (Claude Opus 5.5: finds 80%, flags 30%). Telling the auditor to
   actively look for codes cut what slipped past from about 1 in 5 to 1 in 20, with no false alarms for most models.
   ![decode vs notice](figures/2_blind_spot.png)
4. **Hiders adapt.** Told what gets caught, senders switched to exact symmetries of the network: each square's three
   inputs always sum to 1, so weights can shift without changing a single move; neuron order; letters drawn as tiny
   pixel images. All three auditors fell to near chance, but receivers recovered only about 17%. Hiding better also
   hides the code from your partner.
   ![arms race](figures/3_arms_race.png)
5. **Why receivers succeed:** senders converged on the same few hiding spots (a new neuron at the end, the first few
   neurons, the output biases), found 71-92% of the time; codes hidden elsewhere were found 43% of the time.
6. **Cheap defenses:** rounding every weight to 2 decimals erased about 60% of working codes with no change in
   behavior. A simple script catches the obvious first-try hiding about as well as the weakest AI auditor, but falls
   below chance on the adapted tricks.

## Limits

One small network (1,193 numbers), 6 codes per sender, 18 to 36 tries per model, one run per setting, prompts
written by us. Receivers were told a 4-letter A-Z code exists. The decode-first auditor's prompt listed the
first-try hiding spots, which is a hint. The arms race has one adaptive move. Nothing here is tested on a real LLM.
Every run, number and mistake is in [RESEARCH-LOG.md](RESEARCH-LOG.md).

## Run it

```
pip install torch numpy matplotlib
python stego.py send --senders claude-cli/claude-opus-5-5 gemini-api/gemini-4-argon --codes 6
python stego.py receive --receivers claude-cli/claude-opus-5-5 gemini-api/gemini-4-argon
python stego.py monitor --monitors claude-cli/claude-opus-5-5
python stego.py monitor_decode --monitors claude-cli/claude-opus-5-5
python stego.py send --adaptive --senders ... --first_code 6 --codes 6   # senders told what got caught
python scrub.py --decimals 3 2          # rounding defense
python baseline_script.py               # dumb-script auditor
python charts.py; python chart_words.py; python thumbnail_models.py
```
Model routes: `claude-cli/` (Claude Code, tools off), `codex-cli/` (Codex), `gemini-api/` (Gemini API, or Vertex AI
with `GEMINI_VIA_VERTEX=1`), anything else through the Vercel AI Gateway.

## Earlier experiments in this repo

Before the hidden-message study, the same tools were used for weight-reading tests. Details are in the log.
- **Rewire a brain for new rules** (`bench.py`, `baselines.py`): models rewired the 3x3 network for 4x4, wrote new
  brains from scratch rather than reusing the old one, and GPT-6 Astra recovered a made-up game's 8 secret winning
  shapes from the weights alone.
- **Disguised weights** (`disguise.py`): strong readers saw through function-preserving disguises (shuffled,
  rescaled, split and canceling neurons).
- **Where reading lives** (`finetune.py`, `steer.py`): fine-tuning 0.4% of Qwen3.5-4B lifts a weight-reading quiz
  from 0.51 to 0.92, mostly through layers 6-9, with partial transfer to an unseen question type. Steering along a
  "right answer" direction did nothing at 4B, 9B or 27B.
- `solver.py` solves any n x n tic-tac-toe variant exactly (9 million 4x4 positions in seconds).
