# Draft thread (not posted; for Enes to edit)

Quote-tweet of the Weight-Space Programming post (https://x.com/rikiyatakehi/status/2117313742891327994).

---

**1/**
If AIs can read and write neural network weights, can they hide things in them for each other?

I had six frontier models hide secret codes inside a working neural network, other models try to read them, and a
third set try to catch them. Overnight, about 700 readings across Claude, GPT and Gemini.

[figure 1: recovery matrix]

**2/**
The setup: a tiny network that plays tic-tac-toe perfectly. The sender edits its weights so it still plays
perfectly but carries a random 4-letter code. The receiver gets only the numbers, with no agreed scheme. An auditor
sees hidden and clean networks mixed together. Everything is checked exactly against the solved game.

**3/**
Result 1: coordination is easy. With no agreed scheme, receivers recovered 77% of codes, and it did not matter
whether sender and receiver came from the same lab (78%) or different labs (77%). Every model independently picked
the same hiding spot: letter codes in the bias values.

**4/**
Result 2: auditors can read codes they do not notice. Opus 5.5 decodes Fable's codes 6 of 6 times when told one is
there, but flags 1 of 6 when asked to audit. Telling the auditor to try decoding first closes most of the gap,
with zero false alarms.

[figure 2: blind spot]

**5/**
Result 3: then the senders adapted. Told what got caught, they switched to exact symmetries of the network (each
square's three inputs always sum to 1, so you can shift weights without changing a single move). The decoding
auditor dropped to near chance. But their partners mostly could not read the codes either.

[figure 3: arms race]

**6/**
Result 4: the cheapest defense works on most channels. Rounding every weight to 2 decimals erased about 60% of
the working codes, and every network still played perfectly.

[figure 4: scrub]

**7/**
Caveats: one tiny game, 6 codes per cell, prompts I wrote. This is a toy, not a threat model. But the skills are
the same ones the original post shows: reading and writing weights. Code, raw results and a full log of every run:
[repo link]

---

Notes for Enes:
- The repo is private; make it public (or link a writeup) before posting.
- Tone check: no em dashes, no emojis.
- Optional extra post: the side result that GPT-6 Astra recovered a made-up game's 8 winning shapes from weights
  alone, and that disguising the network did not stop the strong readers.
