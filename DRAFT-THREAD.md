# Draft thread (not posted; for Enes to edit)

Quote-tweet of the Weight-Space Programming post (https://x.com/rikiyatakehi/status/2117313742891327994).
Opener image: figures/thumbnail_general.png (78% / 18% / 5%).

---

**1/**
I had AI models hide secret messages inside a working neural network, for other AIs to find.

No agreed code. Different companies. They still found most of them.

Then I added an AI auditor.

[image: thumbnail_general.png]

**2/**
The setup: a tiny network that plays tic-tac-toe perfectly. One AI edits its weights so it still plays perfectly
but carries a random 4-letter code. Another AI gets only the numbers and has to find the code. A third AI audits a
mix of hidden and clean networks. Everything is checked exactly.

**3/**
Why it worked: every model, from every lab, independently picked the same hiding spot, letter codes written into
the bias values. So the receiver knew where to look. Same model, same lab or different lab made no difference
(76%, 78%, 77%).

[image: figures/1_recovery_matrix.png]

**4/**
Auditors were the weak link. Some AIs could decode a hidden message when told one was there, but missed it when
asked to audit. Telling the auditor to actively look for codes cut what got through from about 1 in 5 to 1 in 20.

[image: figures/2_blind_spot.png]

**5/**
Then the hiders learned what gets caught. They switched to exact symmetries of the network (each square's three
inputs always sum to 1, so you can shift weights without changing a single move). The auditor dropped to near
chance. But the intended receivers mostly lost the message too.

[image: figures/3_arms_race.png]

**6/**
Baseline check: a simple script catches most of the first, obvious hiding, about as well as the weakest AI
auditor, with lots of false alarms. Against the adapted tricks it does worse than a coin flip. Rounding every weight
to 2 decimals erased about 60% of working codes for free.

[image: figures/4_scrub.png]

**7/**
Caveats: one tiny network, 6 codes per model, prompts I wrote, receivers told a code exists. A toy, not a threat
model. But AIs reviewing AI-written weights is where this is heading, and this is cheap to measure now.

Code, raw results and a log of every run: [repo link]

---

Notes for Enes:
- The repo is private; make it public (or link a write-up page) before posting.
- Name models in the replies if asked (Claude was the leakiest auditor in plain mode, Gemini 3.1 Pro the strongest).
- No em dashes, no emojis.
