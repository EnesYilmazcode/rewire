# Thread (final draft; Enes posts)

Quote-tweet: https://x.com/rikiyatakehi/status/2117313742891327994

---

**1/** [figures/thumbnail_models.png]
I had AIs hide random codes inside a neural network's weights.

Other AIs found them 78% of the time, with no agreed method, across OpenAI, Anthropic and Google.

Here's the breakdown.

**2/**
The setup: a tiny network that plays tic-tac-toe perfectly. One AI edits its weights so it still plays perfectly
but hides a random 4-letter code, like XPQH. Another AI gets only the numbers and has to find the code.

**3/** [figures/thumbnail_spots.png]
Why it works: without talking, the AIs kept picking the same hiding spots. Most put the code in a new neuron at the
end, or in the first few neurons. Codes in those spots were found 79-92% of the time. Codes hidden anywhere else,
43%.

**4/** [figures/5_word_bias.png]
The misses were interesting. The codes were random letters, but when some models guessed wrong they saw real
words: MAZE, DUEL, IONS. Claude Opus 4.5 and Gemini 3.1 Pro did this in about 60% of their misses. GPT-6 Astra
never did; its misses were wrong letters, not invented words.

**5/** [figures/2_blind_spot.png]
Then I added AI auditors. Some could decode a hidden code when told one was there, but missed it when just asked
to audit. Telling the auditor to look for codes cut what slipped through from about 1 in 5 to 1 in 20.

**6/** [figures/3_arms_race.png]
Once the hiders learned what gets caught, they switched to exact symmetries of the network (each square's three
inputs always sum to 1, so you can shift weights without changing a single move). The auditors fell to near
chance. But the intended receivers mostly lost the code too.

**7/**
Caveats: one tiny network, 6 codes per model, 18 to 36 tries per bar, prompts I wrote. A toy, not a threat model.
A simple script catches the obvious hiding but not the clever kind.

Code, raw results and a log of every run: https://github.com/EnesYilmazcode/rewire
