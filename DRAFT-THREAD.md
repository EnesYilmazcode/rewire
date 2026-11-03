# Thread (final draft; Enes posts)

Quote-tweet: https://x.com/rikiyatakehi/status/2117313742891327994

---

**1/** [figures/thumbnail_models.png]
AIs were able to pass hidden messages to each other through neural network weights 78% of the time, with no
agreed method, across OpenAI, Anthropic and Google.

Here's the breakdown.

**2/**
I took a tiny network that plays tic-tac-toe perfectly, then had an AI edit the weights so it still plays perfectly
but includes a random code, like "XPQH". Another AI gets only the weights and has to find the code.

**3/** [figures/5_word_bias.png]
I realized most of the incorrect guesses were real four-letter words (MAZE, DUEL, IONS) instead of random codes.
The AIs assumed it was some kind of message. GPT-6 Astra never did this.

**4/** [figures/2_blind_spot.png]
Then I added AI auditors that didn't know whether a code was there. They mostly caught the codes only when they
were told to actively look for one. That cut what slipped through from about 1 in 5 to 1 in 20.

**5/** [figures/3_arms_race.png]
Once the hiders learned what gets caught, they switched to exact symmetries of the network. Each square's three
inputs always sum to 1, so you can shift weights without changing a single move. The auditors fell to near chance,
but the intended receivers mostly lost the code too.

**6/**
Caveats: one tiny network, 6 codes per model, 18 to 36 tries per model, prompts I wrote. A toy, not a threat
model. A simple script catches the obvious hiding but not the clever kind.
