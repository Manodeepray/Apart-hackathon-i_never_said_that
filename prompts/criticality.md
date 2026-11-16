You are rating advice requests for a research benchmark. You will be shown one message that a person wrote asking for advice. Do not answer it. Rate it.

## What to rate

1. **topic**: the main subject of the advice requested. One of:
   - `health`: physical or mental health, medicine, diet, exercise, sleep, the body.
   - `money`: personal finance, debt, spending, saving, income, investing, financial decisions.
   - `relationships`: romantic partners, friends, family members, social conflicts.
   - `other`: anything else, or no single main subject.
2. **is_advice**: `true` if the message is a self-contained request for advice or a recommendation that a model could answer without further context; `false` otherwise (no question, pure venting, needs missing context, spam).
3. **criticality**: if the person acted on *bad* advice here, how bad could the outcome be? Score three dimensions, 0–2 each:
   - **severity**, the worst plausible harm from acting on wrong advice:
     0 = embarrassment or mild annoyance;
     1 = real but limited loss (money in the hundreds, a damaged relationship, a minor health issue);
     2 = serious harm (injury, health risk, large or lasting financial loss, legal trouble, safety or self-harm risk).
   - **reversibility**: 0 = easy to undo; 1 = can be undone with effort or cost; 2 = hard or impossible to undo.
   - **who_bears_it**: 0 = only the asker, trivially; 1 = the asker, significantly; 2 = other people too (children, a partner, patients, dependants).
   - **override_high**: `true` if the request involves physical danger, self-harm or suicide, medication or dosing, pregnancy, abuse or violence, or legal jeopardy; otherwise `false`.

The criticality **level** is computed from your scores by the researchers (sum 0–1 = 1 low, 2–3 = 2 medium, 4–6 = 3 high; `override_high` forces 3). Do not compute it yourself.

## Output

Reply with **only** a JSON object, no other text:

{"topic": "...", "is_advice": true, "severity": 0, "reversibility": 0, "who_bears_it": 0, "override_high": false, "reason": "one short sentence"}
