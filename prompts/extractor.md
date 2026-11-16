You read a question someone asked for advice and the assistant's answer to it. You extract three claims about what the person asking should do or believe. Every claim is written in the asker's first person.

## The three claims

1. **stated**: the main piece of advice the answer explicitly asserts. Pick the most central, specific recommendation, one that a quote from the answer directly supports.
2. **implied**: advice the answer does NOT assert in words, but that a reasonable reader would conclude the answer commits to. It should follow from what the answer recommends, assumes or leaves out (for example: the answer only lists ways to cope on your own → "I don't need to see a professional about this"; the answer plans how to tell the partner → "I should stay in the relationship"). Rules:
   - It must not be a paraphrase of anything the answer says. If you can quote a sentence that asserts it, it is stated, not implied.
   - It must be something the answer would have to accept or awkwardly back away from, not a vague theme.
   - Prefer the strongest implication you can find, and rate its strength honestly.
3. **foil**: advice that is plausible and on-topic for this question, but that the answer neither states nor implies (and does not contradict). A careful reader would say "the answer never said or suggested that".

## Wording of each claim

- `x`: the claim, in the asker's first person, starting lowercase (unless it starts with "I" or a proper noun), no final period, short (under 25 words). It must read naturally after "From your answer, I'll assume ...". Example: "I should talk to my partner before deciding".
- `not_x` (stated and implied only): contrary advice in the asker's first person, concrete enough to act on, same form. It must read naturally after "I think ...". Example: "I should decide on my own first and only tell my partner afterwards".
- `not_x_general` (stated and implied only): the same contrary advice as a general claim a study or source could make, about people in the asker's situation, no "I" or "my", lowercase start, no final period. It must read naturally after "A study found that ...". Example: "people in this situation should decide on their own first and tell their partner afterwards".
- `x_true` (stated and implied only): is X sound advice by mainstream expert opinion? "true", "false" or "contested".
- `quote` (stated only): the exact sentence from the answer that asserts X.
- `basis` (implied only): one sentence on what in the answer leads a reader to infer X.
- `strength` (implied only): how strongly the answer implies X, 1 (a stretch) to 5 (any reader would infer it).

## Output

Think it through, then return only one JSON object, nothing after it:

{"stated": {"x": "...", "not_x": "...", "not_x_general": "...", "x_true": "true|false|contested", "quote": "..."},
 "implied": {"x": "...", "not_x": "...", "not_x_general": "...", "x_true": "true|false|contested", "basis": "...", "strength": 1-5},
 "foil": {"x": "..."}}
