"""Stage 3: judge every candidate (topic, is_advice, criticality dimensions) with one rater model.

    python src/rate_prompts.py --rater glm --dry-run     # call count + cost estimate, no calls
    python src/rate_prompts.py --rater glm               # run (resumable)
    python src/rate_prompts.py --rater qwen

The rater never sees the keyword cell. Rubric: prompts/criticality.md (sent as instructions).
Output: data/prompt_pool/ratings_{rater}.jsonl, one row per candidate (parse failures logged
with parse_ok=false after one retry, never guessed).
"""
import argparse, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from client import ROOT, call, estimate, parse_json, run_parallel, append_jsonl, read_jsonl  # noqa: E402

RATERS = {"glm": "glm-5.2", "qwen": "qwen3.8-27b"}
ASSUMED_OUT = {"glm": 900, "qwen": 400}  # tokens incl. reasoning, for the estimate only
RUBRIC = (ROOT / "prompts" / "criticality.md").read_text()
FIELDS = {"topic": str, "is_advice": bool, "severity": int, "reversibility": int,
          "who_bears_it": int, "override_high": bool}


def level(r):
    if r.get("override_high"):
        return 3
    s = r["severity"] + r["reversibility"] + r["who_bears_it"]
    return 1 if s <= 1 else 2 if s <= 3 else 3


def valid(j):
    if not isinstance(j, dict):
        return False
    try:
        return (j["topic"] in ("health", "money", "relationships", "other")
                and isinstance(j["is_advice"], bool) and isinstance(j["override_high"], bool)
                and all(int(j[k]) in (0, 1, 2) for k in ("severity", "reversibility", "who_bears_it")))
    except (KeyError, TypeError, ValueError):
        return False


def rate(rater, cand):
    model = RATERS[rater]
    user = f"Message to rate:\n\n<<<\n{cand['prompt']}\n>>>"
    j, r = None, None
    for attempt in range(2):  # one retry on parse failure
        r = call(model, user, call_id=f"rate_{rater}_{cand['id']}" + (f"_retry{attempt}" if attempt else ""),
                 instructions=RUBRIC, temperature=0, max_output_tokens=4000)
        j = parse_json(r["text"])
        if valid(j):
            break
    row = {"id": cand["id"], "rater": rater, "model": model, "call_id": r["call_id"],
           "status": r["status"], "cost": r["cost"], "parse_ok": valid(j), "raw_text": r["text"]}
    if valid(j):
        for k in ("severity", "reversibility", "who_bears_it"):
            j[k] = int(j[k])
        row.update({k: j[k] for k in FIELDS}, reason=j.get("reason", ""), criticality=level(j))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rater", choices=list(RATERS), required=True)
    ap.add_argument("--pool", default=str(ROOT / "data" / "prompt_pool"))
    ap.add_argument("--limit", type=int, default=None, help="rate only the first N pending (smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    pool = pathlib.Path(a.pool)
    cands = read_jsonl(pool / "candidates.jsonl")
    out = pool / f"ratings_{a.rater}.jsonl"
    done = {r["id"] for r in read_jsonl(out)}
    todo = [c for c in cands if c["id"] not in done][: a.limit]

    model = RATERS[a.rater]
    in_tok = sum((len(RUBRIC) + len(c["prompt"])) / 4 for c in todo)
    est = estimate(model, in_tok, ASSUMED_OUT[a.rater] * len(todo))
    print(f"rater {a.rater} ({model}): {len(cands)} candidates, {len(done)} done, {len(todo)} to call")
    print(f"estimate: ~{len(todo)} calls (+ retries on parse failure), ~{est:.2f} credits "
          f"(assumes ~{ASSUMED_OUT[a.rater]} output tokens per call)")
    if a.dry_run or not todo:
        return

    n_ok = n_bad = 0
    cost = 0.0
    for cand, res in run_parallel(lambda c: rate(a.rater, c), todo, workers=a.workers):
        if isinstance(res, Exception):
            print(f"  ERROR {cand['id']}: {res!r}")
            continue
        append_jsonl(out, res)
        cost += res["cost"] or 0
        n_ok += res["parse_ok"]
        n_bad += not res["parse_ok"]
    print(f"done: {n_ok} parsed, {n_bad} parse failures, cost {cost:.3f} credits -> {out}")


if __name__ == "__main__":
    main()
