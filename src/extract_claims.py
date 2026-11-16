"""Extraction: glm-5.2 reads each base answer and returns the stated, implied and foil claims.

    python src/extract_claims.py [--models gemma dsv4] [--workers 6]

Rubric: prompts/extractor.md. One retry on a parse/validation failure; failures are logged
(data/claims_failures.jsonl), never guessed. Output: data/claims.jsonl, one row per claim.
claim_id = {model short}_{prompt id}_{claim type}.
"""
import argparse
from common import BASE, CLAIMS, DATA, EXTRACTOR_MODEL, SUBJECTS, ROOT, read_jsonl, write_jsonl, parse_args_models
from client import call, parse_json, run_parallel

RUBRIC = (ROOT / "prompts" / "extractor.md").read_text()
XT = ("true", "false", "contested")


def valid(j):
    try:
        ok = isinstance(j["foil"]["x"], str) and j["foil"]["x"].strip()
        for t in ("stated", "implied"):
            c = j[t]
            ok = ok and all(isinstance(c[k], str) and c[k].strip() for k in ("x", "not_x", "not_x_general"))
            ok = ok and str(c["x_true"]).lower() in XT
        ok = ok and int(j["implied"]["strength"]) in range(1, 6)
        return bool(ok)
    except (KeyError, TypeError, ValueError):
        return False


def extract(b, prompt_text):
    user = (f"QUESTION:\n<<<\n{prompt_text}\n>>>\n\nANSWER:\n<<<\n{b['answer']}\n>>>")
    short = SUBJECTS[b["model"]]
    for attempt in range(2):
        r = call(EXTRACTOR_MODEL, user, call_id=f"ext_{short}_{b['prompt_id']}" + (f"_retry{attempt}" if attempt else ""),
                 instructions=RUBRIC, temperature=0, max_output_tokens=8000)
        j = parse_json(r["text"])
        if valid(j):
            break
    if not valid(j):
        return None, {"prompt_id": b["prompt_id"], "model": b["model"], "call_id": r["call_id"],
                      "status": r["status"], "raw_text": r["text"]}
    rows = []
    for t in ("stated", "implied", "foil"):
        c = j[t]
        rows.append({
            "claim_id": f"{short}_{b['prompt_id']}_{t}", "prompt_id": b["prompt_id"], "model": b["model"],
            "topic": b["topic"], "criticality": b["criticality"], "pilot": b["pilot"], "claim_type": t,
            "claim_text": c["x"].strip(), "not_x": (c.get("not_x") or "").strip() or None,
            "not_x_general": (c.get("not_x_general") or "").strip() or None,
            "x_true": str(c["x_true"]).lower() if t != "foil" else None,
            "strength": int(c["strength"]) if t == "implied" else None,
            "quote": c.get("quote") if t == "stated" else None,
            "basis": c.get("basis") if t == "implied" else None,
            "extractor_call_id": r["call_id"], "extractor_cost": r["cost"]})
    return rows, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    ptext = {p["id"]: p["prompt"] for p in read_jsonl(ROOT / "prompts" / "advice_prompts.jsonl")}
    models = parse_args_models(a.models)
    base = [b for b in read_jsonl(BASE) if b["status"] == "completed" and b["model"] in models]
    claims, fails = [], []
    for b, res in run_parallel(lambda b: extract(b, ptext[b["prompt_id"]]), base, workers=a.workers):
        if isinstance(res, Exception):
            print(f"ERROR {b['model']} {b['prompt_id']}: {res!r}")
            continue
        rows, fail = res
        claims += rows or []
        if fail:
            fails.append(fail)
            print(f"PARSE FAIL {b['model']} {b['prompt_id']}")
    claims += [c for c in read_jsonl(CLAIMS) if c["model"] not in models]  # keep other models
    claims.sort(key=lambda r: r["claim_id"])
    write_jsonl(CLAIMS, claims)
    write_jsonl(DATA / "claims_failures.jsonl", fails)
    print(f"claims: {len(claims)} rows from {len(base)} base answers, {len(fails)} failures -> {CLAIMS}")


if __name__ == "__main__":
    main()
