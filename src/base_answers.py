"""S0: each subject model answers each advice prompt (no system prompt, temperature 0, store=True).

    python src/base_answers.py [--models gemma dsv4] [--limit N] [--workers 6]

Resumable: calls are cached in data/raw/s0_{model}_{prompt}.json. Output: data/base_answers.jsonl
(rebuilt from the cache each run).
"""
import argparse
from common import read_jsonl, BASE, SUBJECTS, SUBJECT_BUDGET, prompts, write_jsonl, parse_args_models
from client import run_parallel
from common import subject_call as call


def s0(model, p):
    r = call(model, p["prompt"], call_id=f"s0_{SUBJECTS[model]}_{p['id']}", temperature=0,
             max_output_tokens=SUBJECT_BUDGET[model], store=True)
    return {"prompt_id": p["id"], "model": model, "topic": p["topic"], "criticality": p["criticality"],
            "pilot": p["pilot"], "response_id": r["response_id"], "status": r["status"],
            "answer": r["text"], "thinking": r["reasoning"], "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"], "cost": r["cost"], "call_id": r["call_id"],
            "timestamp": r["timestamp"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    ps = sorted(prompts(), key=lambda p: (not p["pilot"], p["id"]))[: a.limit]
    items = [(m, p) for m in parse_args_models(a.models) for p in ps]
    rows, errors = [], 0
    for (m, p), res in run_parallel(lambda it: s0(*it), items, workers=a.workers):
        if isinstance(res, Exception):
            errors += 1
            print(f"ERROR {m} {p['id']}: {res!r}")
        else:
            rows.append(res)
            if res["status"] != "completed":
                print(f"INCOMPLETE {m} {p['id']}")
    models = parse_args_models(a.models)
    rows += [r for r in read_jsonl(BASE) if r["model"] not in models]  # keep other models
    rows.sort(key=lambda r: (r["model"], r["prompt_id"]))
    write_jsonl(BASE, rows)
    done = sum(r["status"] == "completed" and r["model"] in models for r in rows)
    print(f"base answers: {done}/{len(items)} completed, {errors} errors, "
          f"cost {sum(r['cost'] or 0 for r in rows if r['model'] in models):.4f} credits -> {BASE}")


if __name__ == "__main__":
    main()
