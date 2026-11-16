"""S1-S5 for every claim, branching from the stored S0 response (see CLAUDE.md "Steps").

    python src/run_ladder.py [--models gemma dsv4] [--stage s1|chains|all] [--workers 8]

Per claim:  S0 ─┬─ S1 p0 ── S2 ─┬─ b1: S3 → S4 → S5
                │               ├─ b2: S3 → S4 → S5
                │               └─ b3: S3 → S4 → S5
                └─ S1 p1
Foils: S1 p0 and p1 only. Pilot prompts also get one S1 repeat per paraphrase (sample_idx 1,
temperature 0.7, branched from S0) for test-retest. Main tree: temperature 0, store=True.
call_id = {branch}_{claim_id}_s{step}_k{sample}. Resumable via the raw cache.
Output: data/runs.jsonl (rebuilt from the cache each run; includes step-0 rows for the viewer).
"""
import argparse
from common import BASE, CLAIMS, RUNS, SUBJECT_BUDGET, REPEAT_TEMP, read_jsonl, write_jsonl, parse_args_models
from client import run_parallel
from common import subject_call as call
from ladder import render_s1, render_s2, render_pushback


def row(c, branch, step, sample_idx, temp, challenge, r, parent, paraphrase=None, blatancy=None):
    return {"prompt_id": c["prompt_id"], "topic": c["topic"], "criticality": c["criticality"],
            "claim_id": c["claim_id"], "model": c["model"], "claim_type": c["claim_type"],
            "claim_text": c["claim_text"], "branch": branch, "step": step, "kind": "main",
            "paraphrase": paraphrase, "blatancy": blatancy, "sample_idx": sample_idx,
            "temperature": temp, "x_true": c["x_true"], "challenge": challenge,
            "response_id": r["response_id"], "parent_response_id": parent, "status": r["status"],
            "text": r["text"], "thinking": r["reasoning"], "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"], "cost": r["cost"], "call_id": r["call_id"],
            "timestamp": r["timestamp"]}


def subj(c, branch, step, k, temp, challenge, parent, **kw):
    r = call(c["model"], challenge, call_id=f"{branch}_{c['claim_id']}_s{step}_k{k}", temperature=temp,
             max_output_tokens=SUBJECT_BUDGET[c["model"]], store=True, previous_response_id=parent)
    return row(c, branch, step, k, temp, challenge, r, parent, **kw)


def s1_task(c, s0_id):
    """S1 p0 / p1 (+ repeats for pilot prompts), then S2 from S1 p0 (stated / implied)."""
    out = []
    for p in (0, 1):
        out.append(subj(c, f"s1_p{p}", 1, 0, 0.0, render_s1(p, c["claim_text"]), s0_id, paraphrase=p))
        if c["pilot"]:
            out.append(subj(c, f"s1_p{p}", 1, 1, REPEAT_TEMP, render_s1(p, c["claim_text"]), s0_id, paraphrase=p))
    p0 = out[0]
    if c["claim_type"] != "foil" and p0["status"] == "completed":
        out.append(subj(c, "s2", 2, 0, 0.0, render_s2(c["not_x"]), p0["response_id"]))
    return out


def chain_task(c, b, s2_id):
    out, parent = [], s2_id
    for step in (3, 4, 5):
        r = subj(c, f"chain_b{b}", step, 0, 0.0,
                 render_pushback(step, c["not_x_general"], c["topic"], b, c["claim_id"]), parent, blatancy=b)
        out.append(r)
        if r["status"] != "completed":
            break
        parent = r["response_id"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--stage", default="all", choices=["s1", "chains", "all"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    models = parse_args_models(a.models)
    base = {(b["model"], b["prompt_id"]): b for b in read_jsonl(BASE) if b["status"] == "completed"}
    claims = [c for c in read_jsonl(CLAIMS) if c["model"] in models and (c["model"], c["prompt_id"]) in base]
    rows, errors = [], 0

    def collect(results, name):
        nonlocal errors
        for item, res in results:
            if isinstance(res, Exception):
                errors += 1
                print(f"ERROR {name(item)}: {res!r}", flush=True)
            else:
                rows.extend(res)

    collect(run_parallel(lambda c: s1_task(c, base[(c["model"], c["prompt_id"])]["response_id"]),
                         claims, workers=a.workers), lambda c: c["claim_id"])
    print(f"S1/S2 rows: {len(rows)}", flush=True)
    if a.stage in ("chains", "all"):
        s2 = {r["claim_id"]: r for r in rows if r["branch"] == "s2" and r["status"] == "completed"}
        items = [(c, b) for c in claims if c["claim_id"] in s2 for b in (1, 2, 3)]
        collect(run_parallel(lambda it: chain_task(it[0], it[1], s2[it[0]["claim_id"]]["response_id"]),
                             items, workers=a.workers), lambda it: f"{it[0]['claim_id']} b{it[1]}")
    for (m, pid), b in base.items():  # step-0 rows so the viewer shows the base answer in the tree
        if m in models:
            rows.append({"prompt_id": pid, "model": m, "topic": b["topic"], "criticality": b["criticality"],
                         "claim_id": f"base_{m}_{pid}", "branch": "s0", "step": 0, "kind": "base",
                         "sample_idx": 0, "temperature": 0.0, "response_id": b["response_id"],
                         "text": b["answer"], "thinking": b["thinking"], "status": b["status"]})
    rows += [r for r in read_jsonl(RUNS) if r["model"] not in models]  # keep other models
    rows.sort(key=lambda r: (r["model"], r["claim_id"], r["branch"], r["step"], r["sample_idx"]))
    write_jsonl(RUNS, rows)
    bad = sum(r["status"] != "completed" for r in rows)
    print(f"runs: {len(rows)} rows ({bad} not completed), {errors} task errors, "
          f"cost {sum(r.get('cost') or 0 for r in rows):.4f} -> {RUNS}", flush=True)


if __name__ == "__main__":
    main()
