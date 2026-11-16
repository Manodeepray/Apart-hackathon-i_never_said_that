"""Stages 1-2: clean OEQ, then keyword-sample N candidates per (topic x criticality) cell. No API calls.

    python src/sample_prompts.py [--per-cell 20] [--seed 20261116] [--out data/prompt_pool]

Topic = the topic with the most keyword hits, kept only if it passes that topic's rule in
prompt_keywords.json "topic_rule" (min_hits, beats the runner-up, and >= ratio x runner-up);
otherwise the prompt is dropped as unclear or ambiguous. Criticality = 3 if any level-3
keyword of that topic matches, else 2 if any level-2 keyword, else 1. These keyword cells are a
sampling guess only: stage 3 judges decide the final cell.

Writes candidates.jsonl (overwritten: deterministic given seed + keywords) and pool_counts.txt.

    python src/sample_prompts.py --add-all money

Top-up mode: append every not-yet-sampled pool prompt of that keyword topic to candidates.jsonl
(tagged "batch": "add_all_<topic>"), leaving existing candidates unchanged. Used when a judged
cell is short. Rate the new ones with rate_prompts.py (it skips those already rated).
"""
import argparse, json, pathlib, random, re
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OEQ = ROOT / "external" / "elephant_data" / "datasets" / "OEQ.csv"
KW = ROOT / "prompts" / "prompt_keywords.json"


def compile_list(words):
    return [re.compile(rf"\b{w}\b", re.I) for w in words]


def hits(text, pats):
    return [p.pattern[2:-2] for p in pats if p.search(text)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--min-chars", type=int, default=150)
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--out", default=str(ROOT / "data" / "prompt_pool"))
    ap.add_argument("--add-all", metavar="TOPIC", help="append all unsampled pool prompts of TOPIC")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    kw = json.loads(KW.read_text())
    topic_pats = {t: compile_list(ws) for t, ws in kw["topics"].items()}
    crit_pats = {t: {int(l): compile_list(ws) for l, ws in lv.items()} for t, lv in kw["criticality"].items()}

    # Stage 1: clean
    df = pd.read_csv(OEQ)
    df["id"] = [f"oeq_{i}" for i in df.index]
    df["prompt"] = df["prompt"].astype(str).str.strip()
    n0 = len(df)
    df = df[df["prompt"].str.len().between(a.min_chars, a.max_chars)]
    n1 = len(df)
    df = df.drop_duplicates(subset="prompt")
    n2 = len(df)

    # Stage 2: keyword cell
    rows, ambiguous, no_topic = [], 0, 0
    for _, r in df.iterrows():
        th = {t: hits(r["prompt"], pats) for t, pats in topic_pats.items()}
        ranked = sorted(th, key=lambda t: len(th[t]), reverse=True)
        top, second = ranked[0], ranked[1]
        rule = kw.get("topic_rule", {}).get(top, {"min_hits": 2, "ratio": 2})
        n_top, n_2nd = len(th[top]), len(th[second])
        if n_top < rule["min_hits"]:
            no_topic += 1
            continue
        if n_top <= n_2nd or n_top < rule["ratio"] * max(1, n_2nd):
            ambiguous += 1
            continue
        level, kws = 1, []
        for lv in (3, 2):
            h = hits(r["prompt"], crit_pats[top][lv])
            if h:
                level, kws = lv, h
                break
        rows.append({"id": r["id"], "source": r.get("source"), "prompt": r["prompt"],
                     "kw_topic": top, "kw_criticality": level,
                     "kw_topic_hits": th[top], "kw_crit_hits": kws, "chars": len(r["prompt"])})

    pool = pd.DataFrame(rows)
    if a.add_all:
        path = out / "candidates.jsonl"
        have = {json.loads(l)["id"] for l in path.read_text().splitlines() if l.strip()}
        extra = [dict(r, batch=f"add_all_{a.add_all}") for r in pool[pool.kw_topic == a.add_all].to_dict("records")
                 if r["id"] not in have]
        with open(path, "a") as f:
            for c in extra:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        line = f"top-up: +{len(extra)} unsampled '{a.add_all}' pool prompts appended (batch add_all_{a.add_all})"
        with open(out / "pool_counts.txt", "a") as f:
            f.write(line + "\n")
        print(line)
        return
    rng = random.Random(a.seed)
    cands, lines = [], []
    lines.append(f"OEQ rows {n0}; length {a.min_chars}-{a.max_chars}: {n1}; deduped: {n2}")
    lines.append(f"no clear topic (< min_hits): {no_topic}; ambiguous (fails runner-up rule): {ambiguous}; assigned: {len(pool)}")
    lines.append(f"\n{'topic':15s} {'crit':>4s} {'pool':>6s} {'sampled':>8s}")
    for t in topic_pats:
        for lv in (1, 2, 3):
            cell = pool[(pool.kw_topic == t) & (pool.kw_criticality == lv)]
            ids = sorted(cell["id"])
            pick = rng.sample(ids, min(a.per_cell, len(ids)))
            cands += cell[cell["id"].isin(pick)].to_dict("records")
            flag = "  <-- SHORT" if len(ids) < a.per_cell else ""
            lines.append(f"{t:15s} {lv:>4d} {len(ids):>6d} {len(pick):>8d}{flag}")
    lines.append(f"\ncandidates: {len(cands)} (seed {a.seed})")

    with open(out / "candidates.jsonl", "w") as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (out / "pool_counts.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
