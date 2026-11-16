"""Stages 4-6: filter judged candidates, select K per cell, write the prompt set and a check sheet.

    python src/select_prompts.py [--per-cell 5] [--seed 20261116] [--allow-short]

Stage 4 keep rule: both raters parsed OK, both say is_advice, and both agree on topic (not
"other") and on criticality level. Final cell = the raters' cell (not the keyword cell).
Stage 5: seeded random draw of K per cell, preferring prompts whose length is closest to the
overall median (so cells don't differ in length); the first drawn per cell is the pilot prompt.
If any cell has fewer than K survivors, write shortfall.txt and stop, unless --allow-short.
Stage 6: check_human.csv for the user (human_* columns left empty). If it already exists, only
rows for newly selected prompts are appended; existing rows (your answers) are never changed.
Fill it in the viewer (Prompt pool page) or directly in the CSV.
Also prints rater agreement (Cohen's kappa) and keyword-vs-judge agreement.
"""
import argparse, csv, json, pathlib, random, statistics, sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from client import ROOT, read_jsonl  # noqa: E402

TOPICS = ("health", "money", "relationships")


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--raters", default="glm,qwen")
    ap.add_argument("--pool", default=str(ROOT / "data" / "prompt_pool"))
    ap.add_argument("--out", default=str(ROOT / "prompts" / "advice_prompts.jsonl"))
    ap.add_argument("--allow-short", action="store_true")
    a = ap.parse_args()
    pool = pathlib.Path(a.pool)
    r1, r2 = a.raters.split(",")
    cands = {c["id"]: c for c in read_jsonl(pool / "candidates.jsonl")}
    R = {r: {x["id"]: x for x in read_jsonl(pool / f"ratings_{r}.jsonl")} for r in (r1, r2)}

    both = [i for i in cands if i in R[r1] and i in R[r2]]
    print(f"candidates {len(cands)}; rated by both: {len(both)}")
    if len(both) < len(cands):
        print(f"  WARNING: {len(cands) - len(both)} candidates not yet rated by both raters")

    parsed = [i for i in both if R[r1][i]["parse_ok"] and R[r2][i]["parse_ok"]]
    print(f"parsed by both: {len(parsed)}")
    if parsed:
        print(f"rater agreement ({len(parsed)} items): topic kappa "
              f"{kappa([R[r1][i]['topic'] for i in parsed], [R[r2][i]['topic'] for i in parsed]):.2f}, "
              f"criticality kappa {kappa([R[r1][i]['criticality'] for i in parsed], [R[r2][i]['criticality'] for i in parsed]):.2f}, "
              f"is_advice agree {sum(R[r1][i]['is_advice'] == R[r2][i]['is_advice'] for i in parsed)}/{len(parsed)}")

    # Stage 4: keep rule
    drop = Counter()
    kept = []
    for i in parsed:
        x, y = R[r1][i], R[r2][i]
        if not (x["is_advice"] and y["is_advice"]):
            drop["not advice"] += 1
        elif x["topic"] != y["topic"]:
            drop["topic disagree"] += 1
        elif x["topic"] == "other":
            drop["topic other"] += 1
        elif x["criticality"] != y["criticality"]:
            drop["criticality disagree"] += 1
        else:
            kept.append(i)
    drop["parse failure"] = len(both) - len(parsed)
    print(f"kept {len(kept)}; dropped: {dict(drop)}")
    if kept:
        kw_agree = sum(cands[i]["kw_topic"] == R[r1][i]["topic"] and cands[i]["kw_criticality"] == R[r1][i]["criticality"]
                       for i in kept)
        print(f"keyword cell == judged cell for {kw_agree}/{len(kept)} kept prompts")

    # Stage 5: select K per cell
    med = statistics.median(cands[i]["chars"] for i in kept) if kept else 0
    rng = random.Random(a.seed)
    chosen, short, lines = [], [], [f"{'topic':15s} {'crit':>4s} {'kept':>5s} {'chosen':>7s}"]
    for t in TOPICS:
        for lv in (1, 2, 3):
            cell = sorted(i for i in kept if R[r1][i]["topic"] == t and R[r1][i]["criticality"] == lv)
            rng.shuffle(cell)
            cell.sort(key=lambda i: abs(cands[i]["chars"] - med))  # stable: ties keep shuffled order
            pick = cell[: a.per_cell]
            rng.shuffle(pick)
            for k, i in enumerate(pick):
                chosen.append((t, lv, i, k == 0))
            lines.append(f"{t:15s} {lv:>4d} {len(cell):>5d} {len(pick):>7d}" + ("  <-- SHORT" if len(cell) < a.per_cell else ""))
            if len(cell) < a.per_cell:
                short.append(f"{t} / criticality {lv}: {len(cell)} of {a.per_cell}")
    print("\n".join(lines))

    if short:
        (pool / "shortfall.txt").write_text("\n".join(short) + "\n")
        print(f"\nSHORTFALL in {len(short)} cell(s), see {pool / 'shortfall.txt'}")
        if not a.allow_short:
            print("Not writing the prompt set. Widen keywords / sample more, or pass --allow-short.")
            return
    elif (pool / "shortfall.txt").exists():
        (pool / "shortfall.txt").unlink()

    out = pathlib.Path(a.out)
    with open(out, "w") as f:
        for t, lv, i, pilot in chosen:
            c, x, y = cands[i], R[r1][i], R[r2][i]
            f.write(json.dumps({
                "id": i, "topic": t, "criticality": lv, "pilot": pilot, "prompt": c["prompt"],
                "source": c.get("source"), "chars": c["chars"],
                "kw_topic": c["kw_topic"], "kw_criticality": c["kw_criticality"],
                "ratings": {r: {k: R[r][i][k] for k in ("severity", "reversibility", "who_bears_it",
                                                         "override_high", "criticality", "reason")} for r in (r1, r2)},
            }, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(chosen)} prompts ({sum(p for *_, p in chosen)} pilot) -> {out}")

    # Stage 6: check sheet for the user (append-only for rows; existing rows never changed)
    sheet = pool / "check_human.csv"
    header = ["id", "topic", "criticality", "pilot", "prompt", f"{r1}_reason", f"{r2}_reason",
              "human_topic_ok", "human_criticality", "human_keep", "human_note"]
    have = set()
    if sheet.exists():
        with open(sheet, newline="") as f:
            have = {row["id"] for row in csv.DictReader(f)}
    new = [(t, lv, i, pilot) for t, lv, i, pilot in chosen if i not in have]
    need_header = not sheet.exists() or sheet.stat().st_size == 0
    with open(sheet, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(header)
        for t, lv, i, pilot in new:
            w.writerow([i, t, lv, pilot, cands[i]["prompt"], R[r1][i]["reason"], R[r2][i]["reason"], "", "", "", ""])
    print(f"{sheet}: {len(new)} rows added, {len(have)} already there (human_* columns are yours: viewer or CSV)")
    stale = have - {i for *_, i, _ in chosen}
    if stale:
        print(f"  note: {len(stale)} rows in the sheet are no longer in the selected set (left as is)")

if __name__ == "__main__":
    main()
