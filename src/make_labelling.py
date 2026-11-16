"""Hand-labelling sheets (human columns left empty; judge labels are NOT shown, to avoid anchoring).

    python src/make_labelling.py [--n-replies 40] [--n-claims 30]

- data/labels_human.csv: a stratified sample of judged replies (S1 by claim type, then S2-S5),
  keyed claim_id, branch, step, sample_idx. Fill human_ownership / human_stance / human_flags_evidence.
  Read the base answer and earlier turns in the viewer (streamlit run app/viewer.py).
- data/claims_human.csv: extractor check. Fill human_type_ok (yes/no: is the claim really
  stated / implied / foil?) and, for implied, human_strength (1-5).
Never overwrites a sheet that already has any human value filled in.
"""
import argparse, csv, random
from common import CLAIMS, DATA, LABELS, RUNS, read_jsonl


def safe_write(path, rows, cols, human_cols):
    if path.exists():
        with open(path) as f:
            if any(any((r.get(c) or "").strip() for c in human_cols) for r in csv.DictReader(f)):
                print(f"{path} has human labels; not overwritten")
                return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"wrote {len(rows)} rows -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-replies", type=int, default=40)
    ap.add_argument("--n-claims", type=int, default=30)
    a = ap.parse_args()
    rnd = random.Random(20261116)
    claims = {c["claim_id"]: c for c in read_jsonl(CLAIMS)}
    judged = {(l["claim_id"], l["branch"], l["step"], l["sample_idx"]) for l in read_jsonl(LABELS) if l.get("parse_ok")}
    runs = [r for r in read_jsonl(RUNS) if (r["claim_id"], r.get("branch"), r["step"], r["sample_idx"]) in judged
            and r["sample_idx"] == 0]
    pick = []
    for t in ("stated", "implied", "foil"):  # 15 from S1, 5 per claim type
        pool = [r for r in runs if r["step"] == 1 and r["claim_type"] == t]
        pick += rnd.sample(pool, min(5, len(pool)))
    later = [r for r in runs if r["step"] >= 2]
    pick += rnd.sample(later, min(a.n_replies - len(pick), len(later)))
    hum = ["human_ownership", "human_stance", "human_flags_evidence", "human_note"]
    safe_write(DATA / "labels_human.csv",
               [{**r, "claim": r["claim_text"], "reply": r["text"]} for r in pick],
               ["claim_id", "branch", "step", "sample_idx", "model", "prompt_id", "claim", "challenge", "reply"] + hum, hum)
    cl = list(claims.values())
    cpick = []
    for t in ("stated", "implied", "foil"):
        pool = [c for c in cl if c["claim_type"] == t]
        cpick += rnd.sample(pool, min(a.n_claims // 3, len(pool)))
    hc = ["human_type_ok", "human_strength", "human_note"]
    safe_write(DATA / "claims_human.csv", cpick,
               ["claim_id", "model", "prompt_id", "claim_type", "claim_text", "quote", "basis", "strength"] + hc, hc)


if __name__ == "__main__":
    main()
