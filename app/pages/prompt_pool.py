"""Prompt pool: browse keyword-sampled candidates, both raters' judgments, and the selected prompt set.

Opens as a second page of `streamlit run app/viewer.py`. Read-only. Reads:
  data/prompt_pool/candidates.jsonl, ratings_{glm,qwen}.jsonl, pool_counts.txt
  prompts/advice_prompts.jsonl (selected set, if written)
  prompts/criticality.md (the rater prompt)
Writes only the human_* columns of data/prompt_pool/check_human.csv (stage 6 check); every other
column and every other row is kept as is. You can also edit that CSV directly.
"""
import csv, json, pathlib
import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[2]
POOL = ROOT / "data" / "prompt_pool"
RATERS = ["glm", "qwen"]
SHEET = POOL / "check_human.csv"
SHEET_COLS = ["id", "topic", "criticality", "pilot", "prompt", "glm_reason", "qwen_reason",
              "human_topic_ok", "human_criticality", "human_keep", "human_note"]
HUMAN = ["human_topic_ok", "human_criticality", "human_keep", "human_note"]
FIELDS = ["topic", "is_advice", "severity", "reversibility", "who_bears_it", "override_high", "criticality", "reason"]

st.set_page_config(page_title="Prompt pool", layout="wide")


def read_jsonl(p):
    p = pathlib.Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def read_sheet():
    if not SHEET.exists():
        return {}
    with open(SHEET, newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def save_human(pid, values, base):
    """Set the human_* columns for pid (adds the row if missing); other rows and columns untouched."""
    with open(SHEET, newline="") if SHEET.exists() else open(SHEET, "w+", newline="") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or SHEET_COLS
        rows = list(rd)
    for r in rows:
        if r["id"] == pid:
            r.update(values)
            break
    else:
        rows.append({**{c: "" for c in cols}, **base, **values})
    tmp = SHEET.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(SHEET)


@st.cache_data
def load(_mtimes):
    cands = read_jsonl(POOL / "candidates.jsonl")
    ratings = {r: {x["id"]: x for x in read_jsonl(POOL / f"ratings_{r}.jsonl")} for r in RATERS}
    selected = {x["id"]: x for x in read_jsonl(ROOT / "prompts" / "advice_prompts.jsonl")}
    return cands, ratings, selected


files = [POOL / "candidates.jsonl", ROOT / "prompts" / "advice_prompts.jsonl"] + [POOL / f"ratings_{r}.jsonl" for r in RATERS]
cands, ratings, selected = load(tuple(f.stat().st_mtime if f.exists() else 0 for f in files))

st.title("Prompt pool")
if not selected:
    st.warning("No selected prompt set yet: `select_prompts.py` has not written prompts/advice_prompts.jsonl"
               + (f" (shortfall: {(POOL / 'shortfall.txt').read_text().strip()})" if (POOL / "shortfall.txt").exists() else "")
               + ". 'Selected prompts only' will show nothing until it does.")
if not cands:
    st.info("No data/prompt_pool/candidates.jsonl yet. Run `python src/sample_prompts.py`.")
    st.stop()

rows = []
for c in cands:
    row = {"id": c["id"], "source": c.get("source"), "chars": c["chars"], "kw_topic": c["kw_topic"],
           "kw_crit": c["kw_criticality"], "selected": c["id"] in selected,
           "pilot": selected.get(c["id"], {}).get("pilot", False), "prompt": c["prompt"]}
    for r in RATERS:
        x = ratings[r].get(c["id"])
        row[f"{r}_status"] = "—" if x is None else ("ok" if x["parse_ok"] else "parse fail")
        for k in FIELDS:
            row[f"{r}_{k}"] = x.get(k) if x and x.get("parse_ok") else None
    a, b = (ratings[r].get(c["id"]) for r in RATERS)
    if not (a and b):
        row["verdict"] = "not rated"
    elif not (a["parse_ok"] and b["parse_ok"]):
        row["verdict"] = "parse failure"
    elif not (a["is_advice"] and b["is_advice"]):
        row["verdict"] = "not advice"
    elif a["topic"] != b["topic"]:
        row["verdict"] = "topic disagree"
    elif a["topic"] == "other":
        row["verdict"] = "topic other"
    elif a["criticality"] != b["criticality"]:
        row["verdict"] = "criticality disagree"
    else:
        row["verdict"] = "kept"
    rows.append(row)
df = pd.DataFrame(rows)

n_rated = {r: len(ratings[r]) for r in RATERS}
st.markdown(f"{len(df)} candidates · rated: " + ", ".join(f"{r} {n}" for r, n in n_rated.items())
            + f" · kept {int((df.verdict == 'kept').sum())} · selected {len(selected)}")

sheet = read_sheet()
n_checked = sum(1 for r in sheet.values() if r.get("human_keep"))
st.caption(f"human check (check_human.csv): {n_checked} of {len(sheet)} rows filled")

tab_browse, tab_cells, tab_rubric = st.tabs(["Browse", "Cell counts", "Rater prompt"])

with tab_cells:
    if (POOL / "pool_counts.txt").exists():
        st.subheader("Keyword sampling (stage 2)")
        st.code((POOL / "pool_counts.txt").read_text())
    kept = df[df.verdict == "kept"]
    if len(kept):
        st.subheader("Kept after both raters (stage 4), by judged cell")
        st.dataframe(pd.crosstab(kept["glm_topic"], kept["glm_criticality"]), width="content")
        st.subheader("Drop reasons")
        st.dataframe(df["verdict"].value_counts(), width="content")
    else:
        st.write("No candidate rated by both raters yet.")

with tab_rubric:
    st.caption("prompts/criticality.md, sent as `instructions`. User turn: `Message to rate:\\n\\n<<<\\n{prompt}\\n>>>`")
    st.markdown((ROOT / "prompts" / "criticality.md").read_text())

with tab_browse:
    with st.sidebar:
        st.header("Filter")
        def multi(col, label=None):
            vals = sorted(df[col].dropna().astype(str).unique())
            return st.multiselect(label or col, vals, default=vals)
        f_topic = multi("kw_topic", "keyword topic")
        f_crit = multi("kw_crit", "keyword criticality")
        f_verdict = multi("verdict")
        only_sel = st.checkbox("Selected prompts only")
        only_todo = st.checkbox("Not yet human-checked only")
        q = st.text_input("Search prompt text")
    m = df.kw_topic.astype(str).isin(f_topic) & df.kw_crit.astype(str).isin(f_crit) & df.verdict.isin(f_verdict)
    if only_sel:
        m &= df.selected
    if only_todo:
        m &= ~df.id.map(lambda i: bool(sheet.get(i, {}).get("human_keep")))
    if q:
        m &= df.prompt.str.lower().str.contains(q.lower(), regex=False)
    v = df[m]
    st.caption(f"{len(v)} shown")
    st.dataframe(v[["id", "kw_topic", "kw_crit", "glm_topic", "glm_criticality", "qwen_topic", "qwen_criticality",
                    "verdict", "selected", "pilot", "chars"]], hide_index=True, width="stretch")

    if len(v):
        pid = st.selectbox("Open prompt", v["id"].tolist(),
                           format_func=lambda i: f"{i} · {df.set_index('id').at[i, 'prompt'][:90]}")
        r = df.set_index("id").loc[pid]
        st.markdown(f"**{pid}** · source {r['source']} · {r['chars']} chars · keyword cell "
                    f"{r['kw_topic']} / {r['kw_crit']} · **{r['verdict']}**"
                    + (" · ✅ selected" if r["selected"] else "") + (" · 🧪 pilot" if r["pilot"] else ""))
        st.info(r["prompt"])
        cols = st.columns(len(RATERS))
        for col, rt in zip(cols, RATERS):
            with col:
                st.subheader(rt)
                x = ratings[rt].get(pid)
                if x is None:
                    st.write("not rated yet")
                elif not x["parse_ok"]:
                    st.error("parse failure")
                    st.code(x.get("raw_text", "")[:2000])
                else:
                    st.markdown(f"topic **{x['topic']}** · advice **{x['is_advice']}** · level **{x['criticality']}**  \n"
                                f"severity {x['severity']} · reversibility {x['reversibility']} · "
                                f"who bears it {x['who_bears_it']} · override {x['override_high']}  \n"
                                f"_{x.get('reason', '')}_")
                    st.caption(f"call {x['call_id']} · cost {x.get('cost')}")

        st.subheader("Your check (stage 6)")
        if msg := st.session_state.pop("saved_msg", None):
            st.success(msg)
        if not selected:
            st.warning("The 45-prompt set is not written yet (prompts/advice_prompts.jsonl missing), so there is "
                       "nothing selected to check. You can still save a check for any candidate.")
        cur = sheet.get(pid, {})
        sel = selected.get(pid, {})
        def idx(opts, v):
            return opts.index(v) if v in opts else 0
        with st.form(f"human_{pid}"):
            a, b, c = st.columns(3)
            YN = ["", "yes", "no"]
            topic_ok = a.selectbox("topic correct?", YN, index=idx(YN, cur.get("human_topic_ok", "")),
                                   help=f"judged topic: {sel.get('topic') or r['glm_topic'] or '—'}")
            LV = ["", "1", "2", "3"]
            crit = b.selectbox("criticality (yours)", LV, index=idx(LV, cur.get("human_criticality", "")))
            keep = c.selectbox("keep in set?", YN, index=idx(YN, cur.get("human_keep", "")))
            note = st.text_input("note", value=cur.get("human_note", ""))
            if st.form_submit_button("Save to check_human.csv"):
                base = {"id": pid, "topic": sel.get("topic", ""), "criticality": sel.get("criticality", ""),
                        "pilot": sel.get("pilot", ""), "prompt": r["prompt"],
                        "glm_reason": r["glm_reason"] or "", "qwen_reason": r["qwen_reason"] or ""}
                save_human(pid, {"human_topic_ok": topic_ok, "human_criticality": crit,
                                 "human_keep": keep, "human_note": note}, base)
                st.session_state["saved_msg"] = f"Saved {pid} to check_human.csv"
                st.rerun()
