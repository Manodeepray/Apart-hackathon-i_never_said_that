"""Ladder viewer: browse prompts, base answers, every ladder step and every label source.

Run:  streamlit run app/viewer.py

Works on any "run folder" under data/ that contains a runs.jsonl
(e.g. data/pilot/ now, data/ for the Adaption pipeline later). Everything is optional except runs.jsonl:

  runs.jsonl                one row per (claim, step, sample). Keys used if present:
                            prompt_id, claim_id, claim_type, claim / claim_text, step, sample_idx,
                            model, condition, criticality, challenge, text, thinking,
                            source, source_real, neg_claim, temperature, response_id, call_id
  claims.jsonl              extra claim fields, joined on claim_id
  base_answers.jsonl        prompt_id (+ model / condition) -> answer, thinking
  *prompts*.jsonl           prompt_id -> prompt text, dataset, subset, criticality
  labels_*.jsonl / .csv     any number of label sources, keyed on claim_id + step (+ sample_idx),
                            with any of: label, ownership, stance, rationale, note
                            (labels_judge.jsonl, labels_human.csv, labels_claude_manual.csv, ...)

Human labels entered here go to <run>/labels_human_app.csv (append-only, newest wins).
"""
import csv, json, pathlib, datetime
import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LABELS = ["Defends", "Corrects", "Softens", "Denies", "n/a"]
OWNERSHIP = ["owns", "disowns", "silent", "unclear", "n/a"]  # silent = takes a stance without addressing its own answer
STANCE = ["endorses", "rejects", "hedges", "n/a"]
LABEL_COLOR = {"Defends": "#2e7d32", "Corrects": "#1565c0", "Softens": "#ef6c00",
               "Denies": "#c62828", "n/a": "#757575"}

st.set_page_config(page_title="Ladder viewer", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.5rem; max-width: 1200px;}
.bubble {border-radius: 10px; padding: 10px 14px; margin: 6px 0; line-height: 1.5;}
.user {background: #eef3fb; border: 1px solid #d6e2f5;}
.model {background: #ffffff; border: 1px solid #e6e6e6;}
.who {font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: .04em;}
.pill {display: inline-block; padding: 1px 9px; border-radius: 999px; color: #fff;
       font-size: 0.8rem; margin-right: 6px;}
.meta {color: #555; font-size: 0.85rem;}
</style>""", unsafe_allow_html=True)


# ---------- loading ----------------------------------------------------------

def read_jsonl(p):
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def read_table(p):
    if p.suffix == ".csv":
        return pd.read_csv(p, dtype=str, keep_default_na=False).to_dict("records")
    return read_jsonl(p)


def norm_runs(rows):
    out = []
    for r in rows:
        r = dict(r)
        r["claim"] = r.get("claim") or r.get("claim_text") or ""
        r["sample_idx"] = int(r.get("sample_idx") or 0)
        r["step"] = int(r.get("step") or 0)
        r.setdefault("model", "?")
        r.setdefault("condition", "-")
        r["branch"] = r.get("branch") or "ladder"
        r["kind"] = r.get("kind") or "main"
        r["claim_id"] = r.get("claim_id") or f"{r.get('prompt_id')}|{r.get('claim_type')}|{r['claim'][:40]}"
        out.append(r)
    return out


@st.cache_data(show_spinner=False)
def load_run(run_dir: str, mtime: float):
    d = pathlib.Path(run_dir)
    runs = norm_runs(read_jsonl(d / "runs.jsonl"))
    claims = {c["claim_id"]: c for c in read_jsonl(d / "claims.jsonl")} if (d / "claims.jsonl").exists() else {}
    for r in runs:  # fill gaps from claims.jsonl
        for k, v in claims.get(r["claim_id"], {}).items():
            if r.get(k) in (None, ""):
                r[k] = v
    base = read_jsonl(d / "base_answers.jsonl") if (d / "base_answers.jsonl").exists() else []
    prompts = {}
    for p in list(d.glob("*prompts*.jsonl")) + [ROOT / "prompts" / "advice_prompts.jsonl"]:
        if p.exists():
            for row in read_jsonl(p):
                prompts.setdefault(row.get("id") or row.get("prompt_id"), row)
    labels = {}
    for p in sorted(list(d.glob("labels_*.jsonl")) + list(d.glob("labels_*.csv"))):
        src = p.stem.removeprefix("labels_")
        table = {}
        for row in read_table(p):
            if not row.get("claim_id"):
                continue
            key = (row["claim_id"], row.get("branch") or "ladder",
                   int(row.get("step") or 0), int(row.get("sample_idx") or 0))
            table[key] = row  # later rows win (append-only files)
        labels[src] = table
    return runs, base, prompts, labels


def run_dirs():
    return sorted({p.parent for p in DATA.rglob("runs.jsonl")}, key=lambda p: str(p))


def find_base(base, r):
    cands = [b for b in base if b.get("prompt_id") == r["prompt_id"]]
    for key in ("model", "condition"):
        narrowed = [b for b in cands if b.get(key) in (None, r.get(key))]
        cands = narrowed or cands
    return cands[0] if cands else None


# ---------- sidebar: choose data and filters -----------------------------------

dirs = run_dirs()
if not dirs:
    st.info("No runs.jsonl found under data/. Run the pipeline first.")
    st.stop()

with st.sidebar:
    st.header("Data")
    run_dir = st.selectbox("Run folder", dirs, format_func=lambda p: str(p.relative_to(ROOT)))
    if st.button("Reload files"):
        st.cache_data.clear()
    runs, base, prompts, labels = load_run(str(run_dir), (run_dir / "runs.jsonl").stat().st_mtime)
    df = pd.DataFrame(runs)

    st.header("Filter")
    def multi(col, label=None):
        if col not in df or df[col].isna().all():
            return None
        vals = sorted(df[col].dropna().astype(str).unique())
        return st.multiselect(label or col, vals, default=vals) if len(vals) > 1 else vals
    f = {c: multi(c) for c in ["model", "condition", "topic", "claim_type", "criticality", "x_true", "branch", "blatancy"]}
    datasets = sorted({str(prompts.get(p, {}).get("dataset", "?")) for p in df["prompt_id"]})
    f_ds = st.multiselect("dataset", datasets, default=datasets) if len(datasets) > 1 else datasets
    label_src = st.selectbox("Filter by label from", ["(none)"] + list(labels))
    f_label = None
    if label_src != "(none)":
        f_label = st.multiselect("label (any step)", LABELS, default=LABELS)
    text_q = st.text_input("Search text (claim, prompt, replies)")

    st.header("Show")
    show_thinking = st.checkbox("Thinking", value=True)
    show_meta = st.checkbox("Call metadata", value=False)
    show_sources = st.multiselect("Label sources", list(labels), default=list(labels))
    enable_labelling = st.checkbox("Enable my labelling", value=False,
                                   help="Writes to labels_human_app.csv in the run folder")

# apply filters at claim level
mask = pd.Series(True, index=df.index)
for c, vals in f.items():
    if vals is not None and c in df:
        mask &= df[c].isna() | df[c].astype(str).isin(vals)  # rows without this field are kept
mask &= df["prompt_id"].map(lambda p: str(prompts.get(p, {}).get("dataset", "?"))).isin(f_ds)
if text_q:
    q = text_q.lower()
    hit_claims = set(df[df["claim"].str.lower().str.contains(q, regex=False)
                        | df.get("text", pd.Series("", index=df.index)).fillna("").str.lower().str.contains(q, regex=False)
                        | df["prompt_id"].map(lambda p: q in str(prompts.get(p, {}).get("prompt", "")).lower())
                        ]["claim_id"])
    mask &= df["claim_id"].isin(hit_claims)
if f_label is not None:
    keep = {k[0] for k, v in labels[label_src].items() if v.get("label") in f_label}
    mask &= df["claim_id"].isin(keep)
view_all = df[mask]
conf_rows = view_all[view_all["kind"] == "confidence"]
view = view_all[view_all["kind"] == "main"]


def conf_for(cid, br, step, si):
    """Confidence side-branch answer for this point, as text, or None."""
    m = conf_rows[(conf_rows["claim_id"] == cid) & (conf_rows["step"] == step) & (conf_rows["sample_idx"] == si)
                  & (conf_rows["branch"].isin([br, "s0"] if step == 0 else [br]))]
    if m.empty:
        return None
    c = m.iloc[0].get("confidence")
    return f"{float(c):g}%" if c not in (None, "") and c == c else f"unparsed: {str(m.iloc[0].get('text', ''))[:80]}"


def conf_line(c):
    if c:
        st.markdown(f"<span class='meta'>🎯 confidence in X (side branch): <b>{c}</b></span>", unsafe_allow_html=True)

# ---------- overview ----------------------------------------------------------

st.title("Ladder viewer")
st.markdown(f"<span class='meta'>{run_dir.relative_to(ROOT)} · {view['claim_id'].nunique()} conversations · "
            f"{len(view)} replies · label sources: {', '.join(labels) or 'none'}</span>", unsafe_allow_html=True)

tab_conv, tab_table = st.tabs(["Conversation", "Label table"])

with tab_table:
    if not labels:
        st.write("No label files yet.")
    else:
        src = st.selectbox("Label source", list(labels), key="tbl_src")
        rows = []
        for cid in view["claim_id"].unique():
            sub = view[view["claim_id"] == cid].iloc[0]
            row = {"claim_id": cid, "type": sub.get("claim_type"), "crit": sub.get("criticality"),
                   "model": sub.get("model"), "cond": sub.get("condition")}
            for br, s in sorted(set(zip(view["branch"], view["step"]))):
                lab = labels[src].get((cid, br, int(s), int(sub["sample_idx"])), {})
                row[f"{br}:S{s}"] = " / ".join(x for x in (lab.get("label"), lab.get("ownership")) if x)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

with tab_conv:
    convs = list(dict.fromkeys(zip(view["claim_id"], view["sample_idx"], view["branch"])))
    if not convs:
        st.warning("Nothing matches the filters.")
        st.stop()
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    st.session_state.idx = min(st.session_state.idx, len(convs) - 1)

    c1, c2, c3 = st.columns([1, 6, 1])
    if c1.button("◀ Prev", width="stretch"):
        st.session_state.idx = (st.session_state.idx - 1) % len(convs)
    if c3.button("Next ▶", width="stretch"):
        st.session_state.idx = (st.session_state.idx + 1) % len(convs)
    names = [f"{cid} · {br} (sample {si})" for cid, si, br in convs]
    pick = c2.selectbox("Conversation", range(len(convs)), index=st.session_state.idx,
                        format_func=lambda i: f"{i + 1}/{len(convs)} · {names[i]}", label_visibility="collapsed")
    st.session_state.idx = pick
    cid, si, br = convs[pick]
    steps = view[(view["claim_id"] == cid) & (view["sample_idx"] == si) & (view["branch"] == br)].sort_values("step")
    r0 = steps.iloc[0].to_dict()
    pr = prompts.get(r0["prompt_id"], {})

    # header
    st.subheader(r0["claim"])
    bits = [f"<b>{r0.get('claim_type', '?')}</b>", f"prompt <code>{r0['prompt_id']}</code>",
            f"model {r0.get('model')}", f"condition {r0.get('condition')}",
            f"criticality {r0.get('criticality', '?')}", f"branch {br}", f"x_true {r0.get('x_true', '?')}", f"dataset {pr.get('dataset', '?')}/{pr.get('subset', '')}"]
    st.markdown("<span class='meta'>" + " · ".join(bits) + "</span>", unsafe_allow_html=True)
    if r0.get("neg_claim") or r0.get("source"):
        st.markdown(f"<span class='meta'>not-X: {r0.get('neg_claim', '')} · source: {r0.get('source', '')} "
                    f"(real: {r0.get('source_real', '?')})</span>", unsafe_allow_html=True)

    def bubble(who, text, cls):
        safe = (text or "").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        st.markdown(f"<div class='bubble {cls}'><div class='who'>{who}</div>{safe}</div>", unsafe_allow_html=True)

    def label_line(src, lab):
        if not lab:
            return
        col = LABEL_COLOR.get(lab.get("label"), "#999")
        extra = " · ".join(f"{k}: {lab[k]}" for k in ("ownership", "stance") if lab.get(k))
        why = lab.get("rationale") or lab.get("note") or ""
        st.markdown(f"<span class='pill' style='background:{col}'>{lab.get('label', '?')}</span>"
                    f"<span class='meta'><b>{src}</b> · {extra}{' · ' + why if why else ''}</span>",
                    unsafe_allow_html=True)

    with st.expander("User prompt", expanded=False):
        st.write(pr.get("prompt", "(prompt text not found)"))
    b = find_base(base, r0)
    st.markdown("**Base answer**")
    if b:
        if show_thinking and b.get("thinking"):
            with st.expander("thinking"):
                st.write(b["thinking"])
        bubble("model · base answer", b.get("answer") or b.get("text"), "model")
        conf_line(conf_for(cid, br, 0, si))
    else:
        st.caption("(no base answer file)")

    for _, r in steps.iterrows():
        r = r.to_dict()
        s = int(r["step"])
        st.markdown(f"---\n**Step {s}**")
        bubble(f"user · step {s}", r.get("challenge", ""), "user")
        if show_thinking and r.get("thinking"):
            with st.expander("thinking"):
                st.write(r["thinking"])
        bubble("model", r.get("text", ""), "model")
        conf_line(conf_for(cid, br, s, si))
        for src in show_sources:
            label_line(src, labels[src].get((cid, br, s, int(si))))
        if show_meta:
            st.caption(" · ".join(f"{k}: {r.get(k)}" for k in
                                  ("call_id", "response_id", "parent_response_id", "temperature", "timestamp")
                                  if r.get(k) not in (None, "")))
        if enable_labelling:
            with st.form(f"lab_{cid}_{br}_{si}_{s}"):
                a, bcol, c, d = st.columns([1, 1, 1, 3])
                lab = a.selectbox("label", [""] + LABELS, key=f"l{cid}{br}{si}{s}")
                own = bcol.selectbox("ownership", [""] + OWNERSHIP, key=f"o{cid}{br}{si}{s}")
                stn = c.selectbox("stance", [""] + STANCE, key=f"s{cid}{br}{si}{s}")
                note = d.text_input("note", key=f"n{cid}{br}{si}{s}")
                if st.form_submit_button("Save my label") and lab:
                    out = run_dir / "labels_human_app.csv"
                    new = not out.exists()
                    with open(out, "a", newline="") as fo:
                        w = csv.writer(fo)
                        if new:
                            w.writerow(["claim_id", "branch", "step", "sample_idx", "label", "ownership", "stance", "note", "timestamp"])
                        w.writerow([cid, br, s, si, lab, own, stn, note, datetime.datetime.now().isoformat()])
                    st.cache_data.clear()
                    st.success("Saved")
