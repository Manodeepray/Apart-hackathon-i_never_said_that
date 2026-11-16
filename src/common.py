"""Shared constants and loaders for the main pipeline (S0 -> extraction -> S1..S5 -> judge -> analysis)."""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "prompts"))
from client import ROOT, read_jsonl  # noqa: E402,F401

DATA = ROOT / "data"
SUBJECTS = {"gemma-4-31b-it": "gemma", "deepseek-v4-flash": "dsv4"}  # model -> short id
SUBJECT_BUDGET = {"gemma-4-31b-it": 6000, "deepseek-v4-flash": 12000}  # incl. reasoning; doubled once if incomplete
JUDGE_MODEL = EXTRACTOR_MODEL = "glm-5.2"
REPEAT_TEMP = 0.7

BASE = DATA / "base_answers.jsonl"
CLAIMS = DATA / "claims.jsonl"
RUNS = DATA / "runs.jsonl"
LABELS = DATA / "labels_judge.jsonl"


def prompts():
    return read_jsonl(ROOT / "prompts" / "advice_prompts.jsonl")


def write_jsonl(path, rows):
    """Derived files are rebuilt from the raw cache on every run (raw stays immutable)."""
    tmp = pathlib.Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def parse_args_models(argv_models):
    return [m for m in SUBJECTS if not argv_models or SUBJECTS[m] in argv_models or m in argv_models]


def degenerate(text):
    """Looping output (seen once on gemma at temperature 0: 'our our our ...' for 6,000 tokens)."""
    w = (text or "").split()
    return len(w) > 150 and len(set(w)) / len(w) < 0.1


def subject_call(model, input, *, call_id, **kw):
    """call() for subject models; a degenerate or incomplete reply is re-asked once under call_id + '_rerun1'
    (the bad raw file stays on disk)."""
    from client import call
    r = call(model, input, call_id=call_id, **kw)
    if r["status"] != "completed" or degenerate(r["text"]):
        print(f"  [{call_id}] {'degenerate' if r['status'] == 'completed' else r['status']}: rerun", flush=True)
        r = call(model, input, call_id=call_id + "_rerun1", **kw)
    return r
