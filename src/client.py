"""Thin Adaption wrapper: retries, write-once raw cache, cost log, completion check, JSON parsing.

    from client import call, parse_json, run_parallel
    r = call("glm-5.2", "text", call_id="rate_glm_oeq_12", instructions="...", temperature=0)
    r["text"], r["status"], r["cost"], r["response_id"]

Raw cache: data/raw/{call_id}.json holds {"request", "response", "timestamp"} and is never
rewritten. A call whose final file exists is not sent again (this makes every script resumable).
Incomplete responses (reasoning ran out of budget) are kept as {call_id}.incomplete{n}.json and
retried with a doubled token budget.
"""
import datetime, json, os, pathlib, re, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
COST_LOG = ROOT / "data" / "cost_log.jsonl"

# credits per 1M tokens (input, output), pasted by the user 2026-11-16
PRICES = {
    "gemma-4-31b-it": (0.09, 0.34), "glm-5.2": (1.40, 4.40), "qwen3.5-397b": (0.55, 3.50),
    "gemma-4-26b-a4b-it": (0.077, 0.26), "qwen3.8-27b": (0.039, 1.25),
    "deepseek-v4-flash": (0.045, 0.14), "glm-5.3-flash": (0.15, 0.50),
}

_lock = threading.Lock()
_client = None


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def client():
    global _client
    if _client is None:
        load_env()
        from adaption import Adaption
        _client = Adaption(timeout=600)
    return _client


def estimate(model, in_tokens, out_tokens):
    pi, po = PRICES[model]
    return (in_tokens * pi + out_tokens * po) / 1e6


def _summarise(call_id, d):
    resp = d["response"]
    reasoning = []
    for item in resp.get("output") or []:
        if item.get("type") == "reasoning":
            for c in item.get("content") or []:
                if c.get("text"):
                    reasoning.append(c["text"])
    u = resp.get("usage") or {}
    return {"call_id": call_id, "response_id": resp.get("id"), "model": resp.get("model"),
            "status": resp.get("status"), "text": resp.get("output_text") or "",
            "reasoning": "\n".join(reasoning), "input_tokens": u.get("input_tokens"),
            "output_tokens": u.get("output_tokens"), "cost": u.get("cost"),
            "timestamp": d.get("timestamp")}


def cached(call_id):
    p = RAW / f"{call_id}.json"
    return _summarise(call_id, json.loads(p.read_text())) if p.exists() else None


def call(model, input, *, call_id, instructions=None, temperature=0.0, max_output_tokens=4000,
         store=False, previous_response_id=None, max_retries=7, max_budget_doublings=1):
    """One API call, cached by call_id. Returns a summary dict (see _summarise)."""
    hit = cached(call_id)
    if hit:
        return hit
    RAW.mkdir(parents=True, exist_ok=True)
    budget = max_output_tokens
    for doubling in range(max_budget_doublings + 1):
        req = dict(model=model, input=input, instructions=instructions, temperature=temperature,
                   max_output_tokens=budget, store=store, previous_response_id=previous_response_id)
        req = {k: v for k, v in req.items() if v is not None}
        for attempt in range(max_retries):
            try:
                resp = client().responses.create(**req)
                break
            except Exception as e:  # network, rate limit, 5xx
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"  [{call_id}] {type(e).__name__}: retry in {wait}s")
                time.sleep(wait)
        d = {"request": req, "response": resp.model_dump(mode="json"),
             "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        u = d["response"].get("usage") or {}
        with _lock, open(COST_LOG, "a") as f:
            f.write(json.dumps({"call_id": call_id, "model": model, "cost": u.get("cost"),
                                "input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
                                "status": d["response"].get("status"), "timestamp": d["timestamp"]}) + "\n")
        if d["response"].get("status") == "completed":
            (RAW / f"{call_id}.json").write_text(json.dumps(d, indent=1))
            return _summarise(call_id, d)
        (RAW / f"{call_id}.incomplete{doubling}.json").write_text(json.dumps(d, indent=1))
        print(f"  [{call_id}] status={d['response'].get('status')} at {budget} tokens; doubling budget")
        budget *= 2
    out = _summarise(call_id, d)
    out["status"] = out["status"] or "incomplete"
    return out


def parse_json(text):
    """First {...} object in text, or None. Tolerates code fences and surrounding prose."""
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text)
    start = t.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = t.find("{", start + 1)
    return None


def run_parallel(fn, items, workers=4):
    """Apply fn to items with a small thread pool; yields (item, result or exception)."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for f in as_completed(futs):
            try:
                yield futs[f], f.result()
            except Exception as e:
                yield futs[f], e


def append_jsonl(path, row):
    with _lock, open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    p = pathlib.Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
