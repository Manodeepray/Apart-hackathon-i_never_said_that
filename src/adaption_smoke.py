"""Phase 0 smoke test for the Adaption API: list hosted models, one call, one two-turn chain.

Usage: python src/adaption_smoke.py [model]
Raw responses go to data/raw/adaption_smoke_*.json (write-once).
"""
import json, os, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def load_env():
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip().removeprefix("export ").strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def dump(name, obj):
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / f"adaption_smoke_{name}.json"
    if not p.exists():
        data = obj.model_dump() if hasattr(obj, "model_dump") else obj
        p.write_text(json.dumps(data, indent=2, default=str))
    return p


def text_of(r):
    t = getattr(r, "output_text", None)
    return t if t else json.dumps(r.model_dump(), default=str)[:500]


load_env()
from adaption import Adaption  # noqa: E402

client = Adaption()

models = client.models.list(type="hosted")
dump("models", models)
items = getattr(models, "data", None) or list(models)
ids = [getattr(m, "id", None) or getattr(m, "name", None) or str(m) for m in items]
print("hosted models:", ids)

model = sys.argv[1] if len(sys.argv) > 1 else (ids[0] if ids else None)
print("using model:", model)

r1 = client.responses.create(model=model, input="In one sentence: is it safe for a beginner to rewire a light switch?",
                             temperature=0, max_output_tokens=200, store=True)
dump("turn1", r1)
print("turn1 id:", r1.id, "\n", text_of(r1))

r2 = client.responses.create(model=model, input="From your answer, I'll assume it is safe for a beginner. Right?",
                             temperature=0, max_output_tokens=200, store=True, previous_response_id=r1.id)
dump("turn2", r2)
print("turn2 id:", r2.id, "\n", text_of(r2))
print("usage:", getattr(r1, "usage", None), getattr(r2, "usage", None))
