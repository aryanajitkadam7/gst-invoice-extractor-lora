"""Evaluate the rule-based baseline.   python -m invoicex.eval_rules"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .metrics import load_gold, score
from .rules import extract

ROOT = Path(__file__).resolve().parents[2]


def main():
    out = {}
    for split in ("test", "test_hard"):
        rows = load_gold(ROOT / "data" / f"{split}.jsonl")
        t0 = time.perf_counter()
        preds = [extract(r["text"]) for r in rows]
        ms = (time.perf_counter() - t0) * 1000 / len(rows)
        res = score(preds, [json.loads(r["target"]) for r in rows])
        res["latency_ms_per_doc"] = round(ms, 3)
        out[split] = res
        print(f"{split:10s} doc_exact={res['doc_exact']:.3f} field_acc={res['field_accuracy_mean']:.3f} "
              f"item_f1={res['line_item_f1']:.3f}")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "results_rules.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
