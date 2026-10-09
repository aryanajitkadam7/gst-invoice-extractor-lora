"""Scoring extracted invoices against gold JSON.

* json_valid          - output parsed as a JSON object
* field accuracy      - per scalar field; money within ±0.01; names compared case/space-insensitively;
                        dates must be exact ISO strings; null must match null
* line_item_f1        - items matched on (hsn, quantity, amount±0.01)
* doc_exact           - every scalar field AND every line item correct (the bar for straight-through processing)
* gstin_valid         - predicted GSTINs pass the mod-36 checksum (catches hallucinated IDs)
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

from .schema import MONEY_FIELDS, SCALAR_FIELDS, is_valid_gstin


def _norm_text(x) -> str:
    return re.sub(r"\s+", "", str(x)).lower() if x is not None else ""


def _num(x):
    try:
        return round(float(str(x).replace(",", "").replace("₹", "").replace("Rs.", "").strip()), 2)
    except (TypeError, ValueError):
        return None


def field_equal(name: str, pred, gold) -> bool:
    if gold is None or pred is None:
        return gold is None and pred in (None, "", "null")
    if name in MONEY_FIELDS:
        p = _num(pred)
        return p is not None and abs(p - float(gold)) <= 0.011
    return _norm_text(pred) == _norm_text(gold)


def item_key(it: dict):
    try:
        return (str(it.get("hsn")).strip(), int(float(it.get("quantity"))), round(float(it.get("amount")), 2))
    except (TypeError, ValueError, AttributeError):
        return None


def items_f1(pred_items, gold_items) -> tuple[float, bool]:
    gold = [item_key(i) for i in gold_items]
    pred = [item_key(i) for i in (pred_items or []) if isinstance(i, dict)]
    unmatched = list(gold)
    tp = 0
    for p in pred:
        if p in unmatched:
            unmatched.remove(p)
            tp += 1
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gold) if gold else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1, (tp == len(gold) == len(pred))


def score(preds: list[dict | None], golds: list[dict]) -> dict:
    n = len(golds)
    field_hits = defaultdict(int)
    f1s, exact, valid, gst_ok, gst_n = [], 0, 0, 0, 0
    for p, g in zip(preds, golds):
        if p is None:
            f1s.append(0.0)
            continue
        valid += 1
        all_ok = True
        for f in SCALAR_FIELDS:
            ok = field_equal(f, p.get(f), g[f])
            field_hits[f] += ok
            all_ok &= ok
        f1, items_ok = items_f1(p.get("line_items"), g["line_items"])
        f1s.append(f1)
        exact += all_ok and items_ok
        for k in ("seller_gstin", "buyer_gstin"):
            if p.get(k):
                gst_n += 1
                gst_ok += is_valid_gstin(str(p[k]))
    fields = {f: field_hits[f] / n for f in SCALAR_FIELDS}
    return {
        "n": n,
        "json_valid": valid / n,
        "field_accuracy_mean": sum(fields.values()) / len(fields),
        "line_item_f1": sum(f1s) / n,
        "doc_exact": exact / n,
        "predicted_gstin_checksum_valid": gst_ok / gst_n if gst_n else None,
        "fields": fields,
    }


def load_gold(path) -> list[dict]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return rows
