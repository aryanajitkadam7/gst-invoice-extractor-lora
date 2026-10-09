import json
import random

from invoicex.datagen import make_split, sample_record
from invoicex.metrics import score
from invoicex.rules import extract
from invoicex.schema import gstin_checksum, is_valid_gstin, parse_json


def test_known_gstin_checksum():
    # publicly documented example GSTIN format check (computed checksum must round-trip)
    first14 = "27AAPFU0939F1Z"
    assert is_valid_gstin(first14 + gstin_checksum(first14))
    assert not is_valid_gstin("27AAPFU0939F1ZX" if gstin_checksum(first14) != "X" else "27AAPFU0939F1ZY")


def test_generated_records_are_internally_consistent():
    rng = random.Random(0)
    for _ in range(200):
        r = sample_record(rng)
        assert is_valid_gstin(r["seller_gstin"])
        assert r["buyer_gstin"] is None or is_valid_gstin(r["buyer_gstin"])
        assert abs(sum(i["amount"] for i in r["line_items"]) - r["taxable_value"]) < 0.01
        assert abs(r["taxable_value"] + r["cgst"] + r["sgst"] + r["igst"] - r["total"]) < 0.011
        intra = r["buyer_gstin"] is None or r["buyer_gstin"][:2] == r["seller_gstin"][:2]
        if r["igst"]:
            assert r["cgst"] == r["sgst"] == 0
        if r["buyer_gstin"] and intra:
            assert r["igst"] == 0


def test_splits_are_deterministic_and_disjoint_templates():
    a, b = make_split(5, ["T1"], 7), make_split(5, ["T1"], 7)
    assert a == b
    assert {r["template"] for r in make_split(50, ["T5", "T6"], 4)} <= {"T5", "T6"}


def test_parse_json_handles_fences_and_garbage():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": {"b": 2}} trailing') == {"a": {"b": 2}}
    assert parse_json("no json here") is None
    assert parse_json('{"a": 1,,}') is None


def test_score_perfect_and_empty():
    rows = make_split(10, ["T1", "T2"], 3)
    gold = [json.loads(r["target"]) for r in rows]
    s = score(gold, gold)
    assert s["doc_exact"] == 1.0 and s["json_valid"] == 1.0 and s["line_item_f1"] == 1.0
    s0 = score([None] * 10, gold)
    assert s0["doc_exact"] == 0 and s0["json_valid"] == 0


def test_rules_work_on_clean_seen_template():
    rows = make_split(30, ["T1"], 11, noise=(0.0, 0.0))
    preds = [extract(r["text"]) for r in rows]
    s = score(preds, [json.loads(r["target"]) for r in rows])
    assert s["doc_exact"] > 0.9
