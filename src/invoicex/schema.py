"""Target schema, GSTIN validation and normalisation helpers."""
from __future__ import annotations

import json
import re

ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_RE = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")

SCALAR_FIELDS = ["invoice_number", "invoice_date", "seller_name", "seller_gstin", "buyer_name", "buyer_gstin",
                 "place_of_supply", "taxable_value", "cgst", "sgst", "igst", "total"]
MONEY_FIELDS = {"taxable_value", "cgst", "sgst", "igst", "total"}
ITEM_FIELDS = ["description", "hsn", "quantity", "rate", "amount"]

STATES = {
    "27": "Maharashtra", "29": "Karnataka", "07": "Delhi", "33": "Tamil Nadu", "24": "Gujarat",
    "09": "Uttar Pradesh", "36": "Telangana", "19": "West Bengal", "08": "Rajasthan", "32": "Kerala",
    "06": "Haryana", "23": "Madhya Pradesh",
}

JSON_TEMPLATE = {
    "invoice_number": "string", "invoice_date": "YYYY-MM-DD", "seller_name": "string",
    "seller_gstin": "15-char GSTIN", "buyer_name": "string", "buyer_gstin": "15-char GSTIN or null",
    "place_of_supply": "state name",
    "line_items": [{"description": "string", "hsn": "string", "quantity": 0, "rate": 0.0, "amount": 0.0}],
    "taxable_value": 0.0, "cgst": 0.0, "sgst": 0.0, "igst": 0.0, "total": 0.0,
}


def gstin_checksum(first14: str) -> str:
    total = 0
    for i, ch in enumerate(first14):
        prod = ALNUM.index(ch) * (1 if i % 2 == 0 else 2)
        total += prod // 36 + prod % 36
    return ALNUM[(36 - total % 36) % 36]


def is_valid_gstin(g: str | None) -> bool:
    return bool(g) and bool(GSTIN_RE.fullmatch(g)) and gstin_checksum(g[:14]) == g[14]


def to_json(record: dict) -> str:
    """Canonical compact JSON used as the training target."""
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def parse_json(text: str) -> dict | None:
    """Extract the first JSON object from model output; None if invalid."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None
