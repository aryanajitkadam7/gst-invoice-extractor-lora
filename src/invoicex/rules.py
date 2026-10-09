"""Rule-based baseline: regexes + heuristics, written against the four training templates (T1-T4).

This is what a team would typically ship first. It is evaluated on the same test sets
as the LLMs, including the held-out templates it was never written for.
"""
from __future__ import annotations

import re
from datetime import datetime

from .schema import GSTIN_RE, STATES

MONEY = r"(?:Rs\.?|INR|₹)?\s*([\d,]+\.\d{2})"
DATE_FORMATS = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d-%b-%Y", "%d %B %Y", "%d-%b-%y", "%Y-%m-%d"]


def _money(pattern: str, text: str):
    m = re.search(pattern + r"\s*[:,\-]?\s*" + MONEY, text, re.I | re.M)
    return round(float(m.group(1).replace(",", "")), 2) if m else None


def _date(text: str):
    m = re.search(r"(?:Date|Dated|Inv\. Date|Bill Date)\s*[:,]?\s*([0-9]{1,4}[\-/. ][0-9A-Za-z]{1,9}[\-/. ][0-9]{2,4})", text)
    if not m:
        return None
    raw = m.group(1).strip()
    for f in DATE_FORMATS:
        try:
            return datetime.strptime(raw, f).date().isoformat()
        except ValueError:
            pass
    return None


def _first(patterns, text):
    for p in patterns:
        m = re.search(p, text, re.I | re.M)
        if m:
            return m.group(1).strip()
    return None


def extract(text: str) -> dict:
    gstins = GSTIN_RE.findall(text)
    seller_g = _first([r"(?:^GSTIN|Supplier.*GSTIN|GST Reg No|Seller.*GSTIN)\s*:?\s*(\w{15})"], text)
    buyer_g = _first([r"(?:Buyer GSTIN|Customer GST No\.|Party GSTIN|Buyer .*? GSTIN)\s*[:,]?\s*(\w{15})"], text)
    if seller_g is None and gstins:
        seller_g = gstins[0]
    if buyer_g is None and len(gstins) > 1:
        buyer_g = [g for g in gstins if g != seller_g][0]

    items = []
    for m in re.finditer(r"^\|\s*\d+\s*\|\s*(.+?)\s*\|\s*(\d{4,8})\s*\|\s*(\d+)\s*\|\s*([\d,]+\.\d{2})\s*\|.*?\|\s*([\d,]+\.\d{2})\s*\|", text, re.M):
        items.append((m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)))
    for m in re.finditer(r"^\d+\s+(.+?)\s+(\d{4,8})\s+(\d+)\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s*$", text, re.M):
        items.append(m.groups())
    for m in re.finditer(r"Item \d+:\s*(.+?);\s*HSN\s*(\d{4,8});\s*Qty\s*(\d+)\s*@\s*([\d,]+\.\d{2})\s*=\s*([\d,]+\.\d{2})", text):
        items.append(m.groups())
    for m in re.finditer(r"^([^,\n]+),(\d{4,8}),(\d+),([\d,]+\.\d{2}),\d+%,([\d,]+\.\d{2})$", text, re.M):
        items.append(m.groups())

    pos = _first([r"(?:Place of Supply|State of supply|POS|Supply State)\s*[:,\-]?\s*(" + "|".join(STATES.values()) + ")"], text)
    rec = {
        "invoice_number": _first([r"Invoice No:\s*(\S+)", r"Invoice #\s*(\S+)", r"Inv\. No\.\s*(\S+)", r"Bill No,(\S+)"], text),
        "invoice_date": _date(text),
        "seller_name": _first([r"^TAX INVOICE\n(.+)$", r"Supplier:\s*(.+?)\s*\(GSTIN", r"Seller\s+(.+?)\s+GSTIN", r"\A(.+)\nGST Reg No"], text),
        "seller_gstin": seller_g,
        "buyer_name": _first([r"Bill To:\s*(.+)$", r"Customer:\s*(.+)$", r"Buyer\s+(.+?)\s+GSTIN", r"Party,(.+)$"], text),
        "buyer_gstin": buyer_g,
        "place_of_supply": pos,
        "line_items": [{"description": d.strip(), "hsn": h, "quantity": int(q),
                        "rate": float(r.replace(",", "")), "amount": float(a.replace(",", ""))} for d, h, q, r, a in items],
        "taxable_value": _money(r"(?:Taxable Value|Sub Total|Taxable Amt|Taxable)", text),
        "cgst": _money(r"(?:CGST Amt|CGST|Central Tax)", text) or 0.0,
        "sgst": _money(r"(?:SGST Amt|SGST|State Tax)", text) or 0.0,
        "igst": _money(r"(?:IGST Amt|IGST|Integrated Tax)", text) or 0.0,
        "total": _money(r"(?:Grand Total|Total Amount|Invoice Total|^Total)", text),
    }
    return rec
