"""Synthetic Indian GST invoice generator: noisy OCR-style text + gold JSON.

    python -m invoicex.datagen --out data --train 2000 --val 200 --test 300 --hard 300

Why synthetic: real invoices contain personal and commercial data and cannot be
published. The generator gives exact labels and lets us control the difficulty.

Realism built in:
  * valid GSTINs (correct mod-36 checksum), state codes that match the place of supply
  * correct GST maths: intra-state -> CGST + SGST, inter-state -> IGST; per-item rates 5/12/18/28%
  * Indian number formatting (1,23,456.00), Rs./INR/₹ prefixes, 7 date formats
  * OCR noise on free text (O/0, l/1, S/5 swaps, dropped or doubled spaces), random field order

Splits:
  train / val / test - templates T1-T4
  hard               - templates T5-T6, never seen in training (tests layout generalisation)
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from .schema import ALNUM, STATES, gstin_checksum, to_json

FIRST = ["Shree", "Sai", "Om", "Balaji", "Ganesh", "Laxmi", "Krishna", "Mahalaxmi", "Siddhi", "Annapurna",
         "Jai Bharat", "New India", "Royal", "Sunrise", "Galaxy", "Pioneer", "Apex", "Vardhman", "Patel", "Sharma"]
LAST = ["Traders", "Enterprises", "Distributors", "Agencies", "Industries", "Electricals", "Hardware",
        "Pharma", "Textiles", "Foods Pvt Ltd", "Steel Corporation", "Infotech LLP", "Stores", "Suppliers"]
BUYERS = ["Kumar Retail", "Mehta Constructions", "Deshmukh Hospital", "Iyer & Sons", "Reddy Foods",
          "Gupta Medical Store", "Joshi Automobiles", "Nair Bakers", "Singh Logistics", "Kulkarni Textiles",
          "Bose Electronics", "Khan Hardware", "Agarwal Sweets", "Pillai Clinic", "Chopra Furnishings"]
CATALOG = [  # description, HSN, GST rate, price range
    ("Basmati Rice 25kg Bag", "1006", 5, (1200, 2600)), ("Toor Dal 1kg", "0713", 5, (110, 180)),
    ("Sunflower Oil 15L Tin", "1512", 5, (1700, 2400)), ("Paracetamol 500mg Strip", "3004", 12, (12, 40)),
    ("Surgical Gloves Box", "4015", 12, (250, 600)), ("Cotton Shirt", "6205", 12, (450, 1400)),
    ("LED Bulb 9W", "8539", 18, (60, 140)), ("Copper Wire 90m Coil", "8544", 18, (1500, 3200)),
    ("Laptop Bag", "4202", 18, (600, 1800)), ("PVC Pipe 4 inch", "3917", 18, (300, 900)),
    ("Office Chair", "9401", 18, (2500, 7500)), ("Printer Cartridge", "8443", 18, (700, 2200)),
    ("Cement 50kg Bag", "2523", 28, (360, 450)), ("Split AC 1.5 Ton", "8415", 28, (28000, 42000)),
    ("Two Wheeler Tyre", "4011", 28, (1400, 3200)), ("Ceramic Tiles Box", "6907", 18, (500, 1300)),
    ("Steel Almirah", "9403", 18, (6000, 12000)), ("Mobile Charger", "8504", 18, (250, 900)),
    ("Notebook A4 Pack", "4820", 12, (180, 420)), ("Ghee 1L", "0405", 12, (520, 680)),
]


def make_gstin(rng: random.Random, state: str) -> str:
    pan = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(3)) + rng.choice("CPFH") + \
          rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + f"{rng.randint(0, 9999):04d}" + rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    first14 = state + pan + rng.choice("123456789") + "Z"
    return first14 + gstin_checksum(first14)


def inr(x: float, rng: random.Random, prefix: bool = True) -> str:
    """Indian digit grouping: 1,23,456.78"""
    s = f"{x:.2f}"
    whole, frac = s.split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    out = f"{whole}.{frac}"
    if prefix and rng.random() < 0.6:
        out = rng.choice(["Rs.", "Rs ", "INR ", "₹", "₹ "]) + out
    return out


DATE_FORMATS = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d-%b-%Y", "%d %B %Y", "%d-%b-%y", "%Y-%m-%d"]


def ocr_noise(text: str, rng: random.Random, p: float) -> str:
    swaps = {"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8", "e": "c"}
    out = []
    for ch in text:
        r = rng.random()
        if ch in swaps and r < p:
            out.append(swaps[ch])
        elif ch == " " and r < p:
            out.append(rng.choice(["", "  "]))
        else:
            out.append(ch)
    return "".join(out)


def sample_record(rng: random.Random) -> dict:
    s_state = rng.choice(list(STATES))
    inter = rng.random() < 0.4
    b_state = rng.choice([s for s in STATES if s != s_state]) if inter else s_state
    b2c = rng.random() < 0.15
    items = []
    for desc, hsn, rate_pct, (lo, hi) in rng.sample(CATALOG, rng.randint(1, 4)):
        qty = rng.choice([1, 1, 2, 3, 4, 5, 6, 10, 12, 20, 25, 50])
        price = round(rng.uniform(lo, hi), rng.choice([0, 2]))
        items.append({"description": desc, "hsn": hsn, "quantity": qty, "rate": float(price),
                      "amount": round(qty * price, 2), "_gst": rate_pct})
    taxable = round(sum(i["amount"] for i in items), 2)
    tax = round(sum(i["amount"] * i["_gst"] / 100 for i in items), 2)
    cgst = sgst = 0.0
    igst = 0.0
    if inter:
        igst = tax
    else:
        cgst = sgst = round(tax / 2, 2)
    d = date(2024, 4, 1) + timedelta(days=rng.randint(0, 700))
    prefix = rng.choice(["INV", "TI", "GST", "SI", "BILL", ""])
    fy = f"{d.year % 100}-{(d.year + 1) % 100:02d}" if d.month >= 4 else f"{(d.year - 1) % 100}-{d.year % 100:02d}"
    num = rng.choice([f"{prefix}/{fy}/{rng.randint(1, 9999):04d}", f"{prefix}-{rng.randint(100, 99999)}",
                      f"{rng.randint(1, 9999)}"]).lstrip("/-")
    rec = {
        "invoice_number": num, "invoice_date": d.isoformat(),
        "seller_name": f"{rng.choice(FIRST)} {rng.choice(LAST)}", "seller_gstin": make_gstin(rng, s_state),
        "buyer_name": rng.choice(BUYERS), "buyer_gstin": None if b2c else make_gstin(rng, b_state),
        "place_of_supply": STATES[b_state],
        "line_items": items,
        "taxable_value": taxable, "cgst": cgst, "sgst": sgst, "igst": igst,
        "total": round(taxable + cgst + sgst + igst, 2),
    }
    return rec


# ---------------------------------------------------------------- templates
def _items_block(rec, rng, style):
    lines = []
    for n, it in enumerate(rec["line_items"], 1):
        q, r, a = it["quantity"], inr(it["rate"], rng, False), inr(it["amount"], rng, False)
        if style == "pipe":
            lines.append(f"| {n} | {it['description']} | {it['hsn']} | {q} | {r} | {it['_gst']}% | {a} |")
        elif style == "space":
            lines.append(f"{n}  {it['description']:<24} {it['hsn']:>6} {q:>4} {r:>10} {a:>12}")
        elif style == "kv":
            lines.append(f"Item {n}: {it['description']}; HSN {it['hsn']}; Qty {q} @ {r} = {a}")
        else:  # csvish
            lines.append(f"{it['description']},{it['hsn']},{q},{r},{it['_gst']}%,{a}")
    return lines


def _tax_lines(rec, rng):
    out = []
    if rec["igst"]:
        out.append(f"{rng.choice(['IGST', 'Integrated Tax', 'IGST Amt'])}: {inr(rec['igst'], rng)}")
    else:
        out.append(f"{rng.choice(['CGST', 'Central Tax', 'CGST Amt'])}: {inr(rec['cgst'], rng)}")
        out.append(f"{rng.choice(['SGST', 'State Tax', 'SGST Amt'])}: {inr(rec['sgst'], rng)}")
    return out


def render(rec: dict, template: str, rng: random.Random, noise: float) -> str:
    d = date.fromisoformat(rec["invoice_date"]).strftime(rng.choice(DATE_FORMATS))
    n = lambda s: ocr_noise(s, rng, noise)
    buyer_g = rec["buyer_gstin"] or rng.choice(["Unregistered", "URP", "N/A", "-"])
    tot_label = rng.choice(["Grand Total", "Total Amount", "Invoice Total", "Total"])
    if template == "T1":  # classic labelled header + pipe table
        L = ["TAX INVOICE", n(rec["seller_name"]), f"GSTIN: {rec['seller_gstin']}",
             f"Invoice No: {rec['invoice_number']}", f"Date: {d}",
             f"Bill To: {n(rec['buyer_name'])}", f"Buyer GSTIN: {buyer_g}",
             f"Place of Supply: {rec['place_of_supply']}",
             "| # | Description | HSN | Qty | Rate | GST | Amount |", *_items_block(rec, rng, "pipe"),
             f"Taxable Value: {inr(rec['taxable_value'], rng)}", *_tax_lines(rec, rng),
             f"{tot_label}: {inr(rec['total'], rng)}"]
    elif template == "T2":  # fixed-width columns, buyer first
        L = [f"Invoice # {rec['invoice_number']}    Dated {d}", f"Customer: {n(rec['buyer_name'])}",
             f"Customer GST No.: {buyer_g}", f"Supplier: {n(rec['seller_name'])} (GSTIN {rec['seller_gstin']})",
             f"State of supply - {rec['place_of_supply']}", "Sl  Item                      HSN  Qty       Rate       Amount",
             *_items_block(rec, rng, "space"), f"Sub Total {inr(rec['taxable_value'], rng)}", *_tax_lines(rec, rng),
             f"{tot_label} {inr(rec['total'], rng)}"]
    elif template == "T3":  # key-value, single stream
        parts = [f"Seller {n(rec['seller_name'])} GSTIN {rec['seller_gstin']}",
                 f"Inv. No. {rec['invoice_number']} Inv. Date {d}",
                 f"Buyer {n(rec['buyer_name'])} GSTIN {buyer_g} POS {rec['place_of_supply']}"]
        rng.shuffle(parts)
        L = parts + _items_block(rec, rng, "kv") + [f"Taxable {inr(rec['taxable_value'], rng)}",
                                                     *_tax_lines(rec, rng), f"{tot_label} {inr(rec['total'], rng)}"]
    elif template == "T4":  # csv-ish export
        L = [f"{n(rec['seller_name'])}", f"GST Reg No {rec['seller_gstin']}", f"Bill No,{rec['invoice_number']}",
             f"Bill Date,{d}", f"Party,{n(rec['buyer_name'])}", f"Party GSTIN,{buyer_g}",
             f"Supply State,{rec['place_of_supply']}", "Description,HSN,Qty,Rate,GST,Amount",
             *_items_block(rec, rng, "csv"), f"Taxable Amt,{inr(rec['taxable_value'], rng, False)}",
             *_tax_lines(rec, rng), f"{tot_label},{inr(rec['total'], rng, False)}"]
    elif template == "T5":  # HELD-OUT: Hinglish labels, totals before items
        L = [f"BIKRI BILL / SALE INVOICE", f"Vikreta (Seller): {n(rec['seller_name'])}",
             f"GST No: {rec['seller_gstin']}", f"Bill Sankhya: {rec['invoice_number']}", f"Dinank: {d}",
             f"Kharidar (Buyer): {n(rec['buyer_name'])} / GST No: {buyer_g}",
             f"Rajya (POS): {rec['place_of_supply']}",
             f"Kul Rashi (Total Payable): {inr(rec['total'], rng)}",
             f"Kar Yogya Rashi (Taxable): {inr(rec['taxable_value'], rng)}", *_tax_lines(rec, rng), "Vivaran:",
             *_items_block(rec, rng, "kv")]
    else:  # T6 HELD-OUT: e-way-bill-like dense lines, labels abbreviated differently
        tx = " ".join(_tax_lines(rec, rng))
        L = [f"DOC {rec['invoice_number']} DT {d} FROM {n(rec['seller_name'])}/{rec['seller_gstin']} "
             f"TO {n(rec['buyer_name'])}/{buyer_g} POS:{rec['place_of_supply']}"]
        L += [" ; ".join(f"{it['hsn']}:{it['description']} x{it['quantity']} @{inr(it['rate'], rng, False)} "
                         f"={inr(it['amount'], rng, False)}" for it in rec["line_items"])]
        L += [f"TXBL {inr(rec['taxable_value'], rng, False)} {tx} NET {inr(rec['total'], rng, False)}"]
    return "\n".join(L)


def public(rec: dict) -> dict:
    out = dict(rec)
    out["line_items"] = [{k: v for k, v in it.items() if not k.startswith("_")} for it in rec["line_items"]]
    return out


def make_split(n: int, templates: list[str], seed: int, noise=(0.0, 0.08)) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        rec = sample_record(rng)
        t = rng.choice(templates)
        text = render(rec, t, rng, rng.uniform(*noise))
        rows.append({"id": f"{seed}-{i}", "template": t, "text": text, "target": to_json(public(rec))})
    return rows


def main(out: str, train: int, val: int, test: int, hard: int):
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    seen = ["T1", "T2", "T3", "T4"]
    splits = {"train": make_split(train, seen, 1), "val": make_split(val, seen, 2),
              "test": make_split(test, seen, 3), "test_hard": make_split(hard, ["T5", "T6"], 4)}
    for name, rows in splits.items():
        (outdir / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                              encoding="utf-8")
        print(f"{name:10s} {len(rows)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--hard", type=int, default=300)
    a = ap.parse_args()
    main(a.out, a.train, a.val, a.test, a.hard)
