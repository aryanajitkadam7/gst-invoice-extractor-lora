"""Prompt construction shared by training and inference (they must match exactly)."""
from __future__ import annotations

import json

from .schema import JSON_TEMPLATE

SYSTEM = (
    "You extract structured data from Indian GST invoices. The text may come from OCR and contain noise. "
    "Return ONLY one JSON object with exactly these keys:\n"
    f"{json.dumps(JSON_TEMPLATE)}\n"
    "Rules: dates as YYYY-MM-DD; amounts as plain numbers without commas or currency symbols; "
    "fix obvious OCR errors in names (e.g. 0->o, 1->l); buyer_gstin is null for unregistered buyers; "
    "use 0.0 for taxes that are not charged."
)

# one worked example for the prompted (few-shot) baseline; never used in fine-tuning
FEWSHOT_TEXT = """TAX INVOICE
Om Enterprises
GSTIN: 27AAACO1234F1Z5
Invoice No: INV/24-25/0012
Date: 05/06/2024
Bill To: Rao Stores
Buyer GSTIN: Unregistered
Place of Supply: Maharashtra
| # | Description | HSN | Qty | Rate | GST | Amount |
| 1 | LED Bulb 9W | 8539 | 10 | 100.00 | 18% | 1,000.00 |
Taxable Value: Rs.1,000.00
CGST: 90.00
SGST: 90.00
Grand Total: ₹1,180.00"""
FEWSHOT_JSON = json.dumps({
    "invoice_number": "INV/24-25/0012", "invoice_date": "2024-06-05", "seller_name": "Om Enterprises",
    "seller_gstin": "27AAACO1234F1Z5", "buyer_name": "Rao Stores", "buyer_gstin": None,
    "place_of_supply": "Maharashtra",
    "line_items": [{"description": "LED Bulb 9W", "hsn": "8539", "quantity": 10, "rate": 100.0, "amount": 1000.0}],
    "taxable_value": 1000.0, "cgst": 90.0, "sgst": 90.0, "igst": 0.0, "total": 1180.0,
}, separators=(",", ":"))


def messages(text: str, target: str | None = None, fewshot: bool = False) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    if fewshot:
        msgs += [{"role": "user", "content": FEWSHOT_TEXT}, {"role": "assistant", "content": FEWSHOT_JSON}]
    msgs.append({"role": "user", "content": text})
    if target is not None:
        msgs.append({"role": "assistant", "content": target})
    return msgs


def encode_for_training(tokenizer, text: str, target: str, max_len: int = 1024) -> dict:
    """input_ids + labels where only the assistant's JSON (and EOS) is learned."""
    # render to a string first, then tokenize: identical to what inference does, and
    # independent of whether apply_chat_template(tokenize=True) returns a list or a BatchEncoding
    prompt = prompt_for_inference(tokenizer, text)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    ids = (list(prompt_ids) + target_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + target_ids)[:max_len]
    return {"input_ids": ids, "labels": labels}


def prompt_for_inference(tokenizer, text: str, fewshot: bool = False) -> str:
    return tokenizer.apply_chat_template(messages(text, fewshot=fewshot), add_generation_prompt=True, tokenize=False)
