# GST Invoice → JSON: QLoRA fine-tuning a small LLM for document extraction

Indian businesses process millions of GST invoices. They arrive as scanned PDFs, WhatsApp photos and ERP exports, each with its own layout. This project fine-tunes **Qwen2.5-0.5B-Instruct with QLoRA** to turn noisy, OCR-style invoice text into validated, structured JSON:

- GSTINs
- invoice number and date
- parties and place of supply
- line items with HSN codes
- CGST, SGST or IGST
- total

It compares the fine-tuned model against a **regex/rules baseline** and the **same model with prompting only**. Generalisation is tested on **invoice layouts never seen in training**.

```
TAX INVOICE                                   {"invoice_number":"INV/24-25/0012",
Om Enterprises                                 "invoice_date":"2024-06-05",
GSTIN: 27AAACO1234F1Z5            ──────▶       "seller_gstin":"27AAACO1234F1Z5",
Invoice No: INV/24-25/0012                     "buyer_gstin":null, ...
Date: 05/06/2024  ...                          "cgst":90.0,"sgst":90.0,"total":1180.0}
```

## Why this setup

- **Synthetic but realistic data** (`src/invoicex/datagen.py`). Real invoices hold private commercial data, so they can't be published. The generator produces:
  - **valid GSTINs** with the official mod-36 checksum
  - **correct GST maths**: intra-state CGST+SGST vs inter-state IGST, at 5/12/18/28% item rates
  - Indian digit grouping (`1,23,456.00`), `Rs.`/`INR`/`₹` prefixes and 7 date formats
  - B2C buyers without a GSTIN
  - **OCR noise** in names (`O→0`, `l→1`, dropped spaces)
- **Layout generalisation is tested, not assumed.** Train, val and test use layouts T1–T4: a labelled header, fixed-width columns, a key-value stream and a CSV export. The `test_hard` split uses **T5 (Hinglish labels: *Bill Sankhya*, *Dinank*, *Kul Rashi*) and T6 (dense e-way-bill-style lines)**, which never appear in training.
- **Strict metrics** (`src/invoicex/metrics.py`):
  - JSON validity
  - per-field accuracy (amounts within ±0.01, ISO dates exact)
  - line-item F1, matched on (HSN, qty, amount)
  - **document exact match**: every field and every item correct, the bar for straight-through processing
  - checksum validity of predicted GSTINs, to catch hallucinated IDs

## Results

All rows are evaluated on the same 300-invoice `test` split (seen layouts) and 300-invoice `test_hard` split (unseen layouts).

| System | Split | JSON valid | Field acc. | Line-item F1 | **Doc exact** | ms/doc |
|---|---|---|---|---|---|---|
| Rule-based (regex, written for T1–T4) | seen layouts | 1.000 | 0.981 | 1.000 | **0.797** | 0.3 (CPU) |
| Rule-based (regex, written for T1–T4) | unseen layouts | 1.000 | 0.458 | 0.503 | **0.000** | 0.3 (CPU) |
| Qwen2.5-0.5B-Instruct, zero-shot | both | *pending: run the notebook* | | | | |
| Qwen2.5-0.5B-Instruct, 1-shot | both | *pending* | | | | |
| **Qwen2.5-0.5B + QLoRA (this repo)** | both | *pending* | | | | |

**What the baseline shows (measured).** On layouts it was written for, regex gets 98% of fields right but only 80% of documents fully right. The misses are OCR-noised party names (`Electr0nics`) that regex can't repair. On the two unseen layouts it **extracts 0% of documents correctly**: invoice number, date, names, taxable value and total all fail as soon as the labels change. That brittleness is the reason to use a learned extractor.

> **The LLM rows are deliberately blank.** They come from a GPU run of `notebooks/colab_train_eval.ipynb` on a free Colab T4, which takes about 60–75 minutes. Running it writes `reports/results_*.json`, `reports/train_log.jsonl` and `reports/loss_curve.png`, and `python -m invoicex.report` regenerates this table. Nothing here is estimated.

## Training setup

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Method | QLoRA: 4-bit NF4 base with double quantisation, fp16 compute |
| LoRA | r = 16, α = 32, dropout 0.05, on q/k/v/o + gate/up/down projections |
| Data | 2,000 train / 200 val invoices, chat format, **loss on the JSON answer only** (prompt tokens masked) |
| Optimiser | AdamW, lr 2e-4, 3% warm-up + cosine decay, effective batch 16, grad-clip 1.0, 2 epochs |
| Decoding | greedy, max 512 new tokens; the output is parsed with a brace-matching JSON extractor |

The training loop (`src/invoicex/train_lora.py`) is a plain PyTorch loop over `transformers` + `peft`, so every step is visible: label masking, mixed precision with `GradScaler`, gradient accumulation, and a scheduler.

`tests/test_smoke_train.py` checks the full train → save adapter → generate path **on CPU**. It uses a tiny randomly initialised Qwen2 model and asserts that the loss goes down. CI therefore catches training-code regressions without a GPU. This test already caught a real bug: `apply_chat_template` changed its return type in transformers 5.

## Run it

```bash
pip install -r requirements.txt
make rules        # regex baseline, CPU, < 1 s
make test         # 7 tests incl. the CPU training smoke test (needs torch, transformers, peft)
```

GPU parts: open `notebooks/colab_train_eval.ipynb` in Colab, select a T4 runtime, and run all cells. Or locally:

```bash
pip install -r requirements-gpu.txt
make base         # prompted baselines
make train        # QLoRA fine-tuning
make lora         # evaluate the adapter
make report
```

The data is deterministic (fixed seeds). `make data` regenerates exactly the committed `data/*.jsonl`.

## Layout

```
src/invoicex/
  schema.py        target schema, GSTIN checksum, robust JSON parsing
  datagen.py       synthetic GST invoice generator (6 layouts, OCR noise, correct tax maths)
  rules.py         regex baseline        eval_rules.py  its evaluation
  prompting.py     chat prompts shared by training and inference
  train_lora.py    QLoRA / LoRA fine-tuning loop
  infer.py         batched generation + scoring (base zero/1-shot, fine-tuned)
  metrics.py       field accuracy, line-item F1, doc exact match, GSTIN validity
  report.py        builds reports/results.md
notebooks/colab_train_eval.ipynb   one-click reproduction on a free T4
```

## Limitations

- The data is synthetic. Real scans add OCR errors inside numbers and GSTINs, multi-page tables and handwriting. The next step is to evaluate on a small set of real, anonymised invoices.
- The input is text after OCR. A full system would add an OCR stage (e.g. Tesseract or PaddleOCR) or a vision-language model.
