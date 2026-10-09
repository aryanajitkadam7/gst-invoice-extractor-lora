"""Batched greedy inference + scoring for the base model (prompted) and the fine-tuned adapter.

    python -m invoicex.infer --mode base      # zero-shot + 1-shot prompted baseline
    python -m invoicex.infer --mode lora --adapter outputs/qwen2.5-0.5b-lora/adapter

Writes reports/results_<tag>.json and reports/predictions_<tag>_<split>.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .metrics import load_gold, score
from .prompting import prompt_for_inference
from .schema import parse_json

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def generate(model, tok, texts, fewshot=False, bs=8, max_new_tokens=512):
    device = next(model.parameters()).device
    tok.padding_side = "left"
    outs, n_tokens = [], 0
    for i in range(0, len(texts), bs):
        prompts = [prompt_for_inference(tok, t, fewshot) for t in texts[i : i + bs]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        n_tokens += int((new != tok.pad_token_id).sum())
        outs += tok.batch_decode(new, skip_special_tokens=True)
    return outs, n_tokens


def run(model, tok, tag: str, fewshot=False, splits=("test", "test_hard"), limit=None, bs=8):
    results = {}
    for split in splits:
        rows = load_gold(ROOT / "data" / f"{split}.jsonl")[:limit]
        t0 = time.perf_counter()
        raw, n_tok = generate(model, tok, [r["text"] for r in rows], fewshot, bs)
        secs = time.perf_counter() - t0
        preds = [parse_json(x) for x in raw]
        res = score(preds, [json.loads(r["target"]) for r in rows])
        res.update({"seconds_per_doc": round(secs / len(rows), 3), "gen_tokens_per_s": round(n_tok / secs, 1)})
        results[split] = res
        with open(ROOT / "reports" / f"predictions_{tag}_{split}.jsonl", "w", encoding="utf-8") as f:
            for r, x in zip(rows, raw):
                f.write(json.dumps({"id": r["id"], "template": r["template"], "raw": x}, ensure_ascii=False) + "\n")
        print(f"[{tag}] {split}: doc_exact={res['doc_exact']:.3f} field_acc={res['field_accuracy_mean']:.3f} "
              f"item_f1={res['line_item_f1']:.3f} json_valid={res['json_valid']:.3f}")
    (ROOT / "reports" / f"results_{tag}.json").write_text(json.dumps(results, indent=2))
    return results


def load(model_name, adapter=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(adapter or model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval(), tok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--mode", choices=["base", "lora"], required=True)
    ap.add_argument("--adapter", default=str(ROOT / "outputs" / "qwen2.5-0.5b-lora" / "adapter"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    if a.mode == "base":
        m, t = load(a.model)
        run(m, t, "base_zeroshot", fewshot=False, limit=a.limit, bs=a.bs)
        run(m, t, "base_1shot", fewshot=True, limit=a.limit, bs=a.bs)
    else:
        m, t = load(a.model, a.adapter)
        run(m, t, "lora", limit=a.limit, bs=a.bs)
