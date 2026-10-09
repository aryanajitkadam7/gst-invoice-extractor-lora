"""LoRA / QLoRA supervised fine-tuning with a plain PyTorch loop (transformers + peft only).

    python -m invoicex.train_lora --model Qwen/Qwen2.5-0.5B-Instruct --quant 4bit --epochs 2

Loss is computed only on the assistant's JSON. Train/val loss is logged to
reports/train_log.jsonl and the adapter is saved to outputs/<run_name>/adapter.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from .prompting import encode_for_training

ROOT = Path(__file__).resolve().parents[2]


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def collate(batch, pad_id):
    ids = pad_sequence([torch.tensor(b["input_ids"]) for b in batch], batch_first=True, padding_value=pad_id)
    labels = pad_sequence([torch.tensor(b["labels"]) for b in batch], batch_first=True, padding_value=-100)
    attn = pad_sequence([torch.ones(len(b["input_ids"]), dtype=torch.long) for b in batch], batch_first=True)
    return ids, attn, labels


def load_model(name: str, quant: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {}
    if quant == "4bit":
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32 if quant == "none" else None, **kw)
    if quant == "none":
        model.to(device)
    return tok, model


def add_lora(model, quant: str, r: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if quant == "4bit":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, trainable, total


@torch.no_grad()
def eval_loss(model, data, bs, pad_id, device, amp):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(data), bs):
        ids, attn, labels = (t.to(device) for t in collate(data[i : i + bs], pad_id))
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=amp):
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
        tot += out.loss.item() * len(ids)
        n += len(ids)
    model.train()
    return tot / max(n, 1)


def train(model_name: str, quant: str = "4bit", epochs: int = 2, lr: float = 2e-4, micro_bs: int = 4,
          grad_accum: int = 4, r: int = 16, alpha: int = 32, dropout: float = 0.05, max_len: int = 1024,
          train_path=None, val_path=None, run_name: str = "qwen2.5-0.5b-lora", max_steps: int | None = None,
          tokenizer=None, model=None, seed: int = 42, out_root: Path = ROOT) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    if model is None:
        tokenizer, model = load_model(model_name, quant if device == "cuda" else "none", device)
    model, n_trainable, n_total = add_lora(model, quant if device == "cuda" else "none", r, alpha, dropout)

    train_rows = load_rows(train_path or ROOT / "data" / "train.jsonl")
    val_rows = load_rows(val_path or ROOT / "data" / "val.jsonl")
    tr = [encode_for_training(tokenizer, x["text"], x["target"], max_len) for x in train_rows]
    va = [encode_for_training(tokenizer, x["text"], x["target"], max_len) for x in val_rows]
    pad_id = tokenizer.pad_token_id

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(tr) / (micro_bs * grad_accum))
    total_steps = max_steps or steps_per_epoch * epochs
    warmup = max(1, int(0.03 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    log_path = Path(out_root) / "reports" / "train_log.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    log = open(log_path, "w")
    step, t0 = 0, time.time()
    model.train()
    done = False
    for epoch in range(epochs):
        order = list(range(len(tr)))
        random.shuffle(order)
        # length-bucketed micro-batches cut padding waste
        batches = [order[i : i + micro_bs] for i in range(0, len(order), micro_bs)]
        running = 0.0
        for bi, b in enumerate(batches):
            ids, attn, labels = (t.to(device) for t in collate([tr[i] for i in b], pad_id))
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss / grad_accum
            scaler.scale(loss).backward()
            running += loss.item()
            if (bi + 1) % grad_accum == 0 or bi == len(batches) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 10 == 0 or step == 1:
                    rec = {"step": step, "epoch": epoch, "train_loss": round(running, 4),
                           "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
                    log.write(json.dumps(rec) + "\n"); log.flush()
                    print(rec)
                running = 0.0
                if max_steps and step >= max_steps:
                    done = True
                    break
        vl = eval_loss(model, va, micro_bs, pad_id, device, amp)
        rec = {"step": step, "epoch": epoch, "val_loss": round(vl, 4), "elapsed_s": round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(rec)
        if done:
            break
    log.close()

    out_dir = Path(out_root) / "outputs" / run_name / "adapter"
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    summary = {"model": model_name, "quant": quant, "device": device, "epochs": epochs, "steps": step,
               "lr": lr, "effective_batch": micro_bs * grad_accum, "lora_r": r, "lora_alpha": alpha,
               "trainable_params": n_trainable, "total_params": n_total,
               "trainable_pct": round(100 * n_trainable / n_total, 3), "train_minutes": round((time.time() - t0) / 60, 1),
               "final_val_loss": vl, "adapter_dir": str(out_dir)}
    (Path(out_root) / "reports" / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--quant", choices=["4bit", "none"], default="4bit")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--run-name", default="qwen2.5-0.5b-lora")
    a = ap.parse_args()
    print(train(a.model, a.quant, a.epochs, a.lr, a.micro_bs, a.grad_accum, a.r, max_steps=a.max_steps,
                run_name=a.run_name))
