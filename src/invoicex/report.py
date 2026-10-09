"""Collect every results_*.json into reports/results.md (the table used in the README).

    python -m invoicex.report
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORDER = [("rules", "Rule-based (regex)"), ("base_zeroshot", "Qwen2.5-0.5B-Instruct, zero-shot"),
         ("base_1shot", "Qwen2.5-0.5B-Instruct, 1-shot"), ("lora", "Qwen2.5-0.5B + LoRA (ours)")]


def main():
    lines = ["| System | Split | JSON valid | Field acc. | Line-item F1 | Doc exact | ms/doc |", "|---|---|---|---|---|---|---|"]
    for tag, label in ORDER:
        p = ROOT / "reports" / f"results_{tag}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        for split, name in (("test", "seen layouts"), ("test_hard", "unseen layouts")):
            if split in r:
                x = r[split]
                ms = x["seconds_per_doc"] * 1000 if "seconds_per_doc" in x else x.get("latency_ms_per_doc", 0)
                lines.append(f"| {label} | {name} | {x['json_valid']:.3f} | {x['field_accuracy_mean']:.3f} | "
                             f"{x['line_item_f1']:.3f} | {x['doc_exact']:.3f} | {ms:.1f} |")
    s = ROOT / "reports" / "train_summary.json"
    if s.exists():
        t = json.loads(s.read_text())
        lines.append(f"\nTraining: {t['steps']} steps, {t['train_minutes']} min on {t['device']}, "
                     f"{t['trainable_params']:,} trainable params ({t['trainable_pct']}%), final val loss {t['final_val_loss']:.4f}")
    md = "\n".join(lines) + "\n"
    (ROOT / "reports" / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
