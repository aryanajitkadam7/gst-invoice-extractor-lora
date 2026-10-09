"""CPU smoke test: a 2-layer randomly initialised Qwen2 model + LoRA trains for a few steps
and the loss goes down, then greedy generation runs. Proves the training/inference code paths
work end-to-end without a GPU or a model download."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from invoicex.datagen import make_split

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}</s>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    chars = sorted(set("".join(chr(i) for i in range(32, 127)) + "₹\n"))
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, **{c: i + 3 for i, c in enumerate(chars)}}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="</s>", unk_token="<unk>")
    tok.chat_template = CHAT_TEMPLATE
    return tok


def test_lora_training_loop_runs_and_learns(tmp_path: Path):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from invoicex.infer import generate
    from invoicex.train_lora import train

    for name, n, seed in [("train", 16, 1), ("val", 4, 2)]:
        rows = make_split(n, ["T1"], seed)
        (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    tok = tiny_tokenizer()
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    model = Qwen2ForCausalLM(cfg)
    summary = train("tiny", quant="none", epochs=3, lr=3e-3, micro_bs=4, grad_accum=1, r=8, alpha=16,
                    max_len=3000, train_path=tmp_path / "train.jsonl", val_path=tmp_path / "val.jsonl",
                    run_name="smoke", tokenizer=tok, model=model, out_root=tmp_path)
    assert 0 < summary["trainable_pct"] < 50
    log = [json.loads(l) for l in (tmp_path / "reports" / "train_log.jsonl").read_text().splitlines()]
    losses = [r["train_loss"] for r in log if "train_loss" in r]
    assert losses[-1] < losses[0], losses
    assert (tmp_path / "outputs" / "smoke" / "adapter" / "adapter_config.json").exists()

    outs, _ = generate(summary_model(model), tok, ["TAX INVOICE\nTest"], bs=1, max_new_tokens=8)
    assert len(outs) == 1


def summary_model(m):
    return m.eval()
