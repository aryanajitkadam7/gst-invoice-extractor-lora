export PYTHONPATH := src

data:           ## regenerate the synthetic dataset (deterministic)
	python -m invoicex.datagen --out data

rules:          ## rule-based baseline -> reports/results_rules.json
	python -m invoicex.eval_rules

base:           ## prompted base model (GPU recommended)
	python -m invoicex.infer --mode base

train:          ## QLoRA fine-tune (GPU)
	python -m invoicex.train_lora --quant 4bit --epochs 2

lora:           ## evaluate the fine-tuned adapter
	python -m invoicex.infer --mode lora

report:
	python -m invoicex.report

test:
	python -m pytest -q
