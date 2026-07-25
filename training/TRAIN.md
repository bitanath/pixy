# Falcon-H1 PII Masking Training

Fine-tune Falcon-H1 (Tiny 90M or 0.5B) to detect and replace personally identifiable information (PII) in text using QLoRA + LoRA, then convert to GGUF Q5_K_M for on-device inference.

## Overview

| Step | What | Output |
|------|------|--------|
| 1 | Load `ai4privacy/pii-masking-200k` (209k examples) | raw dataset |
| 2 | Augment with 10% identity examples (source == target) | ~230k examples |
| 3 | QLoRA fine-tune Falcon-H1 (exclude `conv1d`/`out_proj` from LoRA, skip `out_proj` from quantization) | LoRA adapter |
| 4 | Merge LoRA → base model | full HF model |
| 5 | `convert_hf_to_gguf.py` → FP16 GGUF | `.gguf` f16 |
| 6 | `quantize` → Q5_K_M | `.gguf` q5 |

## Dataset Format

```
source_text: "My email is john.doe@example.com and my phone is 555-0192."
target_text: "My email is [EMAIL] and my phone is [PHONENUMBER]."
```

For identity examples (no PII):
```
source_text: "The weather is nice today."
target_text: "The weather is nice today."
```

## Usage

Open `train_colab.ipynb` in Google Colab with a GPU runtime (T4 works for Tiny, A100 recommended for 0.5B).

Set `MODEL = "tiny"` or `MODEL = "0.5b"` in Cell 1, then run all cells.

## Falcon-H1 Constraints (handled)

| Constraint | Implementation |
|---|---|
| Exclude `conv1d` + `out_proj` from LoRA | `target_modules=["in_proj","x_proj","dt_proj"]` in LoraConfig |
| Skip `out_proj` from 4-bit quantization | `llm_int8_skip_modules=["out_proj"]` in BitsAndBytesConfig |
| `out_proj` must stay in fp16 (used in Mamba2 CUDA kernel) | Handled by the skip above |
| Architecture name `falcon_h1` | Supported by latest transformers + llama.cpp |

## Files

- `train_colab.ipynb` — end-to-end Colab notebook
- Trained GGUFs are downloaded directly from Colab

## Expected Output Sizes

| Model | FP16 GGUF | Q5_K_M GGUF |
|-------|-----------|-------------|
| Tiny 90M | ~180 MB | ~60 MB |
| 0.5B | ~1 GB | ~350 MB |
