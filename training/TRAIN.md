# Booster LoRA Training

Train LoRA adapters for Slack Block Kit JSX generation on Cloudflare Workers AI.

## Prerequisites

- Python 3.10+
- RunPod (or any GPU with 80GB+ VRAM for Qwen, 24GB+ for Llama)
- HuggingFace token with write access (for uploading)
- Cloudflare account + wrangler (for deploying)

## Setup

```bash
pip install transformers>=4.51.0 peft trl datasets bitsandbytes accelerate huggingface_hub
```

For Qwen3 specifically you need `transformers>=4.51.0`.

## Usage

### 1. Prepare dataset

```bash
python prepare_dataset.py
```

Generates `dataset.jsonl` with ~280 training examples.

### 2. Train LoRA

**Qwen3-30B-A3B** (needs ~80GB GPU):
```bash
python train_qwen.py
```
Output: `lora_qwen/`

**Llama-3.1-8B-Instruct** (needs ~24GB GPU):
```bash
python train_llama.py
```
Output: `lora_llama/`

### 3. Upload to HuggingFace

```bash
export HF_TOKEN=hf_your_token_here

# Upload Qwen adapter
python upload_to_hf.py --adapter lora_qwen --repo your-username/blocks-lora-qwen

# Upload Llama adapter
python upload_to_hf.py --adapter lora_llama --repo your-username/blocks-lora-llama
```

### 4. Deploy to Cloudflare

```bash
# Qwen
npx wrangler ai finetune create @cf/qwen/qwen3-30b-a3b-fp8 blocks-lora-qwen lora_qwen/

# Llama
npx wrangler ai finetune create @cf/meta/llama-3.1-8b-instruct-fast blocks-lora-llama lora_llama/
```

### 5. Use in your Worker

```typescript
const response = await env.AI.run(modelName, {
  messages: chatMessages,
  tools: toolDefs,
  lora: "blocks-lora-qwen",  // your finetune name or ID
  stream: true,
});
```

## Files

| File | Description |
|------|-------------|
| `prepare_dataset.py` | Generates training data from embedded SYSTEM.md + BLOCKS.markdown |
| `dataset.jsonl` | ~350 training examples in chat format |
| `train_qwen.py` | QLoRA training for Qwen3-30B-A3B |
| `train_llama.py` | QLoRA training for Llama-3.1-8B-Instruct |
| `upload_to_hf.py` | Upload adapter to HuggingFace Hub |
