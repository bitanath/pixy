# LLM Model Benchmark

**Date:** July 25, 2026
**Hardware:** Mac15,7 — Apple M3 Pro (12 cores: 6P+6E), 18 GB RAM
**OS:** macOS 26.5.2 arm64
**Method:** Model embedded via `@embedFile`, single-turn `generate_conversation` with prompt *"What is the capital of India?"*, measured via `mach_absolute_time` + `task_info` RSS.

| Model | File Size | Binary Size | Gen Time | Gen Time (opt) | Speedup | Memory Δ | Output |
|-------|-----------|-------------|----------|----------------|---------|----------|--------|
| tiny.gguf | 64 MB | 65 MB | 685 ms | **602 ms** | **-12%** | 64 MB | "The capital of India is New Delhi." |
| small.gguf | 94 MB | 95 MB | 322 ms | **326 ms** | ~0% | 94 MB | "Mumbai." |
| large.gguf | 354 MB | 357 MB | 3,684 ms | **3,271 ms** | **-11%** | 332 MB | "The capital of India is New Delhi." |

### Optimization (July 25, 2026)
- `@Vector(4, f32)` → `@Vector(8, f32)` in all matmul hot paths (4 functions in `mats.zig`, 2 in `dequant.zig`)
- `f64` → `f32` in RoPE, softmax, SwiGLU, RMS norm, and attention accumulation across both `transformerLlama` and `transformerPrefillLlama`
- Struct field type changes: `Config.rms_norm_eps`, `RunState.attn_scale`, `RunState.inv_dim`, `RunState.inv_head_size` (`f64` → `f32`)

### Notes
- small.gguf is faster than tiny despite being larger — likely a different/more efficient architecture, but quality regressed (wrong answer to a basic geography question).
- large.gguf is the most accurate but 3.3s generation and 332 MB memory make it impractical for watch deployment.
- tiny.gguf remains the best balance for the watch app: correct, fast (602 ms), 65 MB binary.
