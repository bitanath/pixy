# HOW-TO-LLM.md

# Pixy Developer / Agent Guide

> Developer and agent instructions for building, modifying, and extending Pixy.

## Overview

Pixy is a dependency-free LLM inference framework written primarily in Zig.

The project targets extremely constrained devices, especially Apple Watch,
and prioritizes:

- Low memory usage.
- Predictable execution.
- Cross-platform compilation.
- SIMD acceleration.
- Minimal runtime dependencies.

The inference runtime must remain independent from Python ML frameworks.

Python is only used for training, experimentation, and weight conversion.

---

# Repository Structure

```
framework/
    Zig LLM inference runtime.

visual/
    Zig CNN inference runtime for touch prompt recognition.

training/
    Python/PyTorch training pipeline.

performix/
    Performance analysis tooling.

pixy Watch App/
    SwiftUI watchOS application.

OPTIMIZATIONS.md
    ARM SIMD and performance notes.

README.rst
    User-facing documentation.
```

---

# Core Runtime

Location:

```
framework/core/
```

Important files:

```
main.zig
    Runtime entry point.

index.zig
    Public exports.

mats.zig
    Matrix operations and kernels.

ops.zig
    Tensor primitives.

dequant.zig
    Quantization/dequantization.

accelerate/
    SIMD and architecture-specific optimizations.
```

The runtime implements:

- Transformer execution.
- Quantized inference.
- Tensor operations.
- Matrix multiplication.
- Memory-efficient kernels.

Do not introduce:

- PyTorch.
- TensorFlow.
- ONNX Runtime.
- libtorch.
- External ML dependencies.

---

# Build System

Pixy uses Zig's native build system.

Requirements:

- Zig stable release.
- Xcode for watchOS targets.
- Python only for training.

## Installing Zig

macOS:

```bash
brew install zig
```

Linux:

Download Zig from:

```
https://ziglang.org/download/
```

Verify:

```bash
zig version
```

---

# Building

## Core Runtime

Development build:

```bash
cd framework
zig build
```

Optimized build:

```bash
zig build -Doptimize=ReleaseFast
```

Deployment build:

```bash
cd framework/scripts
./deploy.sh
```

Outputs:

```
framework/outputs/lib
framework/outputs/bin
```

Build targets:

- watchOS.
- watchOS Simulator.
- macOS.
- Linux.

Generated binaries include:

```
llm-linux-piimask
llm-macos-piimask
```

These demonstrate server-side inference use cases:

- Log processing.
- PII masking.
- Privacy-aware pipelines.

---

# Visual CNN Runtime

Location:

```
visual/
```

Build:

```bash
cd visual/scripts
./deploy.sh
```

Outputs:

```
visual/outputs/lib
```

Targets:

- watchOS.
- watchOS Simulator.

The CNN converts Apple Watch drawings into prompts:

```
Drawing
   |
CNN
   |
Text Prompt
   |
LLM
```

Training happens in:

```
training/
```

Inference happens in:

```
visual/
```

Do not mix training dependencies into runtime code.

---

# Training Pipeline

Location:

```
training/
```

Contains:

- PyTorch notebooks.
- Gesture datasets.
- Model checkpoints.

Typical workflow:

1. Train model using Python.
2. Export weights.
3. Convert weights.
4. Load in Zig runtime.

Runtime must not require Python.

---

# Performix

Location:

```
performix/
```

Build:

```bash
cd performix
./deploy.sh
```

Example:

```bash
pxy performix/reports/code_hotspots_3ac75e10feeb.zip
```

or:

```bash
pxy-linux performix/reports/code_hotspots_3ac75e10feeb.zip
```

Included reports:

```
performix/reports/

code_hotspots_3ac75e10feeb.zip
code_hotspots_835a0fc8224b.zip
code_hotspots_a2419b38d3b0.zip
system_utilization_c7e6f3f5707c.zip
```

Supported reports:

- Code hotspots.
- System utilization.

Known limitation:

Some Performix integrations require unavailable PMU counters:

```
tool_integrations.neoprof.INSUFFICIENT_PMU_COUNTERS
```

---

# Performance Rules

Pixy targets constrained ARM devices.

Avoid:

- Allocations inside inference loops.
- Large temporary buffers.
- Duplicate model copies.
- Scalar replacements for SIMD code.

Prefer:

```zig
@Vector
@mulAdd
@reduce
@shuffle
@select
```

See:

```
OPTIMIZATIONS.md
```

for detailed NEON mappings.

---

# Quantization Rules

Pixy relies heavily on compact quantized representations.

When modifying:

- Weight formats.
- Tensor layouts.
- Dequantization.
- Matrix kernels.

Verify:

- Existing checkpoints remain compatible.
- Alignment assumptions remain valid.
- Memory usage does not increase significantly.

---

# Adding Features

Before adding new operators or kernels:

1. Understand existing tensor layouts.
2. Check if SIMD implementation is required.
3. Preserve cross-platform builds.
4. Add tests.
5. Benchmark changes.

Avoid unnecessary abstractions in hot paths.

---

# Coding Guidelines

## Zig

Prefer:

- Explicit memory ownership.
- Compile-time evaluation.
- Small optimized functions.
- SIMD primitives.

Avoid:

- Hidden allocations.
- Runtime-heavy abstractions.
- External dependencies.

## Swift

Keep Swift focused on:

- UI.
- User interaction.
- Calling compiled libraries.

Core ML logic belongs in Zig.

## Python

Python is only for:

- Training.
- Validation.
- Weight conversion.

---

# Agent Instructions

Before modifying code:

Inspect:

```
framework/core/
visual/core/
OPTIMIZATIONS.md
```

Do not:

- Replace Zig with another runtime.
- Add heavyweight ML frameworks.
- Remove SIMD optimizations.
- Assume desktop-level memory.
- Break watchOS/Linux/macOS targets.

Prefer:

- Small targeted changes.
- Benchmark-driven optimization.
- Preserving existing interfaces.
- Documenting non-obvious optimizations.

The goal of Pixy:

> Run useful ML inference on hardware that was never designed for it.
```