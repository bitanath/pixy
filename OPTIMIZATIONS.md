# ARM NEON Optimizations in LLM Inference

How Zig's built-in SIMD primitives deliver high-performance NEON machine code without intrinsics, inline assembly, or compiler flags.

---

## 1. `@Vector` — Zero-Effort SIMD

**Problem:** Process 8 floats at once instead of one at a time.

| Approach      | Code                                                      |
|---------------|-----------------------------------------------------------|
| Scalar Zig    | `for (0..8) \|i\| out[i] = a[i] + b[i]`                  |
| Optimized Zig | `const out: @Vector(8, f32) = a + b`                      |

| Compiler emits                                                   |
|------------------------------------------------------------------|
| `LD1.4S` × 2 → `FADD.4S` × 2 → `ST1.4S` × 2                     |

> **Why it matters:** Changing `[8]f32` to `@Vector(8, f32)` is the only edit needed. The compiler generates NEON register operations for all 8 elements. No intrinsics, no attributes, no special headers.

---

## 2. `@mulAdd` — Fused Multiply-Add

**Problem:** Accumulate product of two vectors without intermediate rounding.

| Approach      | Code                                                      |
|---------------|-----------------------------------------------------------|
| Scalar Zig    | `for (0..8) \|i\| acc[i] += x[i] * w[i]`                 |
| Optimized Zig | `acc = @mulAdd(@Vector(8, f32), x, w, acc)`               |

| Compiler emits                                                   |
|------------------------------------------------------------------|
| `FMLA.4S` × 2                                                     |

> **Why it matters:** Each `FMLA` does `acc += a × b` in one instruction with one rounding step. The scalar version compiles to separate `FMUL` + `FADD` per element — 16 instructions vs 2.

---

## 3. `@reduce(.Add)` — Horizontal Vector Sum

**Problem:** Sum all 8 lanes of a SIMD vector into a single scalar.

| Approach      | Code                                                      |
|---------------|-----------------------------------------------------------|
| Scalar Zig    | `var s: f32 = 0;`<br>`for (0..8) \|i\| s += v[i]`        |
| Optimized Zig | `const s = @reduce(.Add, v)`                              |

| Compiler emits                                                   |
|------------------------------------------------------------------|
| `FADDP.4S` → `FADDP.4S` → `SMOV` (3 instructions)                |

> **Why it matters:** The scalar loop is 8 `FADD` + loop overhead. The NEON version is 3 instructions. Zig exposes horizontal operations as builtins — no `vpaddq_f32` to look up.

---

## 4. `@shuffle` — Arbitrary Element Gather

**Problem:** Extract 4 specific bytes from a 16-byte vector.

| Approach      | Code                                                      |
|---------------|-----------------------------------------------------------|
| Scalar Zig    | `for (0..4) \|i\| c[i] = vec[idx[i]]`<br>`where idx = {0, 1, 4, 5}` |
| Optimized Zig | `const c = @shuffle(u8, vec, undefined,`<br>`    @Vector(4, i32){ 0, 1, 4, 5 })` |

| Compiler emits                                                   |
|------------------------------------------------------------------|
| `TBL.16B v.4B, {v.16B}, v.4B`                                    |

> **Why it matters:** `TBL` gathers 4 bytes from arbitrary positions in 1 instruction. The scalar version is 4 loads + 4 index calculations + loop. Zig's `@shuffle` mirrors the hardware operation directly.

---

## 5. `@select` — Branchless Conditional

**Problem:** Conditionally add 16 to nibble values without a branch.

| Approach      | Code                                                      |
|---------------|-----------------------------------------------------------|
| Scalar Zig    | `for (0..16) \|i\|`<br>`    q[i] = cond[i] ? lo[i] + 16 : lo[i]` |
| Optimized Zig | `const add = @select(u8, cond, sixteen, zero);`<br>`const q = lo \| add` |

| Compiler emits                                                   |
|------------------------------------------------------------------|
| `CMEQ.16B` → `BSL.16B` → `ORR.16B`                               |

> **Why it matters:** Scalar conditionals branch — a mispredict costs ~12 cycles. `@select` compiles to branchless NEON `BSL` with fixed latency regardless of data pattern. The Zig code reads as straightforward logic; the compiler handles the SIMD lowering.

---

## 6. `@Vector(N, i8)` → `@Vector(N, f32)` — SIMD Type Conversion

**Problem:** Load 8 signed 8-bit weights and convert to f32 for dot product.

| Approach      | Code                                                      |
|---------------|-----------------------------------------------------------|
| Scalar Zig    | `for (0..8) \|j\|`<br>`    w[j] = @as(f32, @as(i32, src[i + j]))` |
| Optimized Zig | `const chunk: @Vector(8, i8) = src[i..][0..8].*;`<br>`const w: @Vector(8, f32) = @floatFromInt(@intCast(chunk))` |

| Compiler emits                                                   |
|------------------------------------------------------------------|
| `LD1.8B` → `SSHLL.8H` (8→16) → `SSHLL.8S` (16→32) → `SCVTF.4S` × 2 |

> **Why it matters:** 5 NEON instructions convert 8 weights. The scalar version takes 24 instructions (8 loads + 8 sign-extends + 8 float converts). The type chain reads left-to-right: load `@Vector(8, i8)` → widen to `i32` → convert to `f32`.

---

## 7. `@Vector(16, u8)` Bitwise — SIMD Nibble Extraction

**Problem:** Extract low/high nibbles from 16 packed bytes.

| Approach      | Code                                                      |
|---------------|-----------------------------------------------------------|
| Scalar Zig    | `for (0..16) \|i\| {`<br>`    lo[i] = ql[i] & 0x0f;`<br>`    hi[i] = ql[i] >> 4 }` |
| Optimized Zig | `const lo = ql & @as(@Vector(16, u8), @splat(0x0f));`<br>`const hi = ql >> @as(@Vector(16, u8), @splat(4));` |

| Compiler emits                                                   |
|------------------------------------------------------------------|
| `LD1.16B` → `AND.16B` → `USHR.16B` (3 instructions)              |

> **Why it matters:** 3 NEON instructions vs ~50 scalar (16 loads + 16 AND + 16 shifts + loop). Zig's `&` and `>>` operators work on vectors exactly as they do on scalars — the compiler auto-selects the SIMD instruction.

---

## 8. `inline for` — Compile-Time Unrolling

**Problem:** Eliminate loop overhead from a known-trip-count loop.

| Approach      | Code                                                      |
|---------------|-----------------------------------------------------------|
| Scalar Zig    | `for (0..4) \|b\| acc += x[b] * w[b]`                    |
| Optimized Zig | `inline for (0..4) \|b\| acc += x[b] * w[b]`             |

| Compiler emits                                                   |
|------------------------------------------------------------------|
| No loop at runtime — body duplicated 4× at compile time           |

> **Why it matters:** Each runtime iteration costs ~3 instructions (increment, compare, branch). `inline for` eliminates them at compile time. No `#pragma unroll`, no manual code duplication.

---

## 9. `@Vector` Load/Store — Bulk Memory Access

**Problem:** Load/store N values from/to memory in one expression.

| Approach      | Code                                                      |
|---------------|-----------------------------------------------------------|
| Scalar Zig    | `for (0..8) \|j\| v[j] = src[i + j]`                     |
| Optimized Zig | `const v: @Vector(8, f32) = src[i..][0..8].*;`<br>`dst[j..][0..8].* = v` |

| Compiler emits                                                   |
|------------------------------------------------------------------|
| `LD1.4S` × 2 / `ST1.4S` × 2                                      |

> **Why it matters:** The slice dereference `src[i..][0..8].*` validates at compile time that the slice length matches the vector width. Out-of-bounds is caught during compilation, not at runtime.

---

## 10. Putting It All Together — `matmulByteLocal`

Annotated Q8_0 matmul kernel with each optimization mapped to its NEON instruction:

```
pub fn matmulByteLocal(
        out: []f32, x: []const f32,
        local_u8: []const u8, local_i8: []const i8,
        rows: usize, cols: usize, row_size: usize) void
{
    const nb = cols >> 5;                           // 128 tiles of 32 columns
    const rows4 = rows & ~@as(usize, 3);             // round down to multiple of 4

    for (0..rows4 / 4) |block| {
        const i = block * 4;
        var s0: f32 = 0; var s1: f32 = 0;
        var s2: f32 = 0; var s3: f32 = 0;

        for (0..nb) |_| {
            // Load fp16 scale (scalar — table lookup)
            const d0 = fp16ToFp32(u16_at(local_u8, bo0));

            // Accumulate 32-column dot product in SIMD
            var dot0: @Vector(8, f32) = @splat(0.0);        // [1] @Vector accumulator
            inline for (0..4) |batch| {                       // [8] loop disappears
                const j = batch * 8;
                const xv: @Vector(8, f32) = x[xb+j..][0..8].*;            // [9] bulk load
                const w: @Vector(8, f32) = @floatFromInt(                    // [6] i8→f32
                    @as(@Vector(8, i32), @intCast(
                        @as(@Vector(8, i8), local_i8[q0+j..][0..8].*))));
                dot0 = @mulAdd(@Vector(8, f32), xv, w, dot0);               // [2] FMLA
            }
            s0 += d0 * @reduce(.Add, dot0);                   // [3] horizontal sum
            // ... same for rows 1 through 3 ...
        }
        out[i] = s0; out[i+1] = s1; out[i+2] = s2; out[i+3] = s3;
    }
}
```

### NEON Instruction Map

| Optimization                         | Location                           | NEON Instruction              | Speedup vs Scalar      |
|--------------------------------------|------------------------------------|-------------------------------|------------------------|
| `@Vector(8, f32)` accumulator        | `var dot0: @Vector(8, f32)`        | —                             | 8× register width      |
| `@mulAdd(xv, w, acc)`                | `dot0 = @mulAdd(...)`              | `FMLA.4S` × 2                 | 8× fewer instructions  |
| `@reduce(.Add, dot0)`                | `@reduce(.Add, dot0)`              | `FADDP` × 3                   | 2.7× fewer instructions|
| `@Vector(8, i8)` → `@Vector(8, f32)` | `@floatFromInt(@intCast(...))`     | `SSHLL` + `SCVTF`              | 4.8× fewer instructions|
| `inline for (0..4)`                  | `inline for (0..4) \|batch\|`      | —                             | eliminates loop overhead|
| Bulk memory `src[j..][0..8].*`       | `x[xb+j..][0..8].*`               | `LD1.4S` × 2                   | 8× fewer loads         |

> **Bottom line:** The entire kernel is ~30 lines of Zig — each line maps to 1–5 NEON instructions. No `#include`, no intrinsic calls, no inline assembly. The same `@Vector` primitives work across all ARM, x86, and RISC-V backends. Write once, the compiler handles the rest.
