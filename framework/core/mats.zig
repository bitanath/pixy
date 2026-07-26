const std = @import("std");
const c = @import("constants.zig");
const dequant = @import("dequant.zig");

pub const deqRowNibble = dequant.deqRowNibble;
pub const deqRowWord = dequant.deqRowWord;
pub const getDeqRowFunc = dequant.getDeqRowFunc;
pub const quantToByteCache = dequant.quantToByteCache;
pub const accumByteCache = dequant.accumByteCache;
pub const dotByteByteCache = dequant.dotByteByteCache;
pub const fp32ToFp16 = dequant.fp32ToFp16;
pub const dequantizeByte = dequant.dequantizeByte;
pub const dequantizeNibble = dequant.dequantizeNibble;
pub const dequantizeWord = dequant.dequantizeWord;
pub const dequantizeRow = dequant.dequantizeRow;

pub fn fp16ToFp32(h: u16) f32 {
    return fp16Table[h];
}

pub fn bf16ToFp32(h: u16) f32 {
    const bf16_bits: u32 = @as(u32, @intCast(h)) << 16;
    return @bitCast(bf16_bits);
}

const fp16Table: [65536]f32 = blk: {
    @setEvalBranchQuota(1_000_000);
    var table: [65536]f32 = undefined;
    const fp16_exp_bias = 15;
    const fp32_exp_bias = 127;

    for (0..65536) |i| {
        const h: u16 = @intCast(i);
        const sign: u32 = @as(u32, @intCast((h >> 15) & 1)) << 31;
        const exp: u32 = @as(u32, @intCast((h >> 10) & 0x1f));
        const mant: u32 = @as(u32, @intCast(h & 0x3ff));

        var fp32_bits: u32 = undefined;

        if (exp == 0) {
            if (mant == 0) {
                fp32_bits = sign;
            } else {
                var sub_m = mant;
                var sub_e: i32 = 1 - @as(i32, fp16_exp_bias);
                while ((sub_m & 0x400) == 0) {
                    sub_m <<= 1;
                    sub_e -= 1;
                }
                sub_m &= ~@as(u32, 0x400);
                const new_exp: u32 = @intCast(sub_e + @as(i32, fp32_exp_bias));
                const new_mant = sub_m << 13;
                fp32_bits = sign | (new_exp << 23) | new_mant;
            }
        } else if (exp == 31) {
            if (mant == 0) {
                fp32_bits = sign | (0xff << 23);
            } else {
                fp32_bits = 0x7fc00000;
            }
        } else {
            const new_exp: u32 = @intCast(@as(i32, exp) - fp16_exp_bias + fp32_exp_bias);
            const new_mant = mant << 13;
            fp32_bits = sign | (new_exp << 23) | new_mant;
        }

        table[i] = @bitCast(fp32_bits);
    }

    break :blk table;
};

const bf16Table: [65536]f32 = blk: {
    @setEvalBranchQuota(65536);
    var table: [65536]f32 = undefined;
    for (0..65536) |i| {
        const bf16_bits: u32 = @as(u32, @intCast(i)) << 16;
        table[i] = @bitCast(bf16_bits);
    }
    break :blk table;
};

pub fn getBlockSize(t: c.GGML_TYPE) usize {
    return switch (t) {
        .F32 => 1,
        .Byte => c.QK8_0,
        .Nibble => c.QK_K,
        .Word => c.QK_K,
        else => 1,
    };
}

pub fn getTypeSize(t: c.GGML_TYPE) usize {
    return switch (t) {
        .F32 => 4,
        .Byte => 2 + c.QK8_0,
        .Nibble => 2 + 2 + 12 + c.QK_K / 8 + c.QK_K / 2,
        .Word => c.QK_K / 2 + c.QK_K / 4 + c.QK_K / 16 + 2,
        else => 0,
    };
}

pub fn getRowSize(n_cols: usize, t: c.GGML_TYPE) usize {
    const block_size = getBlockSize(t);
    const type_size = getTypeSize(t);
    return (n_cols / block_size) * type_size;
}

pub fn getVecDotFunc(t: c.GGML_TYPE) ?c.VecDotFunc {
    _ = t;
    return null;
}

pub fn getVecDotQ8Func(t: c.GGML_TYPE) ?c.VecDotQ8Func {
    _ = t;
    return null;
}

pub fn matmulWordLocalDirect(ctx: *c.Context, out: []f32, x: []const f32, data_ptr: usize, rows: usize, cols: usize, row_size: usize) void {
    const nb = cols >> 8;
    const gguf_u8 = ctx.gguf_uint8.?;
    const gguf_i8 = ctx.gguf_int8.?;
    for (0..rows) |i| {
        var bo = data_ptr + i * row_size;
        var sum: f64 = 0.0;
        var xb: usize = 0;
        for (0..nb) |_| {
            const ql_off = bo;
            const qh_off = bo + 128;
            const sc_off = bo + 192;
            const d = fp16ToFp32(gguf_u8[bo + 208] | (@as(u16, gguf_u8[bo + 209]) << 8));
            const d_f64: f64 = @floatCast(d);
            var block_sum: f64 = 0.0;

            for (0..2) |half| {
                const n_base = half * 128;
                for (0..32) |l| {
                    const is: usize = l >> 4;
                    const sc_base = sc_off + half * 8;
                    const q1 = @as(f64, @floatFromInt(@as(i32, ((gguf_u8[ql_off + n_base / 2 + l] & 0xf) | (((gguf_u8[qh_off + n_base / 4 + l] >> 0) & 3) << 4))) - 32));
                    const q2 = @as(f64, @floatFromInt(@as(i32, ((gguf_u8[ql_off + n_base / 2 + l + 32] & 0xf) | (((gguf_u8[qh_off + n_base / 4 + l] >> 2) & 3) << 4))) - 32));
                    const q3 = @as(f64, @floatFromInt(@as(i32, ((gguf_u8[ql_off + n_base / 2 + l] >> 4) | (((gguf_u8[qh_off + n_base / 4 + l] >> 4) & 3) << 4))) - 32));
                    const q4 = @as(f64, @floatFromInt(@as(i32, ((gguf_u8[ql_off + n_base / 2 + l + 32] >> 4) | (((gguf_u8[qh_off + n_base / 4 + l] >> 6) & 3) << 4))) - 32));
                    block_sum += d_f64 * @as(f64, @floatFromInt(gguf_i8[sc_base + is])) * q1 * x[xb + n_base + l];
                    block_sum += d_f64 * @as(f64, @floatFromInt(gguf_i8[sc_base + is + 2])) * q2 * x[xb + n_base + l + 32];
                    block_sum += d_f64 * @as(f64, @floatFromInt(gguf_i8[sc_base + is + 4])) * q3 * x[xb + n_base + l + 64];
                    block_sum += d_f64 * @as(f64, @floatFromInt(gguf_i8[sc_base + is + 6])) * q4 * x[xb + n_base + l + 96];
                }
            }

            sum += block_sum;
            bo += 210;
            xb += 256;
        }
        out[i] = @floatCast(sum);
    }
}

pub fn matmulByteBackup(out: []f32, x: []const f32, local_u8: []const u8, local_i8: []const i8, rows: usize, cols: usize, row_size: usize) void {
    const nb = cols >> 5;
    const rows4 = rows & ~@as(usize, 3);
    const rs2 = row_size + row_size;
    const rs3 = rs2 + row_size;
    for (0..rows4 / 4) |block| {
        const i = block * 4;
        var sum0: f32 = 0.0;
        var sum1: f32 = 0.0;
        var sum2: f32 = 0.0;
        var sum3: f32 = 0.0;
        var bo0: usize = i * row_size;
        var bo1: usize = bo0 + row_size;
        var bo2: usize = bo0 + rs2;
        var bo3: usize = bo0 + rs3;
        var xb: usize = 0;
        for (0..nb) |_| {
            const d0: f32 = fp16ToFp32(local_u8[bo0] | (@as(u16, local_u8[bo0 + 1]) << 8));
            const d1: f32 = fp16ToFp32(local_u8[bo1] | (@as(u16, local_u8[bo1 + 1]) << 8));
            const d2: f32 = fp16ToFp32(local_u8[bo2] | (@as(u16, local_u8[bo2 + 1]) << 8));
            const d3: f32 = fp16ToFp32(local_u8[bo3] | (@as(u16, local_u8[bo3 + 1]) << 8));
            const q0 = bo0 + 2;
            const q1 = bo1 + 2;
            const q2 = bo2 + 2;
            const q3 = bo3 + 2;
            var dot0_vec: @Vector(8, f32) = @splat(0.0);
            var dot1_vec: @Vector(8, f32) = @splat(0.0);
            var dot2_vec: @Vector(8, f32) = @splat(0.0);
            var dot3_vec: @Vector(8, f32) = @splat(0.0);
            inline for (0..4) |batch| {
                const j = batch * 8;
                const xv: @Vector(8, f32) = @as(*const [8]f32, @ptrCast(&x[xb + j])).*;
                const w0: @Vector(8, f32) = @floatFromInt(@as(@Vector(8, i32), @intCast(@as(@Vector(8, i8), local_i8[q0 + j ..][0..8].*))));
                const w1: @Vector(8, f32) = @floatFromInt(@as(@Vector(8, i32), @intCast(@as(@Vector(8, i8), local_i8[q1 + j ..][0..8].*))));
                const w2: @Vector(8, f32) = @floatFromInt(@as(@Vector(8, i32), @intCast(@as(@Vector(8, i8), local_i8[q2 + j ..][0..8].*))));
                const w3: @Vector(8, f32) = @floatFromInt(@as(@Vector(8, i32), @intCast(@as(@Vector(8, i8), local_i8[q3 + j ..][0..8].*))));
                dot0_vec = @mulAdd(@Vector(8, f32), xv, w0, dot0_vec);
                dot1_vec = @mulAdd(@Vector(8, f32), xv, w1, dot1_vec);
                dot2_vec = @mulAdd(@Vector(8, f32), xv, w2, dot2_vec);
                dot3_vec = @mulAdd(@Vector(8, f32), xv, w3, dot3_vec);
            }
            sum0 += d0 * @reduce(.Add, dot0_vec);
            sum1 += d1 * @reduce(.Add, dot1_vec);
            sum2 += d2 * @reduce(.Add, dot2_vec);
            sum3 += d3 * @reduce(.Add, dot3_vec);
            bo0 += 34;
            bo1 += 34;
            bo2 += 34;
            bo3 += 34;
            xb += 32;
        }
        out[i] = sum0;
        out[i + 1] = sum1;
        out[i + 2] = sum2;
        out[i + 3] = sum3;
    }
}

pub fn matmulKQuantLocal(ctx: *c.Context, out: []f32, x: []const f32, local_u8: []const u8, local_i8: []const i8, rows: usize, cols: usize, row_size: usize, deq_func: c.DeqRowFunc) void {
    const deq_buf = ctx.matmul_deq_buf.?;
    const rows4 = rows & ~@as(usize, 3);
    const off1 = cols;
    const off2 = cols + cols;
    const off3 = off2 + cols;
    const cols8 = cols & ~@as(usize, 7);

    for (0..rows4 / 4) |block| {
        const i = block * 4;
        const bo: usize = i * row_size;
        deq_func(ctx, local_u8, bo, deq_buf, 0, cols, local_i8);
        deq_func(ctx, local_u8, bo + row_size, deq_buf, off1, cols, local_i8);
        deq_func(ctx, local_u8, bo + row_size + row_size, deq_buf, off2, cols, local_i8);
        deq_func(ctx, local_u8, bo + row_size + row_size + row_size, deq_buf, off3, cols, local_i8);
        var s0_vec: @Vector(8, f32) = @splat(0.0);
        var s1_vec: @Vector(8, f32) = @splat(0.0);
        var s2_vec: @Vector(8, f32) = @splat(0.0);
        var s3_vec: @Vector(8, f32) = @splat(0.0);
        var j: usize = 0;
        while (j < cols8) : (j += 8) {
            const in_v: @Vector(8, f32) = @as(*const [8]f32, @ptrCast(&x[j])).*;
            s0_vec = @mulAdd(@Vector(8, f32), in_v, @as(*const [8]f32, @ptrCast(&deq_buf[j])).*, s0_vec);
            s1_vec = @mulAdd(@Vector(8, f32), in_v, @as(*const [8]f32, @ptrCast(&deq_buf[off1 + j])).*, s1_vec);
            s2_vec = @mulAdd(@Vector(8, f32), in_v, @as(*const [8]f32, @ptrCast(&deq_buf[off2 + j])).*, s2_vec);
            s3_vec = @mulAdd(@Vector(8, f32), in_v, @as(*const [8]f32, @ptrCast(&deq_buf[off3 + j])).*, s3_vec);
        }
        out[i] = @reduce(.Add, s0_vec);
        out[i + 1] = @reduce(.Add, s1_vec);
        out[i + 2] = @reduce(.Add, s2_vec);
        out[i + 3] = @reduce(.Add, s3_vec);
    }
    for (rows4..rows) |i| {
        deq_func(ctx, local_u8, i * row_size, deq_buf, 0, cols, local_i8);
        var s_vec: @Vector(8, f32) = @splat(0.0);
        var j: usize = 0;
        while (j < cols8) : (j += 8) {
            s_vec = @mulAdd(@Vector(8, f32), @as(*const [8]f32, @ptrCast(&x[j])).*, @as(*const [8]f32, @ptrCast(&deq_buf[j])).*, s_vec);
        }
        out[i] = @reduce(.Add, s_vec);
    }
}

pub fn matmulKQuantLocalBatch(ctx: *c.Context, outs: []const []f32, xs: []const []const f32, local_u8: []const u8, local_i8: []const i8, rows: usize, cols: usize, row_size: usize, batch_size: usize, deq_func: c.DeqRowFunc) void {
    const deq_buf = ctx.matmul_deq_buf.?;
    const rows4 = rows & ~@as(usize, 3);
    const off1 = cols;
    const off2 = cols + cols;
    const off3 = off2 + cols;
    const cols8 = cols & ~@as(usize, 7);
    const batch_trips = batch_size - (batch_size % 3);

    for (0..rows4 / 4) |block| {
        const i = block * 4;
        const bo: usize = i * row_size;
        deq_func(ctx, local_u8, bo, deq_buf, 0, cols, local_i8);
        deq_func(ctx, local_u8, bo + row_size, deq_buf, off1, cols, local_i8);
        deq_func(ctx, local_u8, bo + row_size + row_size, deq_buf, off2, cols, local_i8);
        deq_func(ctx, local_u8, bo + row_size + row_size + row_size, deq_buf, off3, cols, local_i8);

        var bt: usize = 0;
        while (bt < batch_trips) : (bt += 3) {
            const x_a = xs[bt];
            const x_b = xs[bt + 1];
            const x_c = xs[bt + 2];
            var s0_vec: @Vector(8, f32) = @splat(0.0);
            var s1_vec: @Vector(8, f32) = @splat(0.0);
            var s2_vec: @Vector(8, f32) = @splat(0.0);
            var s3_vec: @Vector(8, f32) = @splat(0.0);
            var t0_vec: @Vector(8, f32) = @splat(0.0);
            var t1_vec: @Vector(8, f32) = @splat(0.0);
            var t2_vec: @Vector(8, f32) = @splat(0.0);
            var t3_vec: @Vector(8, f32) = @splat(0.0);
            var v0_vec: @Vector(8, f32) = @splat(0.0);
            var v1_vec: @Vector(8, f32) = @splat(0.0);
            var v2_vec: @Vector(8, f32) = @splat(0.0);
            var v3_vec: @Vector(8, f32) = @splat(0.0);
            var j: usize = 0;
            while (j < cols8) : (j += 8) {
                const in_a_v: @Vector(8, f32) = @as(*const [8]f32, @ptrCast(&x_a[j])).*;
                const in_b_v: @Vector(8, f32) = @as(*const [8]f32, @ptrCast(&x_b[j])).*;
                const in_c_v: @Vector(8, f32) = @as(*const [8]f32, @ptrCast(&x_c[j])).*;
                s0_vec = @mulAdd(@Vector(8, f32), in_a_v, @as(*const [8]f32, @ptrCast(&deq_buf[j])).*, s0_vec);
                t0_vec = @mulAdd(@Vector(8, f32), in_b_v, @as(*const [8]f32, @ptrCast(&deq_buf[j])).*, t0_vec);
                v0_vec = @mulAdd(@Vector(8, f32), in_c_v, @as(*const [8]f32, @ptrCast(&deq_buf[j])).*, v0_vec);
                s1_vec = @mulAdd(@Vector(8, f32), in_a_v, @as(*const [8]f32, @ptrCast(&deq_buf[off1 + j])).*, s1_vec);
                t1_vec = @mulAdd(@Vector(8, f32), in_b_v, @as(*const [8]f32, @ptrCast(&deq_buf[off1 + j])).*, t1_vec);
                v1_vec = @mulAdd(@Vector(8, f32), in_c_v, @as(*const [8]f32, @ptrCast(&deq_buf[off1 + j])).*, v1_vec);
                s2_vec = @mulAdd(@Vector(8, f32), in_a_v, @as(*const [8]f32, @ptrCast(&deq_buf[off2 + j])).*, s2_vec);
                t2_vec = @mulAdd(@Vector(8, f32), in_b_v, @as(*const [8]f32, @ptrCast(&deq_buf[off2 + j])).*, t2_vec);
                v2_vec = @mulAdd(@Vector(8, f32), in_c_v, @as(*const [8]f32, @ptrCast(&deq_buf[off2 + j])).*, v2_vec);
                s3_vec = @mulAdd(@Vector(8, f32), in_a_v, @as(*const [8]f32, @ptrCast(&deq_buf[off3 + j])).*, s3_vec);
                t3_vec = @mulAdd(@Vector(8, f32), in_b_v, @as(*const [8]f32, @ptrCast(&deq_buf[off3 + j])).*, t3_vec);
                v3_vec = @mulAdd(@Vector(8, f32), in_c_v, @as(*const [8]f32, @ptrCast(&deq_buf[off3 + j])).*, v3_vec);
            }
            outs[bt][i] = @reduce(.Add, s0_vec);
            outs[bt][i + 1] = @reduce(.Add, s1_vec);
            outs[bt][i + 2] = @reduce(.Add, s2_vec);
            outs[bt][i + 3] = @reduce(.Add, s3_vec);
            outs[bt + 1][i] = @reduce(.Add, t0_vec);
            outs[bt + 1][i + 1] = @reduce(.Add, t1_vec);
            outs[bt + 1][i + 2] = @reduce(.Add, t2_vec);
            outs[bt + 1][i + 3] = @reduce(.Add, t3_vec);
            outs[bt + 2][i] = @reduce(.Add, v0_vec);
            outs[bt + 2][i + 1] = @reduce(.Add, v1_vec);
            outs[bt + 2][i + 2] = @reduce(.Add, v2_vec);
            outs[bt + 2][i + 3] = @reduce(.Add, v3_vec);
        }
        for (batch_trips..batch_size) |batch| {
            const x_arr = xs[batch];
            var s0_vec: @Vector(8, f32) = @splat(0.0);
            var s1_vec: @Vector(8, f32) = @splat(0.0);
            var s2_vec: @Vector(8, f32) = @splat(0.0);
            var s3_vec: @Vector(8, f32) = @splat(0.0);
            var j: usize = 0;
            while (j < cols8) : (j += 8) {
                const in_v: @Vector(8, f32) = @as(*const [8]f32, @ptrCast(&x_arr[j])).*;
                s0_vec = @mulAdd(@Vector(8, f32), in_v, @as(*const [8]f32, @ptrCast(&deq_buf[j])).*, s0_vec);
                s1_vec = @mulAdd(@Vector(8, f32), in_v, @as(*const [8]f32, @ptrCast(&deq_buf[off1 + j])).*, s1_vec);
                s2_vec = @mulAdd(@Vector(8, f32), in_v, @as(*const [8]f32, @ptrCast(&deq_buf[off2 + j])).*, s2_vec);
                s3_vec = @mulAdd(@Vector(8, f32), in_v, @as(*const [8]f32, @ptrCast(&deq_buf[off3 + j])).*, s3_vec);
            }
            outs[batch][i] = @reduce(.Add, s0_vec);
            outs[batch][i + 1] = @reduce(.Add, s1_vec);
            outs[batch][i + 2] = @reduce(.Add, s2_vec);
            outs[batch][i + 3] = @reduce(.Add, s3_vec);
        }
    }
    for (rows4..rows) |i| {
        deq_func(ctx, local_u8, i * row_size, deq_buf, 0, cols, local_i8);
        var bt: usize = 0;
        while (bt < batch_trips) : (bt += 3) {
            const x_a = xs[bt];
            const x_b = xs[bt + 1];
            const x_c = xs[bt + 2];
            var s_vec: @Vector(8, f32) = @splat(0.0);
            var t_vec: @Vector(8, f32) = @splat(0.0);
            var u_vec: @Vector(8, f32) = @splat(0.0);
            var j: usize = 0;
            while (j < cols8) : (j += 8) {
                const w_v: @Vector(8, f32) = @as(*const [8]f32, @ptrCast(&deq_buf[j])).*;
                s_vec = @mulAdd(@Vector(8, f32), @as(*const [8]f32, @ptrCast(&x_a[j])).*, w_v, s_vec);
                t_vec = @mulAdd(@Vector(8, f32), @as(*const [8]f32, @ptrCast(&x_b[j])).*, w_v, t_vec);
                u_vec = @mulAdd(@Vector(8, f32), @as(*const [8]f32, @ptrCast(&x_c[j])).*, w_v, u_vec);
            }
            outs[bt][i] = @reduce(.Add, s_vec);
            outs[bt + 1][i] = @reduce(.Add, t_vec);
            outs[bt + 2][i] = @reduce(.Add, u_vec);
        }
        for (batch_trips..batch_size) |batch| {
            const x_arr = xs[batch];
            var s_vec: @Vector(8, f32) = @splat(0.0);
            var j: usize = 0;
            while (j < cols8) : (j += 8) {
                s_vec = @mulAdd(@Vector(8, f32), @as(*const [8]f32, @ptrCast(&x_arr[j])).*, @as(*const [8]f32, @ptrCast(&deq_buf[j])).*, s_vec);
            }
            outs[batch][i] = @reduce(.Add, s_vec);
        }
    }
}

pub fn matmulByteLocal(out: []f32, x: []const f32, local_u8: []const u8, local_i8: []const i8, rows: usize, cols: usize, row_size: usize) void {
    const nb = cols >> 5;
    const rows4 = rows & ~@as(usize, 3);
    const rs2 = row_size + row_size;
    const rs3 = rs2 + row_size;
    for (0..rows4 / 4) |block| {
        const i = block * 4;
        var sum0: f32 = 0.0;
        var sum1: f32 = 0.0;
        var sum2: f32 = 0.0;
        var sum3: f32 = 0.0;
        var bo0: usize = i * row_size;
        var bo1: usize = bo0 + row_size;
        var bo2: usize = bo0 + rs2;
        var bo3: usize = bo0 + rs3;
        var xb: usize = 0;
        for (0..nb) |_| {
            const d0: f32 = fp16ToFp32(local_u8[bo0] | (@as(u16, local_u8[bo0 + 1]) << 8));
            const d1: f32 = fp16ToFp32(local_u8[bo1] | (@as(u16, local_u8[bo1 + 1]) << 8));
            const d2: f32 = fp16ToFp32(local_u8[bo2] | (@as(u16, local_u8[bo2 + 1]) << 8));
            const d3: f32 = fp16ToFp32(local_u8[bo3] | (@as(u16, local_u8[bo3 + 1]) << 8));
            const q0 = bo0 + 2;
            const q1 = bo1 + 2;
            const q2 = bo2 + 2;
            const q3 = bo3 + 2;
            var dot0_vec: @Vector(8, f32) = @splat(0.0);
            var dot1_vec: @Vector(8, f32) = @splat(0.0);
            var dot2_vec: @Vector(8, f32) = @splat(0.0);
            var dot3_vec: @Vector(8, f32) = @splat(0.0);
            inline for (0..4) |batch| {
                const j = batch * 8;
                const xv: @Vector(8, f32) = @as(*const [8]f32, @ptrCast(&x[xb + j])).*;
                const w0: @Vector(8, f32) = @floatFromInt(@as(@Vector(8, i32), @intCast(@as(@Vector(8, i8), local_i8[q0 + j ..][0..8].*))));
                const w1: @Vector(8, f32) = @floatFromInt(@as(@Vector(8, i32), @intCast(@as(@Vector(8, i8), local_i8[q1 + j ..][0..8].*))));
                const w2: @Vector(8, f32) = @floatFromInt(@as(@Vector(8, i32), @intCast(@as(@Vector(8, i8), local_i8[q2 + j ..][0..8].*))));
                const w3: @Vector(8, f32) = @floatFromInt(@as(@Vector(8, i32), @intCast(@as(@Vector(8, i8), local_i8[q3 + j ..][0..8].*))));
                dot0_vec = @mulAdd(@Vector(8, f32), xv, w0, dot0_vec);
                dot1_vec = @mulAdd(@Vector(8, f32), xv, w1, dot1_vec);
                dot2_vec = @mulAdd(@Vector(8, f32), xv, w2, dot2_vec);
                dot3_vec = @mulAdd(@Vector(8, f32), xv, w3, dot3_vec);
            }
            sum0 += d0 * @reduce(.Add, dot0_vec);
            sum1 += d1 * @reduce(.Add, dot1_vec);
            sum2 += d2 * @reduce(.Add, dot2_vec);
            sum3 += d3 * @reduce(.Add, dot3_vec);
            bo0 += 34;
            bo1 += 34;
            bo2 += 34;
            bo3 += 34;
            xb += 32;
        }
        out[i] = sum0;
        out[i + 1] = sum1;
        out[i + 2] = sum2;
        out[i + 3] = sum3;
    }
}

pub fn matmulByteLocalBatch(ctx: *c.Context, outs: []const []f32, xs: []const []const f32, local_u8: []const u8, local_i8: []const i8, rows: usize, cols: usize, row_size: usize, batch_size: usize) void {
    const deq_buf = ctx.matmul_deq_buf.?;
    const nb = cols >> 5;
    const rows4 = rows & ~@as(usize, 3);
    const off1 = cols;
    const off2 = cols + cols;
    const off3 = off2 + cols;
    const cols8 = cols & ~@as(usize, 7);
    const batch_trips = batch_size - (batch_size % 3);

    for (0..rows4 / 4) |block| {
        const i = block * 4;
        var bo: usize = i * row_size;
        for (0..4) |r| {
            const b_off = r * cols;
            var bk = bo;
            var idx = b_off;
            for (0..nb) |_| {
                const d: f32 = fp16ToFp32(local_u8[bk] | (@as(u16, local_u8[bk + 1]) << 8));
                const qo = bk + 2;
                const dv: @Vector(8, f32) = @splat(d);
                inline for (0..4) |batch| {
                    const base = batch * 8;
                    const fv: @Vector(8, f32) = @floatFromInt(@as(@Vector(8, i32), @intCast(@as(@Vector(8, i8), local_i8[qo + base ..][0..8].*))));
                    @as(*[8]f32, @ptrCast(&deq_buf[idx + base])).* = fv * dv;
                }
                bk += 34;
                idx += 32;
            }
            bo += row_size;
        }
        var bt: usize = 0;
        while (bt < batch_trips) : (bt += 3) {
            const x_a = xs[bt];
            const x_b = xs[bt + 1];
            const x_c = xs[bt + 2];
            var s0_vec: @Vector(8, f32) = @splat(0.0);
            var s1_vec: @Vector(8, f32) = @splat(0.0);
            var s2_vec: @Vector(8, f32) = @splat(0.0);
            var s3_vec: @Vector(8, f32) = @splat(0.0);
            var t0_vec: @Vector(8, f32) = @splat(0.0);
            var t1_vec: @Vector(8, f32) = @splat(0.0);
            var t2_vec: @Vector(8, f32) = @splat(0.0);
            var t3_vec: @Vector(8, f32) = @splat(0.0);
            var v0_vec: @Vector(8, f32) = @splat(0.0);
            var v1_vec: @Vector(8, f32) = @splat(0.0);
            var v2_vec: @Vector(8, f32) = @splat(0.0);
            var v3_vec: @Vector(8, f32) = @splat(0.0);
            var j: usize = 0;
            while (j < cols8) : (j += 8) {
                const in_a_v: @Vector(8, f32) = @as(*const [8]f32, @ptrCast(&x_a[j])).*;
                const in_b_v: @Vector(8, f32) = @as(*const [8]f32, @ptrCast(&x_b[j])).*;
                const in_c_v: @Vector(8, f32) = @as(*const [8]f32, @ptrCast(&x_c[j])).*;
                s0_vec = @mulAdd(@Vector(8, f32), in_a_v, @as(*const [8]f32, @ptrCast(&deq_buf[j])).*, s0_vec);
                t0_vec = @mulAdd(@Vector(8, f32), in_b_v, @as(*const [8]f32, @ptrCast(&deq_buf[j])).*, t0_vec);
                v0_vec = @mulAdd(@Vector(8, f32), in_c_v, @as(*const [8]f32, @ptrCast(&deq_buf[j])).*, v0_vec);
                s1_vec = @mulAdd(@Vector(8, f32), in_a_v, @as(*const [8]f32, @ptrCast(&deq_buf[off1 + j])).*, s1_vec);
                t1_vec = @mulAdd(@Vector(8, f32), in_b_v, @as(*const [8]f32, @ptrCast(&deq_buf[off1 + j])).*, t1_vec);
                v1_vec = @mulAdd(@Vector(8, f32), in_c_v, @as(*const [8]f32, @ptrCast(&deq_buf[off1 + j])).*, v1_vec);
                s2_vec = @mulAdd(@Vector(8, f32), in_a_v, @as(*const [8]f32, @ptrCast(&deq_buf[off2 + j])).*, s2_vec);
                t2_vec = @mulAdd(@Vector(8, f32), in_b_v, @as(*const [8]f32, @ptrCast(&deq_buf[off2 + j])).*, t2_vec);
                v2_vec = @mulAdd(@Vector(8, f32), in_c_v, @as(*const [8]f32, @ptrCast(&deq_buf[off2 + j])).*, v2_vec);
                s3_vec = @mulAdd(@Vector(8, f32), in_a_v, @as(*const [8]f32, @ptrCast(&deq_buf[off3 + j])).*, s3_vec);
                t3_vec = @mulAdd(@Vector(8, f32), in_b_v, @as(*const [8]f32, @ptrCast(&deq_buf[off3 + j])).*, t3_vec);
                v3_vec = @mulAdd(@Vector(8, f32), in_c_v, @as(*const [8]f32, @ptrCast(&deq_buf[off3 + j])).*, v3_vec);
            }
            outs[bt][i] = @reduce(.Add, s0_vec);
            outs[bt][i + 1] = @reduce(.Add, s1_vec);
            outs[bt][i + 2] = @reduce(.Add, s2_vec);
            outs[bt][i + 3] = @reduce(.Add, s3_vec);
            outs[bt + 1][i] = @reduce(.Add, t0_vec);
            outs[bt + 1][i + 1] = @reduce(.Add, t1_vec);
            outs[bt + 1][i + 2] = @reduce(.Add, t2_vec);
            outs[bt + 1][i + 3] = @reduce(.Add, t3_vec);
            outs[bt + 2][i] = @reduce(.Add, v0_vec);
            outs[bt + 2][i + 1] = @reduce(.Add, v1_vec);
            outs[bt + 2][i + 2] = @reduce(.Add, v2_vec);
            outs[bt + 2][i + 3] = @reduce(.Add, v3_vec);
        }
        for (batch_trips..batch_size) |batch| {
            const x_arr = xs[batch];
            var s0_vec: @Vector(8, f32) = @splat(0.0);
            var s1_vec: @Vector(8, f32) = @splat(0.0);
            var s2_vec: @Vector(8, f32) = @splat(0.0);
            var s3_vec: @Vector(8, f32) = @splat(0.0);
            var j: usize = 0;
            while (j < cols8) : (j += 8) {
                const in_v: @Vector(8, f32) = @as(*const [8]f32, @ptrCast(&x_arr[j])).*;
                s0_vec = @mulAdd(@Vector(8, f32), in_v, @as(*const [8]f32, @ptrCast(&deq_buf[j])).*, s0_vec);
                s1_vec = @mulAdd(@Vector(8, f32), in_v, @as(*const [8]f32, @ptrCast(&deq_buf[off1 + j])).*, s1_vec);
                s2_vec = @mulAdd(@Vector(8, f32), in_v, @as(*const [8]f32, @ptrCast(&deq_buf[off2 + j])).*, s2_vec);
                s3_vec = @mulAdd(@Vector(8, f32), in_v, @as(*const [8]f32, @ptrCast(&deq_buf[off3 + j])).*, s3_vec);
            }
            outs[batch][i] = @reduce(.Add, s0_vec);
            outs[batch][i + 1] = @reduce(.Add, s1_vec);
            outs[batch][i + 2] = @reduce(.Add, s2_vec);
            outs[batch][i + 3] = @reduce(.Add, s3_vec);
        }
    }
}

pub fn matmulQuantized(ctx: *c.Context, out: []f32, x: []const f32, qw: c.QuantizedTensor) void {
    const rows = qw.rows;
    const cols = qw.cols;
    const base_offset = qw.data_offset;
    const row_size = qw.row_size;
    const dot_q8_func = qw.dot_q8_func;
    //FIXME HARD PREFER deq_row_func over dot_q8_func: SIMD deqRow + SIMD f32 dot is faster
    if (qw.quant_type == .Byte) {
        matmulByteLocal(out, x, qw.local_u8.?, qw.local_i8.?, rows, cols, row_size);
    } else if (qw.deq_row_func) |deq_func| {
        matmulKQuantLocal(ctx, out, x, qw.local_u8.?, qw.local_i8.?, rows, cols, row_size, deq_func);
    } else if (dot_q8_func) |dq8f| {
        const x_q8 = ctx.x_q8_buf.?;
        const x_q8i8 = ctx.x_q8_int8_buf.?;
        quantToByteCache(x, 0, x_q8, x_q8i8, 0, cols);
        for (0..rows) |i| {
            out[i] = @floatCast(dq8f(ctx, x_q8, x_q8i8, base_offset + i * row_size, cols));
        }
    } else if (qw.dot_func) |df| {
        for (0..rows) |i| {
            out[i] = @floatCast(df(ctx, x, base_offset + i * row_size, cols));
        }
    }
}

pub fn matmulQuantizedPreQ8(ctx: *c.Context, out: []f32, qw: c.QuantizedTensor, x_q8: []const u8, x_q8i8: []const i8) void {
    const rows = qw.rows;
    const dot_q8_func = qw.dot_q8_func.?;
    const base_offset = qw.data_offset;
    const row_size = qw.row_size;
    const cols = qw.cols;
    for (0..rows) |i| {
        out[i] = @floatCast(dot_q8_func(ctx, x_q8, x_q8i8, base_offset + i * row_size, cols));
    }
}

pub fn matmulQuantizedBatch(ctx: *c.Context, outs: []const []f32, xs: []const []const f32, qw: c.QuantizedTensor, batch_size: usize) void {
    const rows = qw.rows;
    const cols = qw.cols;
    const base_offset = qw.data_offset;
    const row_size = qw.row_size;
    const dot_q8_func = qw.dot_q8_func;
    // PREFER deq_row_func over dot_q8_func for batched matmuls:
    // deq_row_func dequantizes each weight row once and reuses the float row
    // across all batch elements, avoiding re-reading weight data batch_size times.
    if (qw.quant_type == .Byte) {
        matmulByteLocalBatch(ctx, outs, xs, qw.local_u8.?, qw.local_i8.?, rows, cols, row_size, batch_size);
    } else if (qw.deq_row_func) |deq_func| {
        matmulKQuantLocalBatch(ctx, outs, xs, qw.local_u8.?, qw.local_i8.?, rows, cols, row_size, batch_size, deq_func);
    } else if (dot_q8_func) |dq8f| {
        const batch_q8 = ctx.state.?.batch_q8;
        const batch_q8i8 = ctx.state.?.batch_q8i8;
        for (0..batch_size) |b| {
            quantToByteCache(xs[b], 0, batch_q8[b].?, batch_q8i8[b].?, 0, cols);
        }
        for (0..rows) |i| {
            const row_off = base_offset + i * row_size;
            for (0..batch_size) |b| {
                outs[b][i] = @floatCast(dq8f(ctx, batch_q8[b].?, batch_q8i8[b].?, row_off, cols));
            }
        }
    } else if (qw.dot_func) |df| {
        for (0..rows) |i| {
            const row_off = base_offset + i * row_size;
            for (0..batch_size) |b| {
                outs[b][i] = @floatCast(df(ctx, xs[b], row_off, cols));
            }
        }
    }
}
