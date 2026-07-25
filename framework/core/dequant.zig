const std = @import("std");
const c = @import("constants.zig");
const mats = @import("mats.zig");
const fp16ToFp32 = mats.fp16ToFp32;
const bf16ToFp32 = mats.bf16ToFp32;

pub fn deqRowNibble(_: *c.Context, u8_data: []const u8, off: usize, dst: []f32, dst_off: usize, cols: usize, _: []const i8) void {
    const nb = cols >> 8;
    for (0..nb) |i| {
        const bo = off + i * 176;
        const d: f32 = @floatCast(fp16ToFp32(u8_data[bo] | (@as(u16, u8_data[bo + 1]) << 8)));
        const dmin: f32 = @floatCast(fp16ToFp32(u8_data[bo + 2] | (@as(u16, u8_data[bo + 3]) << 8)));
        const sc_off = bo + 4;
        const qh_off = bo + 16;
        const ql_off = bo + 48;
        const y_base = dst_off + i * 256;
        var is: usize = 0;
        var bit1: u32 = 1;
        var bit2: u32 = 2;
        var ql_idx: usize = 0;

        const lo_mask: @Vector(16, u8) = @splat(0x0f);
        const shift4: @Vector(16, u8) = @splat(4);
        const vec16: @Vector(16, u8) = @splat(16);

        var j: usize = 0;
        while (j < 256) : (j += 64) {
            var sc: usize = undefined;
            var m_val: usize = undefined;
            if (is < 4) {
                sc = u8_data[sc_off + is] & 63;
                m_val = u8_data[sc_off + is + 4] & 63;
            } else {
                sc = (u8_data[sc_off + is + 4] & 0x0f) | ((u8_data[sc_off + is - 4] >> 6) << 4);
                m_val = (u8_data[sc_off + is + 4] >> 4) | ((u8_data[sc_off + is] >> 6) << 4);
            }
            const d1: f32 = d * @as(f32, @floatFromInt(sc));
            const m1: f32 = dmin * @as(f32, @floatFromInt(m_val));
            is += 1;

            if (is < 4) {
                sc = u8_data[sc_off + is] & 63;
                m_val = u8_data[sc_off + is + 4] & 63;
            } else {
                sc = (u8_data[sc_off + is + 4] & 0x0f) | ((u8_data[sc_off + is - 4] >> 6) << 4);
                m_val = (u8_data[sc_off + is + 4] >> 4) | ((u8_data[sc_off + is] >> 6) << 4);
            }
            const d2: f32 = d * @as(f32, @floatFromInt(sc));
            const m2: f32 = dmin * @as(f32, @floatFromInt(m_val));
            is += 1;

            const d1v: @Vector(4, f32) = @splat(d1);
            const m1v: @Vector(4, f32) = @splat(m1);
            const d2v: @Vector(4, f32) = @splat(d2);
            const m2v: @Vector(4, f32) = @splat(m2);

            const bit1_u8: u8 = @intCast(bit1);
            const bit2_u8: u8 = @intCast(bit2);
            const bit1v: @Vector(16, u8) = @splat(bit1_u8);
            const bit2v: @Vector(16, u8) = @splat(bit2_u8);
            const zero8: @Vector(16, u8) = @splat(0);

            const qh0: @Vector(16, u8) = u8_data[qh_off ..][0..16].*;
            const qh1: @Vector(16, u8) = u8_data[qh_off + 16 ..][0..16].*;

            inline for (0..2) |half| {
                const byte_off = half * 16;
                const qh = if (half == 0) qh0 else qh1;
                const ql: @Vector(16, u8) = u8_data[ql_off + ql_idx + byte_off ..][0..16].*;
                const ql_lo: @Vector(16, u8) = ql & lo_mask;
                const ql_hi: @Vector(16, u8) = ql >> shift4;

                const add1: @Vector(16, u8) = @select(u8, qh & bit1v != zero8, vec16, zero8);
                const add2: @Vector(16, u8) = @select(u8, qh & bit2v != zero8, vec16, zero8);

                const qv1: @Vector(16, u8) = ql_lo | add1;
                const qv2: @Vector(16, u8) = ql_hi | add2;

                inline for (0..4) |g| {
                    const gbase = g * 4;
                    const idx: @Vector(4, i32) = .{
                        @as(i32, @intCast(gbase)), @as(i32, @intCast(gbase + 1)),
                        @as(i32, @intCast(gbase + 2)), @as(i32, @intCast(gbase + 3)),
                    };

                    const c1: @Vector(4, u8) = @shuffle(u8, qv1, undefined, idx);
                    const f1: @Vector(4, f32) = .{
                        @floatFromInt(c1[0]), @floatFromInt(c1[1]),
                        @floatFromInt(c1[2]), @floatFromInt(c1[3]),
                    };
                    const r1: @Vector(4, f32) = f1 * d1v - m1v;
                    const o1 = y_base + j + byte_off + gbase;
                    @as(*[4]f32, @ptrCast(&dst[o1])).* = r1;

                    const c2: @Vector(4, u8) = @shuffle(u8, qv2, undefined, idx);
                    const f2: @Vector(4, f32) = .{
                        @floatFromInt(c2[0]), @floatFromInt(c2[1]),
                        @floatFromInt(c2[2]), @floatFromInt(c2[3]),
                    };
                    const r2: @Vector(4, f32) = f2 * d2v - m2v;
                    const o2 = y_base + j + 32 + byte_off + gbase;
                    @as(*[4]f32, @ptrCast(&dst[o2])).* = r2;
                }
            }

            ql_idx += 32;
            bit1 <<= 2;
            bit2 <<= 2;
        }
    }
}

pub fn deqRowWord(_: *c.Context, u8_data: []const u8, off: usize, dst: []f32, dst_off: usize, cols: usize, i8_data: []const i8) void {
    const nb = cols >> 8;
    for (0..nb) |i| {
        const bo = off + i * 210;
        const ql_off = bo;
        const qh_off = bo + 128;
        const sc_off = bo + 192;
        const d_off = bo + 208;
        const d: f32 = @floatCast(fp16ToFp32(u8_data[d_off] | (@as(u16, u8_data[d_off + 1]) << 8)));
        const y_base = dst_off + i * 256;

        const lo_mask: @Vector(16, u8) = @splat(0x0f);
        const shift4: @Vector(16, u8) = @splat(4);
        const qh_mask: @Vector(16, u8) = @splat(0x03);

        for (0..2) |n_val| {
            const n: usize = n_val * 128;
            const ql_base = ql_off + (n >> 1);
            const qh_base = qh_off + (n >> 2);
            const sc_base = sc_off + (n >> 7) * 8;

            inline for (0..2) |batch| {
                const l_off = batch * 16;

                const ql_v: @Vector(16, u8) = u8_data[ql_base + l_off ..][0..16].*;
                const ql2_v: @Vector(16, u8) = u8_data[ql_base + 32 + l_off ..][0..16].*;
                const qh_v: @Vector(16, u8) = u8_data[qh_base + l_off ..][0..16].*;

                const lo: @Vector(16, u8) = ql_v & lo_mask;
                const hi: @Vector(16, u8) = ql_v >> shift4;
                const lo2: @Vector(16, u8) = ql2_v & lo_mask;
                const hi2: @Vector(16, u8) = ql2_v >> shift4;

                const qh0: @Vector(16, u8) = qh_v & qh_mask;
                const qh1: @Vector(16, u8) = (qh_v >> @splat(@as(u8, 2))) & qh_mask;
                const qh2: @Vector(16, u8) = (qh_v >> @splat(@as(u8, 4))) & qh_mask;
                const qh3: @Vector(16, u8) = (qh_v >> @splat(@as(u8, 6))) & qh_mask;

                const q1_raw: @Vector(16, u8) = lo | (qh0 << shift4);
                const q2_raw: @Vector(16, u8) = lo2 | (qh1 << shift4);
                const q3_raw: @Vector(16, u8) = hi | (qh2 << shift4);
                const q4_raw: @Vector(16, u8) = hi2 | (qh3 << shift4);

                const s1: f32 = @floatCast(@as(f32, i8_data[sc_base + batch + 0]));
                const s2: f32 = @floatCast(@as(f32, i8_data[sc_base + batch + 2]));
                const s3: f32 = @floatCast(@as(f32, i8_data[sc_base + batch + 4]));
                const s4: f32 = @floatCast(@as(f32, i8_data[sc_base + batch + 6]));
                const d_s1: f32 = d * s1;
                const d_s2: f32 = d * s2;
                const d_s3: f32 = d * s3;
                const d_s4: f32 = d * s4;

                const s1v: @Vector(4, f32) = @splat(d_s1);
                const s2v: @Vector(4, f32) = @splat(d_s2);
                const s3v: @Vector(4, f32) = @splat(d_s3);
                const s4v: @Vector(4, f32) = @splat(d_s4);

                inline for (0..4) |g| {
                    const gbase = g * 4;
                    const idx: @Vector(4, i32) = .{
                        @as(i32, @intCast(gbase)), @as(i32, @intCast(gbase + 1)),
                        @as(i32, @intCast(gbase + 2)), @as(i32, @intCast(gbase + 3)),
                    };

                    const c1: @Vector(4, u8) = @shuffle(u8, q1_raw, undefined, idx);
                    const f1: @Vector(4, f32) = .{
                        @floatFromInt(@as(i32, @intCast(c1[0])) - 32),
                        @floatFromInt(@as(i32, @intCast(c1[1])) - 32),
                        @floatFromInt(@as(i32, @intCast(c1[2])) - 32),
                        @floatFromInt(@as(i32, @intCast(c1[3])) - 32),
                    };
                    const r1: @Vector(4, f32) = f1 * s1v;
                    const o1 = y_base + n + l_off + gbase;
                    @as(*[4]f32, @ptrCast(&dst[o1])).* = r1;

                    const c2: @Vector(4, u8) = @shuffle(u8, q2_raw, undefined, idx);
                    const f2: @Vector(4, f32) = .{
                        @floatFromInt(@as(i32, @intCast(c2[0])) - 32),
                        @floatFromInt(@as(i32, @intCast(c2[1])) - 32),
                        @floatFromInt(@as(i32, @intCast(c2[2])) - 32),
                        @floatFromInt(@as(i32, @intCast(c2[3])) - 32),
                    };
                    const r2: @Vector(4, f32) = f2 * s2v;
                    const o2 = y_base + n + l_off + gbase + 32;
                    @as(*[4]f32, @ptrCast(&dst[o2])).* = r2;

                    const c3: @Vector(4, u8) = @shuffle(u8, q3_raw, undefined, idx);
                    const f3: @Vector(4, f32) = .{
                        @floatFromInt(@as(i32, @intCast(c3[0])) - 32),
                        @floatFromInt(@as(i32, @intCast(c3[1])) - 32),
                        @floatFromInt(@as(i32, @intCast(c3[2])) - 32),
                        @floatFromInt(@as(i32, @intCast(c3[3])) - 32),
                    };
                    const r3: @Vector(4, f32) = f3 * s3v;
                    const o3 = y_base + n + l_off + gbase + 64;
                    @as(*[4]f32, @ptrCast(&dst[o3])).* = r3;

                    const c4: @Vector(4, u8) = @shuffle(u8, q4_raw, undefined, idx);
                    const f4: @Vector(4, f32) = .{
                        @floatFromInt(@as(i32, @intCast(c4[0])) - 32),
                        @floatFromInt(@as(i32, @intCast(c4[1])) - 32),
                        @floatFromInt(@as(i32, @intCast(c4[2])) - 32),
                        @floatFromInt(@as(i32, @intCast(c4[3])) - 32),
                    };
                    const r4: @Vector(4, f32) = f4 * s4v;
                    const o4 = y_base + n + l_off + gbase + 96;
                    @as(*[4]f32, @ptrCast(&dst[o4])).* = r4;
                }
            }
        }
    }
}

pub fn getDeqRowFunc(t: c.GGML_TYPE) ?c.DeqRowFunc {
    return switch (t) {
        .Nibble => deqRowNibble,
        .Word => deqRowWord,
        else => null,
    };
}

pub fn fp32ToFp16(f: f32) u16 {
    const bits: u32 = @bitCast(f);
    const sign: u16 = @truncate((bits >> 16) & 0x8000);
    const exp: i32 = @as(i32, @intCast((bits >> 23) & 0xff)) - 127 + 15;
    var mant: u16 = @truncate((bits >> 13) & 0x3ff);
    if (exp <= 0) {
        if (exp < -10) {
            return sign;
        }
        mant = (mant | 0x400) >> @as(u4, @intCast(1 - exp));
        return sign | mant;
    } else if (exp >= 31) {
        return sign | 0x7c00;
    }
    return sign | (@as(u16, @intCast(exp)) << 10) | mant;
}

pub fn accumByteCache(out: []f32, out_off: usize, cache: []const u8, cache_i8: []const i8, cache_off: usize, weight: f32, count: usize) void {
    if (weight > -1e-8 and weight < 1e-8) return;
    const nb = count >> 5;
    var bo = cache_off;
    var ob = out_off;
    for (0..nb) |_| {
        const d: f32 = fp16ToFp32(cache[bo] | (@as(u16, cache[bo + 1]) << 8));
        const scale: f32 = d * weight;
        const q_off = bo + 2;
        const sv: @Vector(8, f32) = @splat(scale);
        inline for (0..4) |batch| {
            const base = batch * 8;
            const i8_chunk: @Vector(8, i8) = cache_i8[q_off + base ..][0..8].*;
            const f_chunk: @Vector(8, f32) = @floatFromInt(@as(@Vector(8, i32), @intCast(i8_chunk)));
            const o_v: @Vector(8, f32) = @as(*const [8]f32, @ptrCast(&out[ob + base])).*;
            const r: @Vector(8, f32) = o_v + f_chunk * sv;
            @as(*[8]f32, @ptrCast(&out[ob + base])).* = r;
        }
        bo += c.Q8_0_BLOCK_SIZE;
        ob += 32;
    }
}

pub fn dotByteByteCache(a_q8: []const u8, a_i8: []const i8, a_off: usize, b_q8: []const u8, b_i8: []const i8, b_off: usize, count: usize) f32 {
    const nb = count >> 5;
    var sum: f32 = 0.0;
    var ao = a_off;
    var bo = b_off;
    for (0..nb) |_| {
        const da: f32 = fp16ToFp32(a_q8[ao] | (@as(u16, a_q8[ao + 1]) << 8));
        const db: f32 = fp16ToFp32(b_q8[bo] | (@as(u16, b_q8[bo + 1]) << 8));
        const qa = ao + 2;
        const qb = bo + 2;
        var isum: i32 = 0;
        inline for (0..8) |batch| {
            const base = batch * 4;
            const av: @Vector(4, i8) = a_i8[qa + base ..][0..4].*;
            const bv: @Vector(4, i8) = b_i8[qb + base ..][0..4].*;
            isum += @as(i32, av[0]) * @as(i32, bv[0]);
            isum += @as(i32, av[1]) * @as(i32, bv[1]);
            isum += @as(i32, av[2]) * @as(i32, bv[2]);
            isum += @as(i32, av[3]) * @as(i32, bv[3]);
        }
        sum += (da * db) * @as(f32, @floatFromInt(isum));
        ao += c.Q8_0_BLOCK_SIZE;
        bo += c.Q8_0_BLOCK_SIZE;
    }
    return sum;
}

pub fn quantToByteCache(x: []const f32, x_off: usize, q8_u8: []u8, q8_i8: []i8, q8_off: usize, n: usize) void {
    const nb = n >> 5;
    for (0..nb) |b| {
        const base = x_off + b * 32;
        var max_v: @Vector(4, f32) = @splat(0.0);
        inline for (0..8) |batch| {
            const vals: @Vector(4, f32) = @as(*const [4]f32, @ptrCast(&x[base + batch * 4])).*;
            max_v = @select(f32, @abs(vals) > max_v, @abs(vals), max_v);
        }
        const max_val: f32 = @reduce(.Max, max_v);
        const d: f32 = if (max_val == 0.0) 0.0 else max_val / 127.0;
        const inv_d: f32 = if (d == 0.0) 0.0 else 1.0 / d;
        const block_off = q8_off + b * 34;
        const d_fp16 = fp32ToFp16(d);
        q8_u8[block_off] = @truncate(d_fp16);
        q8_u8[block_off + 1] = @as(u8, @intCast(d_fp16 >> 8));
        inline for (0..8) |batch| {
            const q_base = batch * 4;
            const vals: @Vector(4, f32) = @as(*const [4]f32, @ptrCast(&x[base + q_base])).*;
            const scaled: @Vector(4, f32) = vals * @as(@Vector(4, f32), @splat(inv_d));
            const r0: f32 = scaled[0]; const r1: f32 = scaled[1]; const r2: f32 = scaled[2]; const r3: f32 = scaled[3];
            const v0: i8 = @intFromFloat(if (r0 > 0) r0 + 0.5 else r0 - 0.5);
            const v1: i8 = @intFromFloat(if (r1 > 0) r1 + 0.5 else r1 - 0.5);
            const v2: i8 = @intFromFloat(if (r2 > 0) r2 + 0.5 else r2 - 0.5);
            const v3: i8 = @intFromFloat(if (r3 > 0) r3 + 0.5 else r3 - 0.5);
            const c0: i8 = @intCast(@max(-128, @min(127, @as(i32, v0))));
            const c1: i8 = @intCast(@max(-128, @min(127, @as(i32, v1))));
            const c2: i8 = @intCast(@max(-128, @min(127, @as(i32, v2))));
            const c3: i8 = @intCast(@max(-128, @min(127, @as(i32, v3))));
            q8_u8[block_off + 2 + q_base + 0] = @bitCast(c0); q8_i8[block_off + 2 + q_base + 0] = c0;
            q8_u8[block_off + 2 + q_base + 1] = @bitCast(c1); q8_i8[block_off + 2 + q_base + 1] = c1;
            q8_u8[block_off + 2 + q_base + 2] = @bitCast(c2); q8_i8[block_off + 2 + q_base + 2] = c2;
            q8_u8[block_off + 2 + q_base + 3] = @bitCast(c3); q8_i8[block_off + 2 + q_base + 3] = c3;
        }
    }
}

pub fn dequantizeByte(ctx: *c.Context, src_offset: usize, dst: []f32, dst_offset: usize, count: usize) void {
    const nb = count >> 5;
    const block_size = 2 + c.QK8_0;
    const src = ctx.gguf_uint8.?[src_offset..];
    const src_i8: []const i8 = @ptrCast(src);
    for (0..nb) |i| {
        const bo = i * block_size;
        const d = fp16ToFp32(src[bo] | (@as(u16, src[bo + 1]) << 8));
        for (0..c.QK8_0) |j| {
            dst[dst_offset + i * c.QK8_0 + j] = @floatCast(@as(f64, @floatFromInt(src_i8[bo + 2 + j])) * @as(f64, d));
        }
    }
}

pub fn dequantizeNibble(ctx: *c.Context, src_offset: usize, dst: []f32, dst_offset: usize, count: usize) void {
    const nb = count >> 8;
    const block_size = 2 + 2 + 12 + c.QK_K / 8 + c.QK_K / 2;
    const src = ctx.gguf_uint8.?[src_offset..];
    for (0..nb) |i| {
        const bo = i * block_size;
        const d = @as(f64, fp16ToFp32(src[bo] | (@as(u16, src[bo + 1]) << 8)));
        const dmin = @as(f64, fp16ToFp32(src[bo + 2] | (@as(u16, src[bo + 3]) << 8)));
        const sc_off = bo + 4;
        const qh_off = bo + 16;
        const ql_off = bo + 48;
        const y_base = dst_offset + i * c.QK_K;
        var is_idx: usize = 0;
        var bit1: u32 = 1;
        var bit2: u32 = 2;
        var ql_idx: usize = 0;
        var j: usize = 0;
        while (j < c.QK_K) : (j += 64) {
            var sc: usize = undefined;
            var m_val: usize = undefined;
            if (is_idx < 4) {
                sc = src[sc_off + is_idx] & 63;
                m_val = src[sc_off + is_idx + 4] & 63;
            } else {
                sc = (src[sc_off + is_idx + 4] & 0x0f) | ((src[sc_off + is_idx - 4] >> 6) << 4);
                m_val = (src[sc_off + is_idx + 4] >> 4) | ((src[sc_off + is_idx] >> 6) << 4);
            }
            const d1 = d * @as(f64, @floatFromInt(sc));
            const m1 = dmin * @as(f64, @floatFromInt(m_val));
            is_idx += 1;
            if (is_idx < 4) {
                sc = src[sc_off + is_idx] & 63;
                m_val = src[sc_off + is_idx + 4] & 63;
            } else {
                sc = (src[sc_off + is_idx + 4] & 0x0f) | ((src[sc_off + is_idx - 4] >> 6) << 4);
                m_val = (src[sc_off + is_idx + 4] >> 4) | ((src[sc_off + is_idx] >> 6) << 4);
            }
            const d2 = d * @as(f64, @floatFromInt(sc));
            const m2 = dmin * @as(f64, @floatFromInt(m_val));
            is_idx += 1;
            for (0..32) |l| {
                const q_val = @as(f64, @floatFromInt(src[ql_off + ql_idx + l] & 0x0f)) + if ((src[qh_off + l] & @as(u8, @intCast(bit1))) != 0) @as(f64, 16.0) else @as(f64, 0.0);
                dst[y_base + j + l] = @floatCast(d1 * q_val - m1);
            }
            for (0..32) |l| {
                const q_val = @as(f64, @floatFromInt(src[ql_off + ql_idx + l] >> 4)) + if ((src[qh_off + l] & @as(u8, @intCast(bit2))) != 0) @as(f64, 16.0) else @as(f64, 0.0);
                dst[y_base + j + l + 32] = @floatCast(d2 * q_val - m2);
            }
            ql_idx += 32;
            bit1 <<= 2;
            bit2 <<= 2;
        }
    }
}

pub fn dequantizeWord(ctx: *c.Context, src_offset: usize, dst: []f32, dst_offset: usize, count: usize) void {
    const nb = count >> 8;
    const block_size = c.QK_K / 2 + c.QK_K / 4 + c.QK_K / 16 + 2;
    const src = ctx.gguf_uint8.?[src_offset..];
    const src_i8: []const i8 = @ptrCast(src);
    for (0..nb) |i| {
        const bo = i * block_size;
        const ql_off = bo;
        const qh_off = bo + 128;
        const sc_off = bo + 192;
        const d_off = bo + 208;
        const d = fp16ToFp32(src[d_off] | (@as(u16, src[d_off + 1]) << 8));
        const y = dst_offset + i * c.QK_K;
        for (0..2) |n_val| {
            const n: usize = n_val * 128;
            const ql_base = ql_off + (n >> 1);
            const qh_base = qh_off + (n >> 2);
            const sc_base = sc_off + ((n >> 7) * 8);
            for (0..32) |l| {
                const is_idx = l >> 4;
                const qh_val = src[qh_base + l];
                const q1 = @as(i32, @intCast((src[ql_base + l] & 0x0f) | (((qh_val >> 0) & 0x03) << 4))) - 32;
                const q2 = @as(i32, @intCast((src[ql_base + l + 32] & 0x0f) | (((qh_val >> 2) & 0x03) << 4))) - 32;
                const q3 = @as(i32, @intCast((src[ql_base + l] >> 4) | (((qh_val >> 4) & 0x03) << 4))) - 32;
                const q4 = @as(i32, @intCast((src[ql_base + l + 32] >> 4) | (((qh_val >> 6) & 0x03) << 4))) - 32;
                const s1 = @as(f64, @floatCast(@as(f32, src_i8[sc_base + is_idx + 0])));
                const s2 = @as(f64, @floatCast(@as(f32, src_i8[sc_base + is_idx + 2])));
                const s3 = @as(f64, @floatCast(@as(f32, src_i8[sc_base + is_idx + 4])));
                const s4 = @as(f64, @floatCast(@as(f32, src_i8[sc_base + is_idx + 6])));
                dst[y + n + l] = @floatCast(@as(f64, d) * s1 * @as(f64, @floatFromInt(q1)));
                dst[y + n + l + 32] = @floatCast(@as(f64, d) * s2 * @as(f64, @floatFromInt(q2)));
                dst[y + n + l + 64] = @floatCast(@as(f64, d) * s3 * @as(f64, @floatFromInt(q3)));
                dst[y + n + l + 96] = @floatCast(@as(f64, d) * s4 * @as(f64, @floatFromInt(q4)));
            }
        }
    }
}

pub fn dequantizeRow(ctx: *c.Context, dst: []f32, src_offset: usize, count: usize, ggml_type: c.GGML_TYPE) void {
    switch (ggml_type) {
        .Byte => dequantizeByte(ctx, src_offset, dst, 0, count),
        .Nibble => dequantizeNibble(ctx, src_offset, dst, 0, count),
        .Word => dequantizeWord(ctx, src_offset, dst, 0, count),
        else => {},
    }
}
