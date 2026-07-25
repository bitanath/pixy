const std = @import("std");

pub fn conv2d(
    input: []const f32,
    weight: []const f32,
    bias: []const f32,
    C: usize,
    H: usize,
    W: usize,
    K: usize,
    kh: usize,
    kw: usize,
    pad: usize,
    output: []f32,
) void {
    const H_out = H + 2 * pad + 1 - kh;
    const W_out = W + 2 * pad + 1 - kw;
    const patch_size = C * kh * kw;
    var patch: [144]f32 = undefined;

    for (0..H_out) |y| {
        for (0..W_out) |x| {
            var pi: usize = 0;
            for (0..C) |c| {
                const in_off = c * H * W;
                for (0..kh) |dy| {
                    const iy = y + dy;
                    if (iy < pad or iy >= H + pad) {
                        @memset(patch[pi..][0..kw], 0.0);
                        pi += kw;
                        continue;
                    }
                    for (0..kw) |dx| {
                        const ix = x + dx;
                        if (ix < pad or ix >= W + pad) {
                            patch[pi] = 0.0;
                        } else {
                            patch[pi] = input[in_off + (iy - pad) * W + (ix - pad)];
                        }
                        pi += 1;
                    }
                }
            }

            var k: usize = 0;
            while (k + 4 <= K) : (k += 4) {
                const w_off0 = (k + 0) * patch_size;
                const w_off1 = (k + 1) * patch_size;
                const w_off2 = (k + 2) * patch_size;
                const w_off3 = (k + 3) * patch_size;
                var sum0 = bias[k + 0];
                var sum1 = bias[k + 1];
                var sum2 = bias[k + 2];
                var sum3 = bias[k + 3];
                var j: usize = 0;
                while (j + 4 <= patch_size) : (j += 4) {
                    const pv: @Vector(4, f32) = @as(*const [4]f32, @ptrCast(&patch[j])).*;
                    const w0v: @Vector(4, f32) = @as(*const [4]f32, @ptrCast(&weight[w_off0 + j])).*;
                    const w1v: @Vector(4, f32) = @as(*const [4]f32, @ptrCast(&weight[w_off1 + j])).*;
                    const w2v: @Vector(4, f32) = @as(*const [4]f32, @ptrCast(&weight[w_off2 + j])).*;
                    const w3v: @Vector(4, f32) = @as(*const [4]f32, @ptrCast(&weight[w_off3 + j])).*;
                    sum0 += @reduce(.Add, pv * w0v);
                    sum1 += @reduce(.Add, pv * w1v);
                    sum2 += @reduce(.Add, pv * w2v);
                    sum3 += @reduce(.Add, pv * w3v);
                }
                while (j < patch_size) : (j += 1) {
                    sum0 += patch[j] * weight[w_off0 + j];
                    sum1 += patch[j] * weight[w_off1 + j];
                    sum2 += patch[j] * weight[w_off2 + j];
                    sum3 += patch[j] * weight[w_off3 + j];
                }
                const o_off0 = (k + 0) * H_out * W_out + y * W_out + x;
                const o_off1 = (k + 1) * H_out * W_out + y * W_out + x;
                const o_off2 = (k + 2) * H_out * W_out + y * W_out + x;
                const o_off3 = (k + 3) * H_out * W_out + y * W_out + x;
                output[o_off0] = sum0;
                output[o_off1] = sum1;
                output[o_off2] = sum2;
                output[o_off3] = sum3;
            }
            while (k < K) : (k += 1) {
                const w_off = k * patch_size;
                var sum = bias[k];
                var j: usize = 0;
                while (j + 4 <= patch_size) : (j += 4) {
                    const pv: @Vector(4, f32) = @as(*const [4]f32, @ptrCast(&patch[j])).*;
                    const wv: @Vector(4, f32) = @as(*const [4]f32, @ptrCast(&weight[w_off + j])).*;
                    sum += @reduce(.Add, pv * wv);
                }
                while (j < patch_size) : (j += 1) {
                    sum += patch[j] * weight[w_off + j];
                }
                output[k * H_out * W_out + y * W_out + x] = sum;
            }
        }
    }
}

pub fn maxPool2d(
    input: []const f32,
    C: usize,
    H: usize,
    W: usize,
    pool_size: usize,
    output: []f32,
) void {
    const H_out = H / pool_size;
    const W_out = W / pool_size;

    for (0..C) |c| {
        const in_off = c * H * W;
        const out_off = c * H_out * W_out;
        for (0..H_out) |y| {
            for (0..W_out) |x| {
                var max_val: f32 = -std.math.inf(f32);
                for (0..pool_size) |dy| {
                    for (0..pool_size) |dx| {
                        const val = input[in_off + (y * pool_size + dy) * W + (x * pool_size + dx)];
                        if (val > max_val) max_val = val;
                    }
                }
                output[out_off + y * W_out + x] = max_val;
            }
        }
    }
}

pub fn maxPool2dSimd(
    input: []const f32,
    C: usize,
    H: usize,
    W: usize,
    output: []f32,
) void {
    const H_out = H / 2;
    const W_out = W / 2;

    for (0..C) |c| {
        const in_off = c * H * W;
        const out_off = c * H_out * W_out;
        for (0..H_out) |y| {
            for (0..W_out) |x| {
                const iy = y * 2;
                const ix = x * 2;
                const v00 = input[in_off + iy * W + ix];
                const v01 = input[in_off + iy * W + ix + 1];
                const v10 = input[in_off + (iy + 1) * W + ix];
                const v11 = input[in_off + (iy + 1) * W + ix + 1];
                const v: @Vector(4, f32) = .{ v00, v01, v10, v11 };
                output[out_off + y * W_out + x] = @reduce(.Max, v);
            }
        }
    }
}

pub fn relu(data: []f32) void {
    const zero: @Vector(4, f32) = @splat(0.0);
    var i: usize = 0;
    while (i + 4 <= data.len) : (i += 4) {
        var v: @Vector(4, f32) = @as(*const [4]f32, @ptrCast(&data[i])).*;
        const mask = v > zero;
        v = @select(f32, mask, v, zero);
        @as(*[4]f32, @ptrCast(&data[i])).* = v;
    }
    while (i < data.len) : (i += 1) {
        if (data[i] < 0) data[i] = 0;
    }
}

pub fn linear(
    input: []const f32,
    weight: []const f32,
    bias: []const f32,
    out_features: usize,
    in_features: usize,
    output: []f32,
) void {
    for (0..out_features) |i| {
        const w_row = weight[i * in_features ..][0..in_features];
        var sum: @Vector(4, f32) = @splat(0.0);
        var j: usize = 0;
        while (j + 4 <= in_features) : (j += 4) {
            const iv: @Vector(4, f32) = @as(*const [4]f32, @ptrCast(&input[j])).*;
            const wv: @Vector(4, f32) = @as(*const [4]f32, @ptrCast(&w_row[j])).*;
            sum = @mulAdd(@Vector(4, f32), iv, wv, sum);
        }
        var result = @reduce(.Add, sum) + bias[i];
        while (j < in_features) : (j += 1) {
            result += input[j] * w_row[j];
        }
        output[i] = result;
    }
}
