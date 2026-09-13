const std = @import("std");
const c = @import("constants.zig");
const dm = @import("dequant.zig");
const mt = @import("mats.zig");
const op = @import("ops.zig");
const log = @import("util.zig").log;

const debug = @import("util.zig").debugLogs;
fn debugDumpArr(arr: []const f32) void {
    if (!debug) return;
    const n = arr.len;
    if (n < 6) return;
    log(.{ "[", arr[0], ",", arr[1], ",", arr[2], "]", "[", arr[n/2 - 1], ",", arr[n/2], ",", arr[n/2 + 1], "]", "[", arr[n-3], ",", arr[n-2], ",", arr[n-1], "]" });
}


fn ensureBatchBuffers(ctx: *c.Context, allocator: std.mem.Allocator) void {
    const s = ctx.state orelse return;
    if (s.batch_buffers_ready) return;
    const n = c.PREFILL_BATCH_SIZE;
    const dim = s.batch_dim;
    const max_dim = s.batch_max_dim;
    const q_dim = s.batch_q_dim;
    const kv_dim = s.batch_kv_dim;
    const hidden_dim = s.batch_hidden_dim;
    const x_q8_size = s.batch_x_q8_size;

    for (0..n) |i| {
        s.batch_x[i] = allocator.alloc(f32, dim) catch null;
        s.batch_xb[i] = allocator.alloc(f32, max_dim) catch null;
        s.batch_xb2[i] = allocator.alloc(f32, dim) catch null;
        s.batch_q_arr[i] = allocator.alloc(f32, q_dim) catch null;
        s.batch_k_arr[i] = allocator.alloc(f32, kv_dim) catch null;
        s.batch_v_arr[i] = allocator.alloc(f32, kv_dim) catch null;
        s.batch_hb[i] = allocator.alloc(f32, hidden_dim) catch null;
        s.batch_hb2[i] = allocator.alloc(f32, hidden_dim) catch null;
        const q8_buf = allocator.alloc(u8, x_q8_size) catch null;
        if (q8_buf) |buf| {
            s.batch_q8[i] = buf;
            s.batch_q8i8[i] = @ptrCast(buf);
        }
    }
    s.batch_buffers_ready = true;
}

fn freeBatchBuffers(ctx: *c.Context) void {
    const s = ctx.state orelse return;
    if (!s.batch_buffers_ready) return;
    const n = c.PREFILL_BATCH_SIZE;
    for (0..n) |i| {
        if (s.batch_x[i]) |_| {}
        s.batch_x[i] = null;
        s.batch_xb[i] = null;
        s.batch_xb2[i] = null;
        s.batch_q_arr[i] = null;
        s.batch_k_arr[i] = null;
        s.batch_v_arr[i] = null;
        s.batch_hb[i] = null;
        s.batch_hb2[i] = null;
        s.batch_q8[i] = null;
        s.batch_q8i8[i] = null;
    }
    s.batch_buffers_ready = false;
}

fn ensureLogits(ctx: *c.Context) void {
    const s = ctx.state orelse return;
    if (s.logits != null) return;
    const config = ctx.config orelse return;
    const allocator = ctx.allocator orelse return;
    s.logits = allocator.alloc(f32, config.vocab_size) catch return;
}

fn ensureKvCapacity(ctx: *c.Context, needed: usize, allocator: std.mem.Allocator) void {
    const s = ctx.state orelse return;
    const cap = s.kv_capacity;
    if (needed <= cap) return;
    const max_cap = s.seq_len;
    var new_cap = if (cap > 0) cap else 1;
    while (new_cap < needed) new_cap *= 2;
    if (new_cap > max_cap) new_cap = max_cap;

    const n_layers = s.n_layers;
    const n_kv_heads = s.n_kv_heads;
    const head_bytes_q8 = s.head_bytes_q8;
    const old_head_seq = cap * head_bytes_q8;
    const new_head_seq = new_cap * head_bytes_q8;
    const new_layer_bytes = n_kv_heads * new_head_seq;
    const new_total = n_layers * new_layer_bytes;

    const old_key = s.key_cache orelse return;
    const old_val = s.value_cache orelse return;
    const old_layer_bytes = n_kv_heads * old_head_seq;

    const new_key = allocator.alloc(u8, new_total) catch return;
    const new_val = allocator.alloc(u8, new_total) catch return;
    const new_key_i8: []i8 = @ptrCast(new_key);
    const new_val_i8: []i8 = @ptrCast(new_val);

    for (0..n_layers) |l| {
        const old_layer_base = l * old_layer_bytes;
        const new_layer_base = l * new_layer_bytes;
        for (0..n_kv_heads) |h| {
            const old_head_base = old_layer_base + h * old_head_seq;
            const new_head_base = new_layer_base + h * new_head_seq;
            @memcpy(new_key[new_head_base..][0..old_head_seq], old_key[old_head_base..][0..old_head_seq]);
            @memcpy(new_val[new_head_base..][0..old_head_seq], old_val[old_head_base..][0..old_head_seq]);
        }
    }

    allocator.free(old_key);
    allocator.free(old_val);

    s.key_cache = new_key;
    s.value_cache = new_val;
    s.key_cache_int8 = new_key_i8;
    s.value_cache_int8 = new_val_i8;
    s.head_seq_bytes = new_head_seq;
    s.kv_cache_layer_size = new_layer_bytes;
    s.kv_capacity = new_cap;

    const hk = s.head_kv_byte_offsets orelse return;
    for (0..s.n_heads) |h| hk[h] = @intCast((h / s.kv_mul) * new_head_seq);
}

fn fillRopeBuffers(ctx: *c.Context, start_pos: usize, batch_size: usize) void {
    const s = ctx.state orelse return;
    const rope_size = s.rope_size;
    const freqs = s.rope_freqs orelse return;
    const cos = s.rope_cos_all orelse return;
    const sin = s.rope_sin_all orelse return;

    for (0..batch_size) |b| {
        const pos: f64 = @floatFromInt(start_pos + b);
        const base = b * rope_size;
        for (0..rope_size) |i| {
            const v = pos * @as(f64, freqs[i]);
            cos[base + i] = @floatCast(@cos(v));
            sin[base + i] = @floatCast(@sin(v));
        }
    }
}

fn mamba2Layer(ctx: *c.Context, _: usize, layer: usize, x_arr: []const f32, y_arr: []f32) void {
    const config = ctx.config orelse return;
    const w = ctx.weights orelse return;
    const s = ctx.state orelse return;
    const lw = w.layers[layer];

    const d_conv = config.ssm_d_conv;
    const d_inner = config.ssm_d_inner;
    const d_state = config.ssm_d_state;
    const dt_rank = config.ssm_dt_rank;
    const n_group = config.ssm_n_group;
    const head_dim = if (dt_rank > 0) @divTrunc(d_inner, dt_rank) else d_inner;
    const conv_dim = d_inner + 2 * n_group * d_state;

    const conv_state = s.conv_state[layer] orelse return;
    const ssm_state = s.ssm_state[layer] orelse return;
    const proj_buf = s.ssm_proj_buf orelse return;
    const x_bc_conv = s.ssm_xbc_conv orelse return;
    const y_buf = s.ssm_y_buf orelse return;

    const conv1d_w = lw.ssm_conv1d_w orelse return;
    const conv1d_b = lw.ssm_conv1d_b orelse return;
    const dt_bias = lw.ssm_dt_b orelse return;
    const a_param = lw.ssm_a orelse return;
    const d_param = lw.ssm_d orelse return;
    const ssm_norm_w = lw.ssm_norm orelse return;
    const ssm_out_w = lw.ssm_out orelse return;

    const ssm_in = lw.ssm_in orelse return;

    mt.matmulQuantized(ctx, proj_buf, x_arr, ssm_in);

    const z_arr = proj_buf[0..d_inner];
    const x_bc_arr = proj_buf[d_inner..][0..conv_dim];
    const dt_arr = proj_buf[d_inner + conv_dim ..][0..dt_rank];

    const conv_history_len = d_conv - 1;
    for (0..conv_dim) |conv_i| {
        var sum: f64 = @floatCast(conv1d_b[conv_i]);
        for (0..conv_history_len) |k| {
            sum += @as(f64, conv1d_w[conv_i * d_conv + k]) * @as(f64, conv_state[conv_i * conv_history_len + k]);
        }
        sum += @as(f64, conv1d_w[conv_i * d_conv + conv_history_len]) * @as(f64, x_bc_arr[conv_i]);
        x_bc_conv[conv_i] = @floatCast(sum);
    }

    for (0..conv_dim) |ci| {
        const val: f64 = @floatCast(x_bc_conv[ci]);
        x_bc_conv[ci] = @floatCast(val * (1.0 / (1.0 + op.fastExpf(-val))));
    }
    for (0..conv_dim) |cj| {
        for (0..conv_history_len - 1) |i| {
            conv_state[cj * conv_history_len + i] = conv_state[cj * conv_history_len + i + 1];
        }
        conv_state[cj * conv_history_len + (conv_history_len - 1)] = x_bc_arr[cj];
    }

    const x_ssm = x_bc_conv[0..d_inner];
    const b_ssm = x_bc_conv[d_inner..][0..n_group * d_state];
    const c_ssm = x_bc_conv[d_inner + n_group * d_state ..][0..n_group * d_state];

    for (0..dt_rank) |h| {
        dt_arr[h] += dt_bias[h];
    }

    @memset(y_buf[0..d_inner], 0.0);

    const heads_per_group = @divTrunc(dt_rank, n_group);

    for (0..dt_rank) |h| {
        const dt_val: f64 = @floatCast(dt_arr[h]);
        const dt_softplus: f64 = if (dt_val > 20.0) dt_val else @log(1.0 + @exp(dt_val));
        const d_a_v: @Vector(4, f64) = @splat(@exp(dt_softplus * @as(f64, a_param[h])));
        const g = @divTrunc(h, heads_per_group);

        for (0..head_dim) |d| {
            const dim_idx = d + h * head_dim;
            const x_dt: f64 = @as(f64, x_ssm[dim_idx]) * dt_softplus;
            const state_base = dim_idx * d_state;
            const bg = g * d_state;

            const x_dt_v: @Vector(4, f64) = @splat(x_dt);

            var sumf: @Vector(4, f64) = @splat(0.0);

            var k: usize = 0;
            while (k + 4 <= d_state) : (k += 4) {
                const sv32: @Vector(4, f32) = @as(*const [4]f32, @ptrCast(&ssm_state[state_base + k])).*;
                const sv: @Vector(4, f64) = @floatCast(sv32);
                const bv32: @Vector(4, f32) = @as(*const [4]f32, @ptrCast(&b_ssm[bg + k])).*;
                const bv: @Vector(4, f64) = @floatCast(bv32);
                const cv32: @Vector(4, f32) = @as(*const [4]f32, @ptrCast(&c_ssm[bg + k])).*;
                const cv: @Vector(4, f64) = @floatCast(cv32);

                const state_new: @Vector(4, f64) = sv * d_a_v + bv * x_dt_v;
                const result_v: @Vector(4, f32) = @floatCast(state_new);
                @as(*[4]f32, @ptrCast(&ssm_state[state_base + k])).* = result_v;
                sumf += state_new * cv;
            }
            while (k < d_state) : (k += 1) {
                const d_a: f64 = d_a_v[0];
                const state_new: f64 = @as(f64, ssm_state[state_base + k]) * d_a + @as(f64, b_ssm[bg + k]) * x_dt;
                ssm_state[state_base + k] = @floatCast(state_new);
                sumf[0] += state_new * @as(f64, c_ssm[bg + k]);
            }
            y_buf[dim_idx] = @floatCast(@reduce(.Add, sumf));
        }
    }

    for (0..dt_rank) |h| {
        const d_val: f64 = @floatCast(d_param[h]);
        for (0..head_dim) |d| {
            const dim_idx = d + h * head_dim;
            y_buf[dim_idx] = @floatCast(@as(f64, y_buf[dim_idx]) + @as(f64, x_ssm[dim_idx]) * d_val);
        }
    }

    for (0..d_inner) |i| {
        const zv: f64 = @floatCast(z_arr[i]);
        y_buf[i] = @floatCast(zv * (1.0 / (1.0 + op.fastExpf(-zv))) * @as(f64, y_buf[i]));
    }
    {
        var has_ssm_norm = false;
        for (ssm_norm_w) |v| {
            if (@as(f64, @floatCast(v)) != 0.0) has_ssm_norm = true;
        }
        if (has_ssm_norm) {
            const inv_d_inner: f64 = 1.0 / @as(f64, @floatFromInt(d_inner));
            if (n_group == 1) {
                var ss: f64 = 0.0;
                for (y_buf[0..d_inner]) |v| ss += @as(f64, v) * @as(f64, v);
                ss = 1.0 / @sqrt(ss * inv_d_inner + 1e-5);
                for (y_buf[0..d_inner], ssm_norm_w) |*yv, nw| {
                    yv.* = @floatCast(@as(f64, nw) * (ss * @as(f64, yv.*)));
                }
            } else {
                const group_size = @divTrunc(d_inner, n_group);
                const inv_group_size: f64 = 1.0 / @as(f64, @floatFromInt(group_size));
                for (0..n_group) |g| {
                    var ss: f64 = 0.0;
                    const base = g * group_size;
                    for (y_buf[base..][0..group_size]) |v| ss += @as(f64, v) * @as(f64, v);
                    ss = 1.0 / @sqrt(ss * inv_group_size + 1e-5);
                    for (y_buf[base..][0..group_size], ssm_norm_w[base..][0..group_size]) |*yv, nw| {
                        yv.* = @floatCast(@as(f64, nw) * (ss * @as(f64, yv.*)));
                    }
                }
            }
        }
    }

    mt.matmulQuantized(ctx, y_arr, y_buf, ssm_out_w);
}

fn transformerLlama(ctx: *c.Context, token: i32, pos: usize, compute_logits: bool) void {
    if (debug) { ts(); log(.{ "[TL] token=", token, "pos=", pos, "compute_logits=", compute_logits }); }
    const config = ctx.config orelse return;
    const w = ctx.weights orelse return;
    const s = ctx.state orelse return;

    ensureKvCapacity(ctx, pos + 1, ctx.allocator orelse return);

    const dim = s.dim;
    const head_size = config.head_dim;
    const q_dim = s.n_heads * head_size;
    const hidden_dim = s.hidden_dim;
    const n_layers = s.n_layers;
    const n_kv_heads = s.n_kv_heads;
    const inv_dim = s.inv_dim;
    const kv_mul = s.kv_mul;
    const head_bytes_q8 = s.head_bytes_q8;
    const head_seq_bytes = s.head_seq_bytes;
    const attn_scale = s.attn_scale;
    const seq_len = s.seq_len;

    const x_arr = s.x orelse return;
    const xb_arr = s.xb orelse return;
    const xb2_arr = s.xb2 orelse return;
    const q_arr = s.q orelse return;
    const k_arr = s.k orelse return;
    const v_arr = s.v orelse return;
    const s_att = s.att orelse return;
    const key_cache = s.key_cache orelse return;
    const value_cache = s.value_cache orelse return;
    const key_cache_int8 = s.key_cache_int8 orelse return;
    const value_cache_int8 = s.value_cache_int8 orelse return;
    const q_q8 = s.q_q8 orelse return;
    const q_q8i8 = s.q_q8i8 orelse return;

    const emb = w.token_embedding orelse return;
    dm.dequantizeRow(ctx, x_arr, emb.data_offset + @as(usize, @intCast(token)) * emb.row_size, dim, emb.quant_type);

    fillRopeBuffers(ctx, pos, 1);

    for (0..n_layers) |l| {
        const lw = w.layers[l];

        op.rmsnorm(xb_arr, x_arr, lw.rms_att_weight orelse return, dim, inv_dim, config.rms_norm_eps);

        const wq = lw.wq orelse return;
        const wk = lw.wk orelse return;
        const wv = lw.wv orelse return;

        if (wq.dot_q8_func != null and wq.deq_row_func == null and
            wk.dot_q8_func != null and wk.deq_row_func == null and
            wv.dot_q8_func != null and wv.deq_row_func == null)
        {
            if (debug and l == 0) { ts(); log(.{ "[DBG] pos", pos, "_l", l, "_qkv_path=preQ8 quant=", wq.quant_type }); }
            const x_q8_buf = ctx.x_q8_buf orelse return;
            const x_q8_int8_buf = ctx.x_q8_int8_buf orelse return;
            dm.quantToByteCache(xb_arr, 0, x_q8_buf, x_q8_int8_buf, 0, dim);
            mt.matmulQuantizedPreQ8(ctx, q_arr, wq, x_q8_buf, x_q8_int8_buf);
            mt.matmulQuantizedPreQ8(ctx, k_arr, wk, x_q8_buf, x_q8_int8_buf);
            mt.matmulQuantizedPreQ8(ctx, v_arr, wv, x_q8_buf, x_q8_int8_buf);
        } else {
            if (debug and l == 0) { ts(); log(.{ "[DBG] pos", pos, "_l", l, "_qkv_path=generic quant=", wq.quant_type }); }
            mt.matmulQuantized(ctx, q_arr, xb_arr, wq);
            mt.matmulQuantized(ctx, k_arr, xb_arr, wk);
            mt.matmulQuantized(ctx, v_arr, xb_arr, wv);
        }


        const half = head_size >> 1;
        const rope_cos = s.rope_cos_layer[l] orelse return;
        const rope_sin = s.rope_sin_layer[l] orelse return;

        for (0..s.n_heads) |_h| {
            const q_base = _h * head_size;
            for (0..half) |i| {
                const fcr: f32 = rope_cos[i];
                const fci: f32 = rope_sin[i];
                const v0: f32 = q_arr[q_base + i];
                const v1: f32 = q_arr[q_base + i + half];
                q_arr[q_base + i] = (v0 * fcr - v1 * fci) * attn_scale;
                q_arr[q_base + i + half] = (v0 * fci + v1 * fcr) * attn_scale;
            }
        }
        for (0..n_kv_heads) |_h| {
            const k_base = _h * head_size;
            for (0..half) |i| {
                const fcr: f32 = rope_cos[i];
                const fci: f32 = rope_sin[i];
                const v0: f32 = k_arr[k_base + i];
                const v1: f32 = k_arr[k_base + i + half];
                k_arr[k_base + i] = v0 * fcr - v1 * fci;
                k_arr[k_base + i + half] = v0 * fci + v1 * fcr;
            }
        }

        const l_off = l * s.kv_cache_layer_size;
        for (0..n_kv_heads) |h| {
            const head_off = l_off + h * head_seq_bytes + pos * head_bytes_q8;
            dm.quantToByteCache(k_arr, h * head_size, key_cache, key_cache_int8, head_off, head_size);
            dm.quantToByteCache(v_arr, h * head_size, value_cache, value_cache_int8, head_off, head_size);
        }
        dm.quantToByteCache(q_arr, 0, q_q8, q_q8i8, 0, q_dim);

        @memset(xb_arr[0..q_dim], 0.0);

        for (0..n_kv_heads) |kv_h| {
            const k_base = l_off + kv_h * head_seq_bytes;

            for (0..pos + 1) |t| {
                const k_off = k_base + t * head_bytes_q8;
                for (0..kv_mul) |mh| {
                    const h = kv_h * kv_mul + mh;
                    s_att[h * seq_len + t] = dm.dotByteByteCache(q_q8, q_q8i8, h * head_bytes_q8, key_cache, key_cache_int8, k_off, head_size);
                }
            }

            for (0..kv_mul) |mh| {
                const h = kv_h * kv_mul + mh;
                    const att_offset = h * seq_len;
                    const softmax_end = att_offset + pos;

                    var max_val: f32 = s_att[att_offset];
                    var ai = att_offset + 1;
                    while (ai <= softmax_end) : (ai += 1) {
                        if (s_att[ai] > max_val) max_val = s_att[ai];
                    }
                    var exp_sum: f32 = 0.0;
                    ai = att_offset;
                    while (ai <= softmax_end) : (ai += 1) {
                        const e = @exp(s_att[ai] - max_val);
                        s_att[ai] = e;
                        exp_sum += e;
                    }
                    const inv_sum: f32 = 1.0 / exp_sum;
                    ai = att_offset;
                    while (ai <= softmax_end) : (ai += 1) s_att[ai] = s_att[ai] * inv_sum;

                    const xb_offset = h * head_size;
                    for (0..pos + 1) |t| {
                        dm.accumByteCache(xb_arr, xb_offset, value_cache, value_cache_int8, k_base + t * head_bytes_q8, s_att[att_offset + t], head_size);
                    }
            }
        }

        const wo = lw.wo orelse return;
        mt.matmulQuantized(ctx, xb2_arr, xb_arr, wo);

        op.rmsnorm(xb_arr, x_arr, lw.rms_att_weight orelse return, dim, inv_dim, config.rms_norm_eps);
        mamba2Layer(ctx, pos, l, xb_arr, s.hb orelse return);

        for (0..dim) |_i| {
            xb2_arr[_i] = xb2_arr[_i] + (s.hb orelse return)[_i];
        }
        op.accum(x_arr, xb2_arr, dim);

        const rms_ffn = lw.rms_ffn_weight orelse return;
        op.rmsnorm(xb_arr, x_arr, rms_ffn, dim, inv_dim, config.rms_norm_eps);

        const hb_arr = s.hb orelse return;
        const hb2_arr = s.hb2 orelse return;
        const w1 = lw.w1 orelse return;
        const w3 = lw.w3 orelse return;

        // FASTPATH: prefer deq_row_func for Word (SIMD deqRow beats scalar vecDot)
        if (w1.dot_q8_func != null and w1.deq_row_func == null and
            w3.dot_q8_func != null and w3.deq_row_func == null)
        {
            const x_q8_buf = ctx.x_q8_buf orelse return;
            const x_q8_int8_buf = ctx.x_q8_int8_buf orelse return;
            dm.quantToByteCache(xb_arr, 0, x_q8_buf, x_q8_int8_buf, 0, dim);
            mt.matmulQuantizedPreQ8(ctx, hb_arr, w1, x_q8_buf, x_q8_int8_buf);
            mt.matmulQuantizedPreQ8(ctx, hb2_arr, w3, x_q8_buf, x_q8_int8_buf);
        } else {
            mt.matmulQuantized(ctx, hb_arr, xb_arr, w1);
            mt.matmulQuantized(ctx, hb2_arr, xb_arr, w3);
        }

        const hd4 = hidden_dim & ~@as(usize, 3);
        var hi: usize = 0;
        while (hi < hd4) : (hi += 4) {
            const v0: f32 = hb_arr[hi];
            const v1: f32 = hb_arr[hi + 1];
            const v2: f32 = hb_arr[hi + 2];
            const v3: f32 = hb_arr[hi + 3];
            hb_arr[hi] = 0.5 * v0 * (1.0 + op.fastTanh(0.5 * v0)) * hb2_arr[hi];
            hb_arr[hi + 1] = 0.5 * v1 * (1.0 + op.fastTanh(0.5 * v1)) * hb2_arr[hi + 1];
            hb_arr[hi + 2] = 0.5 * v2 * (1.0 + op.fastTanh(0.5 * v2)) * hb2_arr[hi + 2];
            hb_arr[hi + 3] = 0.5 * v3 * (1.0 + op.fastTanh(0.5 * v3)) * hb2_arr[hi + 3];
        }
        while (hi < hidden_dim) : (hi += 1) {
            const val: f32 = hb_arr[hi];
            hb_arr[hi] = 0.5 * val * (1.0 + op.fastTanh(0.5 * val)) * hb2_arr[hi];
        }
        const w2 = lw.w2 orelse return;
        mt.matmulQuantized(ctx, xb_arr, hb_arr, w2);
        op.accum(x_arr, xb_arr, dim);
    }

    const rms_final = w.rms_final_weight orelse return;
    op.rmsnorm(x_arr, x_arr, rms_final, dim, inv_dim, config.rms_norm_eps);
    if (debug) { ts(); log(.{ "[DBG] pos", pos, "_final_norm" }); debugDumpArr(x_arr); }

    if (compute_logits) {
        const logits = s.logits orelse blk: {
            const allocator = ctx.allocator orelse return;
            const buf = allocator.alloc(f32, s.vocab_size) catch return;
            s.logits = buf;
            break :blk buf;
        };
        const wcls = w.wcls orelse return;
        mt.matmulQuantized(ctx, logits, x_arr, wcls);
        if (debug) { ts(); log(.{ "[DBG] pos", pos, "_logits" }); debugDumpArr(logits); }
        if (pos >= 77 and pos <= 85) {
            log(.{ "[CK] pos", pos, "_logits[0..5]" });
            for (0..5) |i| log(.{ logits[i] });
        }
    }
}


fn transformerPrefillLlama(ctx: *c.Context, all_tokens: []const i32, start_pos: usize, batch_size: usize, allocator: std.mem.Allocator) void {
    const config = ctx.config orelse return;
    const w = ctx.weights orelse return;
    const s = ctx.state orelse return;

    ensureKvCapacity(ctx, start_pos + batch_size, allocator);
    ensureBatchBuffers(ctx, allocator);

    if (debug) { ts(); log(.{ "[PF] prefill start pos=", start_pos, "batch=", batch_size }); }

    const dim = s.dim;
    const head_size = config.head_dim;
    const q_dim = s.n_heads * head_size;
    const hidden_dim = s.hidden_dim;
    const n_layers = s.n_layers;
    const n_kv_heads = s.n_kv_heads;
    const inv_dim = s.inv_dim;
    const kv_mul = s.kv_mul;
    const head_bytes_q8 = s.head_bytes_q8;
    const head_seq_bytes = s.head_seq_bytes;
    const attn_scale = s.attn_scale;
    const seq_len = s.seq_len;

    const key_cache = s.key_cache orelse return;
    const value_cache = s.value_cache orelse return;
    const key_cache_int8 = s.key_cache_int8 orelse return;
    const value_cache_int8 = s.value_cache_int8 orelse return;
    const q_q8 = s.q_q8 orelse return;
    const q_q8i8 = s.q_q8i8 orelse return;
    const s_att = s.att orelse return;

    const emb = w.token_embedding orelse return;
    for (0..batch_size) |b| {
        const arr = s.batch_x[b] orelse return;
        dm.dequantizeRow(ctx, arr, emb.data_offset + @as(usize, @intCast(all_tokens[start_pos + b])) * emb.row_size, dim, emb.quant_type);
    }

    fillRopeBuffers(ctx, start_pos, batch_size);

    for (0..n_layers) |l| {
        const lw = w.layers[l];

        for (0..batch_size) |b| {
            const bx = s.batch_x[b] orelse return;
            const bxb = s.batch_xb[b] orelse return;
            op.rmsnorm(bxb, bx, lw.rms_att_weight orelse return, dim, inv_dim, config.rms_norm_eps);
        }

        var b_q_nn: [c.PREFILL_BATCH_SIZE][]f32 = undefined;
        var b_k_nn: [c.PREFILL_BATCH_SIZE][]f32 = undefined;
        var b_v_nn: [c.PREFILL_BATCH_SIZE][]f32 = undefined;
        var b_xb_nn: [c.PREFILL_BATCH_SIZE][]f32 = undefined;
        var b_xb2_nn: [c.PREFILL_BATCH_SIZE][]f32 = undefined;
        var b_hb_nn: [c.PREFILL_BATCH_SIZE][]f32 = undefined;
        var b_hb2_nn: [c.PREFILL_BATCH_SIZE][]f32 = undefined;
        for (0..batch_size) |i| {
            b_q_nn[i] = s.batch_q_arr[i] orelse return;
            b_k_nn[i] = s.batch_k_arr[i] orelse return;
            b_v_nn[i] = s.batch_v_arr[i] orelse return;
            b_xb_nn[i] = s.batch_xb[i] orelse return;
            b_xb2_nn[i] = s.batch_xb2[i] orelse return;
            b_hb_nn[i] = s.batch_hb[i] orelse return;
            b_hb2_nn[i] = s.batch_hb2[i] orelse return;
        }

        const wq = lw.wq orelse return;
        const wk = lw.wk orelse return;
        const wv = lw.wv orelse return;

        mt.matmulQuantizedBatch(ctx, b_q_nn[0..batch_size], b_xb_nn[0..batch_size], wq, batch_size);
        mt.matmulQuantizedBatch(ctx, b_k_nn[0..batch_size], b_xb_nn[0..batch_size], wk, batch_size);
        mt.matmulQuantizedBatch(ctx, b_v_nn[0..batch_size], b_xb_nn[0..batch_size], wv, batch_size);

        const half = head_size >> 1;
        const rope_cos = s.rope_cos_layer[l] orelse return;
        const rope_sin = s.rope_sin_layer[l] orelse return;
        const l_off = l * s.kv_cache_layer_size;

        for (0..batch_size) |b| {
            const pos = start_pos + b;
            const q_arr = b_q_nn[b];
            const k_arr = b_k_nn[b];
            const v_arr = b_v_nn[b];
            const xb_arr = b_xb_nn[b];

            const rope_base = b * s.rope_size;

            for (0..s.n_heads) |_h| {
                const q_base = _h * head_size;
                for (0..half) |i| {
                    const fcr: f32 = rope_cos[rope_base + i];
                    const fci: f32 = rope_sin[rope_base + i];
                    const v0: f32 = q_arr[q_base + i];
                    const v1: f32 = q_arr[q_base + i + half];
                    q_arr[q_base + i] = (v0 * fcr - v1 * fci) * attn_scale;
                    q_arr[q_base + i + half] = (v0 * fci + v1 * fcr) * attn_scale;
                }
            }
            for (0..n_kv_heads) |_h| {
                const k_base = _h * head_size;
                for (0..half) |i| {
                    const fcr: f32 = rope_cos[rope_base + i];
                    const fci: f32 = rope_sin[rope_base + i];
                    const v0: f32 = k_arr[k_base + i];
                    const v1: f32 = k_arr[k_base + i + half];
                    k_arr[k_base + i] = v0 * fcr - v1 * fci;
                    k_arr[k_base + i + half] = v0 * fci + v1 * fcr;
                }
            }

            for (0..n_kv_heads) |h| {
                const head_off = l_off + h * head_seq_bytes + pos * head_bytes_q8;
                dm.quantToByteCache(k_arr, h * head_size, key_cache, key_cache_int8, head_off, head_size);
                dm.quantToByteCache(v_arr, h * head_size, value_cache, value_cache_int8, head_off, head_size);
            }
            dm.quantToByteCache(q_arr, 0, q_q8, q_q8i8, 0, q_dim);

            @memset(xb_arr[0..q_dim], 0.0);

            for (0..n_kv_heads) |kv_h| {
                const k_base = l_off + kv_h * head_seq_bytes;
                for (0..pos + 1) |t| {
                    const k_off = k_base + t * head_bytes_q8;
                    for (0..kv_mul) |mh| {
                        const h = kv_h * kv_mul + mh;
s_att[h * seq_len + t] = dm.dotByteByteCache(q_q8, q_q8i8, h * head_bytes_q8, key_cache, key_cache_int8, k_off, head_size);
                    }
                }
                for (0..kv_mul) |mh| {
                    const h = kv_h * kv_mul + mh;
                    const att_offset = h * seq_len;
                    const softmax_end = att_offset + pos;
                    var max_val: f32 = s_att[att_offset];
                    var ai = att_offset + 1;
                    while (ai <= softmax_end) : (ai += 1) {
                        if (s_att[ai] > max_val) max_val = s_att[ai];
                    }
                    var exp_sum: f32 = 0.0;
                    ai = att_offset;
                    while (ai <= softmax_end) : (ai += 1) {
                        const e = @exp(s_att[ai] - max_val);
                        s_att[ai] = e;
                        exp_sum += e;
                    }
                    const inv_sum: f32 = 1.0 / exp_sum;
                    ai = att_offset;
                    while (ai <= softmax_end) : (ai += 1) s_att[ai] = s_att[ai] * inv_sum;
                    const xb_offset = h * head_size;
                    for (0..pos + 1) |t| {
                        dm.accumByteCache(xb_arr, xb_offset, value_cache, value_cache_int8, k_base + t * head_bytes_q8, s_att[att_offset + t], head_size);
                    }
                }
            }
        }

        const wo = lw.wo orelse return;
        mt.matmulQuantizedBatch(ctx, b_xb2_nn[0..batch_size], b_xb_nn[0..batch_size], wo, batch_size);

        {
            const hb_arr = s.hb orelse return;
            for (0..batch_size) |b| {
                const bx = s.batch_x[b] orelse return;
                const bxb = s.batch_xb[b] orelse return;
                op.rmsnorm(bxb, bx, lw.rms_att_weight orelse return, dim, inv_dim, config.rms_norm_eps);
                mamba2Layer(ctx, 0, l, bxb, hb_arr);
                for (0..dim) |i| {
                    b_xb2_nn[b][i] += hb_arr[i];
                }
            }
        }

        for (0..batch_size) |b| {
            op.accum(s.batch_x[b] orelse return, b_xb2_nn[b], dim);
        }
        for (0..batch_size) |b| {
            const bx = s.batch_x[b] orelse return;
            const bxb = s.batch_xb[b] orelse return;
            op.rmsnorm(bxb, bx, lw.rms_ffn_weight orelse return, dim, inv_dim, config.rms_norm_eps);
        }

        const w1 = lw.w1 orelse return;
        const w3 = lw.w3 orelse return;
        mt.matmulQuantizedBatch(ctx, b_hb_nn[0..batch_size], b_xb_nn[0..batch_size], w1, batch_size);
        mt.matmulQuantizedBatch(ctx, b_hb2_nn[0..batch_size], b_xb_nn[0..batch_size], w3, batch_size);

        const hd4 = hidden_dim & ~@as(usize, 3);
        for (0..batch_size) |b| {
            const hb_arr = b_hb_nn[b];
            const hb2_arr = b_hb2_nn[b];
            var hi: usize = 0;
            while (hi < hd4) : (hi += 4) {
                const v0: f32 = hb_arr[hi];
                const v1: f32 = hb_arr[hi + 1];
                const v2: f32 = hb_arr[hi + 2];
                const v3: f32 = hb_arr[hi + 3];
                hb_arr[hi] = 0.5 * v0 * (1.0 + op.fastTanh(0.5 * v0)) * hb2_arr[hi];
                hb_arr[hi + 1] = 0.5 * v1 * (1.0 + op.fastTanh(0.5 * v1)) * hb2_arr[hi + 1];
                hb_arr[hi + 2] = 0.5 * v2 * (1.0 + op.fastTanh(0.5 * v2)) * hb2_arr[hi + 2];
                hb_arr[hi + 3] = 0.5 * v3 * (1.0 + op.fastTanh(0.5 * v3)) * hb2_arr[hi + 3];
            }
            while (hi < hidden_dim) : (hi += 1) {
                const val: f32 = hb_arr[hi];
                hb_arr[hi] = 0.5 * val * (1.0 + op.fastTanh(0.5 * val)) * hb2_arr[hi];
            }
        }

        const w2 = lw.w2 orelse return;
        mt.matmulQuantizedBatch(ctx, b_xb_nn[0..batch_size], b_hb_nn[0..batch_size], w2, batch_size);
        for (0..batch_size) |b| {
            op.accum(s.batch_x[b] orelse return, b_xb_nn[b], dim);
        }
    }

    if (debug) { ts(); log(.{ "[PF] prefill done" }); }
}


pub fn transformer(ctx: *c.Context, token: i32, pos: usize, compute_logits: bool) void {
    transformerLlama(ctx, token, pos, compute_logits);
}

pub fn transformerPrefill(ctx: *c.Context, all_tokens: []const i32, start_pos: usize, batch_size: usize, allocator: std.mem.Allocator) void {
    transformerPrefillLlama(ctx, all_tokens, start_pos, batch_size, allocator);
}

fn randomF32() f64 {
    var prng = std.Random.DefaultPrng.init(0);
    return prng.random().float(f64);
}

fn sampleArgmax(logits: []const f32, n: usize) i32 {
    var max_i: i32 = 0;
    var max_p: f64 = @floatCast(logits[0]);
    for (1..n) |i| {
        if (logits[i] > max_p) {
            max_i = @intCast(i);
            max_p = @floatCast(logits[i]);
        }
    }
    return max_i;
}

fn sample(ctx: *c.Context, logits: []const f32, temp: f64) i32 {
    const config = ctx.config orelse return 0;
    if (temp == 0.0) return sampleArgmax(logits, config.vocab_size);
    const vocab_size = config.vocab_size;
    var k: usize = @intCast(ctx.top_k);
    if (k > vocab_size) k = vocab_size;
    const s = ctx.state orelse return 0;
    const top_k_idx = s.top_k_indices orelse return 0;
    const top_k_val = s.top_k_values orelse return 0;

    for (0..k) |i| {
        top_k_idx[i] = @intCast(i);
        top_k_val[i] = logits[i];
    }

    var min_pos: usize = 0;
    var min_val: f64 = @floatCast(top_k_val[0]);
    for (1..k) |i| {
        if (top_k_val[i] < min_val) {
            min_val = @floatCast(top_k_val[i]);
            min_pos = i;
        }
    }

    for (k..vocab_size) |i| {
        if (logits[i] > min_val) {
            top_k_val[min_pos] = logits[i];
            top_k_idx[min_pos] = @intCast(i);
            min_val = @floatCast(top_k_val[0]);
            min_pos = 0;
            for (1..k) |j| {
                if (top_k_val[j] < min_val) {
                    min_val = @floatCast(top_k_val[j]);
                    min_pos = j;
                }
            }
        }
    }

    const inv_temp: f64 = 1.0 / temp;
    var max_v: f64 = @floatCast(top_k_val[0]);
    for (1..k) |i| {
        if (top_k_val[i] > max_v) max_v = @floatCast(top_k_val[i]);
    }
    const max_vt = max_v * inv_temp;
    var sum: f64 = 0.0;
    for (0..k) |i| {
        const e = @exp(@as(f64, @floatCast(top_k_val[i])) * inv_temp - max_vt);
        top_k_val[i] = @floatCast(e);
        sum += e;
    }

    var n: usize = k;
    if (ctx.top_p < 1.0) {
        for (0..k) |i| {
            var j = i;
            while (j > 0 and top_k_val[j - 1] < top_k_val[j]) : (j -= 1) {
                const tmp_v = top_k_val[j];
                const tmp_i = top_k_idx[j];
                top_k_val[j] = top_k_val[j - 1];
                top_k_idx[j] = top_k_idx[j - 1];
                top_k_val[j - 1] = tmp_v;
                top_k_idx[j - 1] = tmp_i;
            }
        }

        var cum_sum: f64 = 0.0;
        const threshold = ctx.top_p * sum;
        n = k;
        for (0..k) |i| {
            cum_sum += @floatCast(top_k_val[i]);
            if (cum_sum >= threshold) {
                n = i + 1;
                break;
            }
        }

        sum = 0.0;
        for (0..n) |i| sum += @floatCast(top_k_val[i]);
    }

    const r: f64 = randomF32() * sum;
    var cdf: f64 = 0.0;
    for (0..n) |i| {
        cdf += @floatCast(top_k_val[i]);
        if (r < cdf) return top_k_idx[i];
    }
    return top_k_idx[n - 1];
}


fn vocabOffsetOf(ctx: *c.Context, i: usize) usize {
    const t = ctx.tokenizer orelse return 0;
    const bucket = i >> 8;
    var base: usize = @intCast(t.vocab_sparse_cum[bucket]);
    const lengths = t.vocab_lengths;
    const bucket_start = bucket << 8;
    var k = bucket_start;
    while (k < i) : (k += 1) {
        base += @as(usize, lengths[k]) + 8;
    }
    return base;
}

fn vocabString(ctx: *c.Context, i: usize) []const u8 {
    const t = ctx.tokenizer orelse return "";
    if (i >= t.vocab_size) return "";
    const len: usize = t.vocab_lengths[i];
    if (len == 0) return "";
    const off = vocabOffsetOf(ctx, i);
    const u8_data = ctx.gguf_uint8 orelse return "";
    return u8_data[off..][0..len];
}

fn parseHexByte(hex: []const u8) u8 {
    var val: u8 = 0;
    for (0..2) |i| {
        const ch = hex[i];
        val <<= 4;
        if (ch >= '0' and ch <= '9') {
            val |= (ch - '0');
        } else if (ch >= 'A' and ch <= 'F') {
            val |= (ch - ('A' - 10));
        } else if (ch >= 'a' and ch <= 'f') {
            val |= (ch - ('a' - 10));
        }
    }
    return val;
}

fn compareTokens(a: i32, b: i32, sk: []const u32, lens: []const u16, offs: []const u32, u8_data: []const u8) i32 {
    const la: i32 = @intCast(lens[@intCast(a)]);
    const lb: i32 = @intCast(lens[@intCast(b)]);
    const oa: i32 = @intCast(offs[@intCast(a)]);
    const ob: i32 = @intCast(offs[@intCast(b)]);
    const ml: i32 = if (la < lb) la else lb;
    const first: i32 = if (ml < 4) ml else 4;

    for (0..@as(usize, @intCast(first))) |k| {
        const d: i32 = @as(i32, u8_data[@as(usize, @intCast(oa + @as(i32, @intCast(k))))]) - @as(i32, u8_data[@as(usize, @intCast(ob + @as(i32, @intCast(k))))]);
        if (d != 0) return d;
    }

    if (ml >= 4) {
        const ka = sk[@intCast(a)];
        const kb = sk[@intCast(b)];
        if (ka < kb) return -1;
        if (ka > kb) return 1;
    }

    var k: i32 = 4;
    while (k < ml) : (k += 1) {
        const d: i32 = @as(i32, u8_data[@as(usize, @intCast(oa + k))]) - @as(i32, u8_data[@as(usize, @intCast(ob + k))]);
        if (d != 0) return d;
    }

    return la - lb;
}

fn buildSortedVocab(ctx: *c.Context, allocator: std.mem.Allocator) void {
    if (ctx.trie_node_id != null) return;
    const t = ctx.tokenizer orelse return;
    const vocab_len = t.vocab_size;
    const lengths = t.vocab_lengths;
    const u8_data = ctx.gguf_uint8 orelse return;
    const sparse_cum = t.vocab_sparse_cum;

    const offsets = allocator.alloc(u32, vocab_len) catch return;
    defer allocator.free(offsets);

    for (0..sparse_cum.len) |bk| {
        var acc: u32 = sparse_cum[bk];
        var end = (bk + 1) << 8;
        if (end > vocab_len) end = vocab_len;
        var i = bk << 8;
        while (i < end) : (i += 1) {
            offsets[i] = acc;
            acc += @as(u32, lengths[i]) + 8;
        }
    }

    var count: usize = 0;
    var max_len: usize = 0;
    for (0..vocab_len) |i| {
        const li = lengths[i];
        if (li > 0) {
            count += 1;
            if (li > max_len) max_len = li;
        }
    }

    const indices = allocator.alloc(i32, count) catch return;
    defer allocator.free(indices);
    var w: usize = 0;
    for (0..vocab_len) |i| {
        if (lengths[i] > 0) {
            indices[w] = @intCast(i);
            w += 1;
        }
    }

    var idx_arr = allocator.alloc(i32, count) catch return;
    defer allocator.free(idx_arr);
    for (0..count) |i| idx_arr[i] = indices[i];

    const sort_key = allocator.alloc(u32, vocab_len) catch return;
    defer allocator.free(sort_key);
    for (0..count) |i| {
        const idx: usize = @intCast(indices[i]);
        const slen = lengths[idx];
        const off = offsets[idx];
        const b0: u32 = u8_data[off];
        const b1: u32 = if (slen > 1) u8_data[off + 1] else 0;
        const b2: u32 = if (slen > 2) u8_data[off + 2] else 0;
        const b3: u32 = if (slen > 3) u8_data[off + 3] else 0;
        sort_key[idx] = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
    }

    // Iterative quicksort
    {
        const StackFrame = struct { l: usize, r: usize };
        var stack: [64]StackFrame = undefined;
        var sp: usize = 0;
        stack[sp] = .{ .l = 0, .r = count - 1 };
        sp += 1;
        while (sp > 0) {
            sp -= 1;
            const l = stack[sp].l;
            const r = stack[sp].r;
            if (l >= r) continue;
            var i: i64 = @as(i64, @intCast(l)) - 1;
            var j: i64 = @intCast(r);
            const pivot = idx_arr[r];
            while (true) {
                while (true) {
                    i += 1;
                    if (i >= @as(i64, @intCast(r))) break;
                    if (compareTokens(idx_arr[@intCast(i)], pivot, sort_key, lengths, offsets, u8_data) > 0) break;
                }
                while (true) {
                    j -= 1;
                    if (j <= @as(i64, @intCast(l))) break;
                    if (compareTokens(idx_arr[@intCast(j)], pivot, sort_key, lengths, offsets, u8_data) < 0) break;
                }
                if (i >= j) break;
                const swap = idx_arr[@intCast(i)];
                idx_arr[@intCast(i)] = idx_arr[@intCast(j)];
                idx_arr[@intCast(j)] = swap;
            }
            const pivot_swap = idx_arr[@intCast(i)];
            idx_arr[@intCast(i)] = idx_arr[r];
            idx_arr[r] = pivot_swap;
            if (i > 0 and l < @as(usize, @intCast(i - 1))) {
                stack[sp] = .{ .l = l, .r = @intCast(i - 1) };
                sp += 1;
            }
            if (@as(usize, @intCast(i + 1)) < r) {
                stack[sp] = .{ .l = @intCast(i + 1), .r = r };
                sp += 1;
            }
        }
    }

    var total_nodes: usize = 1;
    var prev_off: i64 = 0;
    var prev_len: usize = 0;
    for (0..count) |m| {
        const idx: usize = @intCast(idx_arr[m]);
        const off: i64 = offsets[idx];
        const s_len = lengths[idx];
        const min_len = if (prev_len < s_len) prev_len else s_len;
        var lcp: usize = 0;
        while (lcp < min_len and u8_data[@as(usize, @intCast(prev_off + @as(i64, @intCast(lcp))))] == u8_data[@as(usize, @intCast(off + @as(i64, @intCast(lcp))))]) lcp += 1;
        total_nodes += s_len - lcp;
        prev_off = off;
        prev_len = s_len;
    }

    const total_edges = total_nodes - 1;

    const node_id = allocator.alloc(i32, total_nodes) catch return;
    for (0..total_nodes) |i| node_id[i] = -1;
    const child_start = allocator.alloc(i32, total_nodes + 1) catch return;
    for (0..total_nodes + 1) |i| child_start[i] = 0;
    const edge_char = allocator.alloc(u8, total_edges) catch return;
    const edge_target = allocator.alloc(i32, total_edges) catch return;

    var path = allocator.alloc(usize, max_len + 1) catch return;
    defer allocator.free(path);
    path[0] = 0;
    var node_idx: usize = 1;
    prev_off = 0;
    prev_len = 0;
    for (0..count) |m| {
        const idx: usize = @intCast(idx_arr[m]);
        const off: i64 = offsets[idx];
        const s_len = lengths[idx];
        const min_len = if (prev_len < s_len) prev_len else s_len;
        var lcp: usize = 0;
        while (lcp < min_len and u8_data[@as(usize, @intCast(prev_off + @as(i64, @intCast(lcp))))] == u8_data[@as(usize, @intCast(off + @as(i64, @intCast(lcp))))]) lcp += 1;
        var j = lcp;
        while (j < s_len) : (j += 1) {
            const parent = path[j];
            child_start[parent + 1] += 1;
            const new_node = node_idx;
            node_idx += 1;
            path[j + 1] = new_node;
        }
        prev_off = off;
        prev_len = s_len;
    }

    for (1..total_nodes + 1) |n| child_start[n] += child_start[n - 1];

    var write_cursor = allocator.alloc(i32, total_nodes) catch return;
    defer allocator.free(write_cursor);
    for (0..total_nodes) |i| write_cursor[i] = child_start[i];

    path[0] = 0;
    node_idx = 1;
    prev_off = 0;
    prev_len = 0;
    for (0..count) |m| {
        const idx: usize = @intCast(idx_arr[m]);
        const off: i64 = offsets[idx];
        const s_len = lengths[idx];
        const min_len = if (prev_len < s_len) prev_len else s_len;
        var lcp: usize = 0;
        while (lcp < min_len and u8_data[@as(usize, @intCast(prev_off + @as(i64, @intCast(lcp))))] == u8_data[@as(usize, @intCast(off + @as(i64, @intCast(lcp))))]) lcp += 1;
        var j = lcp;
        while (j < s_len) : (j += 1) {
            const parent = path[j];
            const new_node = node_idx;
            node_idx += 1;
            const ww = write_cursor[parent];
            write_cursor[parent] = ww + 1;
            edge_char[@intCast(ww)] = u8_data[@as(usize, @intCast(off + @as(i64, @intCast(j))))];
            edge_target[@intCast(ww)] = @intCast(new_node);
            path[j + 1] = new_node;
        }
        node_id[path[s_len]] = @intCast(idx);
        prev_off = off;
        prev_len = s_len;
    }

    ctx.trie_node_id = node_id;
    ctx.trie_child_start = child_start;
    ctx.trie_edge_char = edge_char;
    ctx.trie_edge_target = edge_target;
}

fn buildTiktokenByteToUnicodeMap(ctx: *c.Context, allocator: std.mem.Allocator) void {
    if (ctx.tiktoken_byte_to_unicode != null) return;

    // Use AutoHashMap for the mapping (sparse keys on the unicode side)
    // For the byte->unicode direction, it's a dense 0-255 mapping
    var map = std.AutoHashMap(i32, i32).init(allocator);
    var n: i32 = 0;
    for (0..256) |b_raw| {
        const b: i32 = @intCast(b_raw);
        if ((b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255)) {
            map.put(b, b) catch {};
        } else {
            map.put(b, 256 + n) catch {};
            n += 1;
        }
    }
    ctx.tiktoken_byte_to_unicode = map;
}

fn buildTiktokenMap(ctx: *c.Context, allocator: std.mem.Allocator) void {
    if (ctx.tiktoken_unicode_to_byte != null) return;
    var map = std.AutoHashMap(i32, i32).init(allocator);
    var n: i32 = 0;
    for (0..256) |b_raw| {
        const b: i32 = @intCast(b_raw);
        if ((b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255)) {
            map.put(b, b) catch {};
        } else {
            map.put(256 + n, b) catch {};
            n += 1;
        }
    }
    ctx.tiktoken_unicode_to_byte = map;
}

fn tokenToBytes(ctx: *c.Context, token: i32) []const u8 {
    if (token < 0 or token >= @as(i32, @intCast((ctx.tokenizer orelse return "").vocab_size))) return "";
    const piece = vocabString(ctx, @intCast(token));
    if (piece.len == 0) return "";
    if (piece.len >= 6 and piece[0] == '<' and piece[1] == '0' and piece[2] == 'x') {
        const hex = piece[3..5];
        const byte = parseHexByte(hex);
        const buf = &ctx.token_to_bytes_buf;
        buf[0] = byte;
        return buf[0..1];
    }
    buildTiktokenMap(ctx, ctx.allocator orelse return "");
    const map = ctx.tiktoken_unicode_to_byte orelse return piece;
    const buf = &ctx.token_to_bytes_buf;
    var len: usize = 0;
    var i: usize = 0;
    while (i < piece.len) {
        var cp: i32 = piece[i];
        if (cp & 0x80 != 0) {
            var extra: usize = 0;
            if (cp & 0xE0 == 0xC0) { cp &= 0x1F; extra = 1; }
            else if (cp & 0xF0 == 0xE0) { cp &= 0x0F; extra = 2; }
            else if (cp & 0xF8 == 0xF0) { cp &= 0x07; extra = 3; }
            else { i += 1; continue; }
            var j: usize = 1;
            while (j <= extra and i + j < piece.len) : (j += 1) {
                cp = (cp << 6) | (piece[i + j] & 0x3F);
            }
            i += 1 + extra;
        } else {
            i += 1;
        }
        const mapped = map.get(cp) orelse cp;
        buf[len] = @as(u8, @intCast(mapped & 0xFF));
        len += 1;
    }
    return buf[0..len];
}

fn decodeToken(ctx: *c.Context, token: i32) []const u8 {
    return tokenToBytes(ctx, token);
}


fn textToTiktoken(text: []const u8, allocator: std.mem.Allocator) ![]u8 {
    var parts = std.array_list.AlignedManaged(u8, null).init(allocator);
    errdefer parts.deinit();

    var n: u32 = 0;
    var byte_to_unicode: [256]u32 = undefined;
    for (0..256) |b| {
        const ub = @as(u8, @intCast(b));
        if ((ub >= 33 and ub <= 126) or (ub >= 161 and ub <= 172) or (ub >= 174 and ub <= 255)) {
            byte_to_unicode[b] = ub;
        } else {
            byte_to_unicode[b] = 256 + n;
            n += 1;
        }
    }

    for (text) |b| {
        const cp = byte_to_unicode[b];
        if (cp < 0x80) {
            try parts.append(@as(u8, @intCast(cp)));
        } else if (cp < 0x800) {
            try parts.append(@as(u8, @intCast(0xC0 | (cp >> 6))));
            try parts.append(@as(u8, @intCast(0x80 | (cp & 0x3F))));
        } else if (cp < 0x10000) {
            try parts.append(@as(u8, @intCast(0xE0 | (cp >> 12))));
            try parts.append(@as(u8, @intCast(0x80 | ((cp >> 6) & 0x3F))));
            try parts.append(@as(u8, @intCast(0x80 | (cp & 0x3F))));
        } else {
            try parts.append(@as(u8, @intCast(0xF0 | (cp >> 18))));
            try parts.append(@as(u8, @intCast(0x80 | ((cp >> 12) & 0x3F))));
            try parts.append(@as(u8, @intCast(0x80 | ((cp >> 6) & 0x3F))));
            try parts.append(@as(u8, @intCast(0x80 | (cp & 0x3F))));
        }
    }
    return parts.toOwnedSlice();
}

fn bpeEncode(ctx: *c.Context, text: []const u8, allocator: std.mem.Allocator) ![]i32 {
    buildSortedVocab(ctx, allocator);
    const t_node_id = ctx.trie_node_id orelse return error.TrieNotBuilt;
    const t_child_start = ctx.trie_child_start orelse return error.TrieNotBuilt;
    const t_edge_char = ctx.trie_edge_char orelse return error.TrieNotBuilt;
    const t_edge_target = ctx.trie_edge_target orelse return error.TrieNotBuilt;

    const tiktok = try textToTiktoken(text, allocator);
    defer allocator.free(tiktok);

    var tokens = std.array_list.AlignedManaged(i32, null).init(allocator);
    errdefer tokens.deinit();

    const byte_len = tiktok.len;
    var pos: usize = 0;
    while (pos < byte_len) {
        var node: usize = 0;
        var best_id: i32 = -1;
        var best_len: usize = 0;
        var j = pos;
        while (j < byte_len) : (j += 1) {
            const ch = tiktok[j];
            const ss = t_child_start[node];
            const e = t_child_start[node + 1];
            var next: i64 = -1;
            var k = ss;
            while (k < e) : (k += 1) {
                if (t_edge_char[@intCast(k)] == ch) {
                    next = t_edge_target[@intCast(k)];
                    break;
                }
            }
            if (next == -1) break;
            node = @intCast(next);
            const nid = t_node_id[node];
            if (nid >= 0) {
                best_id = nid;
                best_len = j - pos + 1;
            }
        }

        if (best_id != -1) {
            try tokens.append(best_id);
            pos += best_len;
        } else {
            pos += 1;
        }
    }

    return tokens.toOwnedSlice();
}


fn encodeChatML(ctx: *c.Context, chat_history: []const c.ChatMessage, sys_prompt: []const u8, allocator: std.mem.Allocator) ![]i32 {
    var tokens = std.array_list.AlignedManaged(i32, null).init(allocator);
    errdefer tokens.deinit();

    var bos_token = findSpecialToken(ctx, "<|begin_of_text|>");
    if (bos_token < 0) bos_token = (ctx.tokenizer orelse return error.NoTokenizer).bos_token;

    var im_start = findSpecialToken(ctx, "<|im_start|>");
    if (im_start < 0) im_start = 0;
    var im_end = findSpecialToken(ctx, "<|im_end|>");
    if (im_end < 0) im_end = 0;

    try tokens.append(bos_token);

    if (sys_prompt.len > 0) {
        try tokens.append(im_start);
        const combined = try std.fmt.allocPrint(allocator, "system\n{s}", .{sys_prompt});
        defer allocator.free(combined);
        const sys_tokens = try bpeEncode(ctx, combined, allocator);
        defer allocator.free(sys_tokens);
        for (sys_tokens) |t| try tokens.append(t);
        try tokens.append(im_end);
        const newline_tokens = try bpeEncode(ctx, "\n", allocator);
        defer allocator.free(newline_tokens);
        for (newline_tokens) |t| try tokens.append(t);
    }

    for (chat_history) |msg| {
        try tokens.append(im_start);
        const combined = try std.fmt.allocPrint(allocator, "{s}\n{s}", .{ msg.role, msg.content });
        defer allocator.free(combined);
        const role_tokens = try bpeEncode(ctx, combined, allocator);
        defer allocator.free(role_tokens);
        for (role_tokens) |t| try tokens.append(t);
        try tokens.append(im_end);
        const newline_tokens = try bpeEncode(ctx, "\n", allocator);
        defer allocator.free(newline_tokens);
        for (newline_tokens) |t| try tokens.append(t);
    }

    try tokens.append(im_start);
    const assistant_tokens = try bpeEncode(ctx, "assistant\n", allocator);
    defer allocator.free(assistant_tokens);
    for (assistant_tokens) |t| try tokens.append(t);

    return tokens.toOwnedSlice();
}

pub fn generate(ctx: *c.Context, chat_history: []const c.ChatMessage, on_token: ?c.FnRender, cancel: ?*const std.atomic.Value(bool), allocator: std.mem.Allocator) ![]u8 {
    const config = ctx.config orelse return error.NoConfig;
    const t = ctx.tokenizer orelse return error.NoTokenizer;

    var prompt_tokens: []i32 = undefined;
    if (debug) log(.{ "[GEN] chat_format: chatml" });
    prompt_tokens = try encodeChatML(ctx, chat_history, ctx.system_prompt, allocator);
    if (debug) {
        ts(); log(.{ "[GEN] prompt_tokens=", prompt_tokens.len });
        log(.{ "[GEN] prompt_tokens=[" });
        for (prompt_tokens[0..@min(prompt_tokens.len, 20)]) |pt| log(.{ pt, "," });
        log(.{ "]" });
    }
    defer allocator.free(prompt_tokens);

    ctx.trie_node_id = null;
    ctx.trie_child_start = null;
    ctx.trie_edge_char = null;
    ctx.trie_edge_target = null;

    if (prompt_tokens.len == 0) {
        prompt_tokens = try allocator.dupe(i32, &.{t.bos_token});
    }

    var token: i32 = prompt_tokens[0];
    var pos: usize = 0;
    var output = std.array_list.AlignedManaged(u8, null).init(allocator);
    errdefer output.deinit();
    const num_prompt_tokens = prompt_tokens.len;
    var pending_newline = false;

    var generated_tokens = std.array_list.AlignedManaged(i32, null).init(allocator);
    defer generated_tokens.deinit();

    var effective_max_tokens: usize = if (ctx.max_tokens > 0) @intCast(ctx.max_tokens) else config.seq_len;
    if (effective_max_tokens > config.seq_len) effective_max_tokens = config.seq_len;

    if (num_prompt_tokens > 1) {
        const prefill_end = num_prompt_tokens - 1;
        while (pos < prefill_end) {
            const remaining = prefill_end - pos;
            const bs = if (c.PREFILL_BATCH_SIZE < remaining) c.PREFILL_BATCH_SIZE else remaining;
            transformerPrefill(ctx, prompt_tokens, pos, bs, allocator);
            pos += bs;
        }
        token = prompt_tokens[pos];
    }

    freeBatchBuffers(ctx);

    var step: usize = pos;
    while (step < effective_max_tokens) : (step += 1) {
        // Cooperative cancellation for session streaming (null = run to end).
        if (cancel) |flag| if (flag.load(.unordered)) break;
        const should_compute_logits = pos >= num_prompt_tokens - 1;
        if (debug) { ts(); log(.{ "[GEN] gen step", step, "/", effective_max_tokens, "pos=", pos, "token=", token }); }
        transformer(ctx, token, pos, should_compute_logits);

        var next: i32 = undefined;
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            const logits = ctx.state orelse return error.NoState;
            const l = logits.logits orelse return error.NoLogits;
            next = sample(ctx, l, ctx.temperature);
            if (debug) { ts(); log(.{ "[GEN] sampled next=", next }); }
            if (debug) { ts(); log(.{ "[DBG] pos", pos, "_token", next }); }
        }

        if (pos >= num_prompt_tokens - 1) {
            if (next == t.eos_token) {
                if (debug) { ts(); log(.{ "[GEN] break: eos" }); }
                break;
            }
            if (next == t.eot_token) {
                if (debug) { ts(); log(.{ "[GEN] break: eot" }); }
                break;
            }

            try generated_tokens.append(next);

            const bytes = tokenToBytes(ctx, next);
            if (bytes.len == 1 and bytes[0] == '\n') {
                pending_newline = true;
            } else {
                if (pending_newline) {
                    try output.appendSlice("\n");
                    if (on_token) |cb| cb("\n");
                    pending_newline = false;
                }
                if (bytes.len > 0) {
                    try output.appendSlice(bytes);
                    if (on_token) |cb| cb(bytes);
                }

                const gt_len = generated_tokens.items.len;
                if (gt_len > 20 and step % 10 == 0) {
                    var stuck = false;
                    const pat_len = [_]usize{ 3, 5, 8, 12 };
                    const pat_min = [_]usize{ 8, 6, 4, 3 };

                    for (pat_len, pat_min) |pl, min_rep| {
                        if (stuck) break;
                        if (gt_len < pl * (min_rep + 1)) continue;
                        var repeats: usize = 0;
                        var r: usize = 1;
                        while (r <= min_rep + 2) : (r += 1) {
                            const off = gt_len - pl - r * pl;
                            if (off >= gt_len or off < off) break;
                            var ok = true;
                            for (0..pl) |p| {
                                if (generated_tokens.items[gt_len - pl + p] != generated_tokens.items[off + p]) {
                                    ok = false;
                                    break;
                                }
                            }
                            if (ok) {
                                repeats += 1;
                            } else {
                                break;
                            }
                        }
                        if (repeats > min_rep) stuck = true;
                    }

                    if (!stuck and gt_len >= 64) {
                        var seen = std.AutoHashMap(i32, void).init(allocator);
                        defer seen.deinit();
                        var unique: usize = 0;
                        var ui = gt_len - 64;
                        while (ui < gt_len) : (ui += 1) {
                            const tok = generated_tokens.items[ui];
                            if (!seen.contains(tok)) {
                                try seen.put(tok, {});
                                unique += 1;
                            }
                        }
                        if (unique < 20) stuck = true;
                    }

                    if (stuck) break;
                }
            }
        }

        token = next;
        pos += 1;
    }

    return output.toOwnedSlice();
}


pub fn findSpecialToken(ctx: *c.Context, token_str: []const u8) i32 {
    const t = ctx.tokenizer orelse return -1;
    const u8_data = ctx.gguf_uint8 orelse return -1;
    const lengths = t.vocab_lengths;
    const sparse_cum = t.vocab_sparse_cum;
    const vocab_size = t.vocab_size;
    const t_len = token_str.len;

    var off: usize = 0;
    for (0..vocab_size) |i| {
        if (i & 255 == 0) off = sparse_cum[i >> 8];
        const li = lengths[i];
        if (li == t_len) {
            var match = true;
            for (0..t_len) |k| {
                if (u8_data[off + k] != token_str[k]) {
                    match = false;
                    break;
                }
            }
            if (match) return @intCast(i);
        }
        off += @as(usize, li) + 8;
    }
    return -1;
}

fn nowMs() u64 {
    var tv: std.c.timeval = undefined;
    if (std.c.gettimeofday(&tv, null) != 0) return 0;
    return @as(u64, @intCast(tv.sec)) * 1000 + @as(u64, @intCast(tv.usec)) / 1000;
}

fn ts() void {
    const ms = nowMs();
    const s = ms / 1000;
    std.debug.print("[{d:0>2}:{d:0>2}:{d:0>2}.{d:0>3}] ", .{ (s/3600)%24, (s/60)%60, s%60, ms%1000 });
}

fn getRSS() u64 {
    var usage: std.c.rusage = undefined;
    if (std.c.getrusage(std.c.rusage.SELF, &usage) != 0) return 0;
    return @as(u64, @intCast(usage.maxrss));
}

fn rssToMB(rss: u64) f64 {
    // macOS: bytes, Linux: KB
    if (comptime @import("builtin").target.os.tag == .linux) {
        return @as(f64, @floatFromInt(rss)) / 1024.0;
    }
    return @as(f64, @floatFromInt(rss)) / (1024.0 * 1024.0);
}

fn perfPrint(label: []const u8, time_ms: u64, rss: u64, baseline: u64) void {
    const rss_mb = rssToMB(rss);
    const delta_mb = rssToMB(rss -| baseline);
    if (time_ms > 0) {
        std.debug.print("[PERF]  {s:<14} {d:>8}  {d:>8.1}   +{d:>.1}\n", .{ label, time_ms, rss_mb, delta_mb });
    } else {
        std.debug.print("[PERF]  {s:<14} {s:>8}  {d:>8.1}\n", .{ label, "--", rss_mb });
    }
}


pub fn conversation(ctx: *c.Context, data: c.LlamaInput, model_data: []const u8, on_progress: ?c.FnProgress, cancel: ?*const std.atomic.Value(bool), allocator: std.mem.Allocator) ![]u8 {
    if (std.mem.eql(u8, data.type, "load")) {
        ctx.perf_baseline_rss = getRSS();
        const t0_load = nowMs();

        if (data.max_tokens > 0) ctx.max_tokens = data.max_tokens;
        if (data.context_size > 0) ctx.context_size = data.context_size;
        if (data.system_prompt.len > 0) {
            ctx.system_prompt = try allocator.dupe(u8, data.system_prompt);
        }
        if (data.temperature != 0.0) ctx.temperature = data.temperature;
        if (data.top_p != 0.0) ctx.top_p = data.top_p;
        if (data.top_k > 0) ctx.top_k = data.top_k;
        if (data.cb_render) |cb| ctx.cb_render = cb;
        ctx.allocator = allocator;

        try op.loadModel(ctx, model_data, findSpecialToken, on_progress, allocator);

        ctx.perf_load_rss = getRSS();
        ctx.perf_load_time_ns = (nowMs() - t0_load) * std.time.ns_per_ms;
        if (debug) {
            log(.{ "[PERF] Model loaded" });
        }
        return "";
    } else if (std.mem.eql(u8, data.type, "generate")) {
        if (ctx.gguf_uint8 != null) {
            const t0_gen = nowMs();
            const cb = data.cb_render;
            const result = try generate(ctx, data.chat_history orelse &.{}, cb, cancel, allocator);

            ctx.perf_gen_rss = getRSS();
            ctx.perf_gen_time_ns = (nowMs() - t0_gen) * std.time.ns_per_ms;
            if (debug) {
                const load_ms = ctx.perf_load_time_ns / std.time.ns_per_ms;
                const gen_ms = ctx.perf_gen_time_ns / std.time.ns_per_ms;
                const total_ms = load_ms + gen_ms;
                const peak_rss = @max(ctx.perf_baseline_rss, @max(ctx.perf_load_rss, ctx.perf_gen_rss));

                std.debug.print("[PERF] ── Summary ──────────────────────────────────\n", .{});
                std.debug.print("[PERF]   Phase          Time (ms)    RSS (MB)   ΔRSS\n", .{});
                std.debug.print("[PERF]   ───────────────────────────────────────────\n", .{});
                perfPrint("Baseline", 0, ctx.perf_baseline_rss, ctx.perf_baseline_rss);
                perfPrint("Load", load_ms, ctx.perf_load_rss, ctx.perf_baseline_rss);
                perfPrint("Generate", gen_ms, ctx.perf_gen_rss, ctx.perf_baseline_rss);
                std.debug.print("[PERF]   ───────────────────────────────────────────\n", .{});
                perfPrint("Peak", 0, peak_rss, ctx.perf_baseline_rss);
                std.debug.print("[PERF]   Total          {d:>8}\n", .{total_ms});
                std.debug.print("[PERF] ────────────────────────────────────────────────\n", .{});
            }
            return result;
        }
        return "";
    }
    return "";
}
