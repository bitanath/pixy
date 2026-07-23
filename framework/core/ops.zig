const std = @import("std");
const c = @import("constants.zig");
const dm = @import("dequant.zig");
const mt = @import("mats.zig");

const ReadState = struct {
    data: []const u8,
    pos: usize = 0,
};

fn readU8(rs: *ReadState) u8 {
    const val = rs.data[rs.pos];
    rs.pos += 1;
    return val;
}

fn readU16(rs: *ReadState) u16 {
    const val = std.mem.readInt(u16, rs.data[rs.pos..][0..2], .little);
    rs.pos += 2;
    return val;
}

fn readU32(rs: *ReadState) u32 {
    const val = std.mem.readInt(u32, rs.data[rs.pos..][0..4], .little);
    rs.pos += 4;
    return val;
}

fn readU64(rs: *ReadState) u64 {
    const val = std.mem.readInt(u64, rs.data[rs.pos..][0..8], .little);
    rs.pos += 8;
    return val;
}

fn readI8(rs: *ReadState) i8 {
    const val: i8 = @bitCast(rs.data[rs.pos]);
    rs.pos += 1;
    return val;
}

fn readI16(rs: *ReadState) i16 {
    const val = std.mem.readInt(i16, rs.data[rs.pos..][0..2], .little);
    rs.pos += 2;
    return val;
}

fn readI32(rs: *ReadState) i32 {
    const val = std.mem.readInt(i32, rs.data[rs.pos..][0..4], .little);
    rs.pos += 4;
    return val;
}

fn readI64(rs: *ReadState) i64 {
    const val = std.mem.readInt(i64, rs.data[rs.pos..][0..8], .little);
    rs.pos += 8;
    return val;
}

fn readF32(rs: *ReadState) f32 {
    const bits = std.mem.readInt(u32, rs.data[rs.pos..][0..4], .little);
    rs.pos += 4;
    return @bitCast(bits);
}

fn readF64(rs: *ReadState) f64 {
    const bits = std.mem.readInt(u64, rs.data[rs.pos..][0..8], .little);
    rs.pos += 8;
    return @bitCast(bits);
}

fn readString(rs: *ReadState, allocator: std.mem.Allocator) ![]const u8 {
    const len: usize = @intCast(readU64(rs));
    if (len == 0 or len > 1_000_000) return error.InvalidStringLength;
    const str = try allocator.alloc(u8, len);
    @memcpy(str, rs.data[rs.pos..][0..len]);
    rs.pos += len;
    return str;
}

pub fn fastTanh(x: f64) f64 {
    if (x < -4.0) return -1.0;
    if (x > 4.0) return 1.0;
    const x2 = x * x;
    return (x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)))) /
        (135135.0 + x2 * (62370.0 + x2 * (3150.0 + 28.0 * x2)));
}

pub fn fastExpf(x: f64) f64 {
    return @exp(x);
}

pub fn rmsnorm(out: []f32, x: []const f32, w: []const f32, size: usize, inv_size: f64, eps: f64) void {
    const inv = if (inv_size == 0) 1.0 / @as(f64, @floatFromInt(size)) else inv_size;
    var ss: f64 = 0.0;
    const size8 = size & ~@as(usize, 7);
    var i: usize = 0;
    while (i < size8) : (i += 8) {
        const xv: @Vector(8, f32) = x[i..][0..8].*;
        const fv: @Vector(8, f32) = xv * xv;
        ss += @reduce(.Add, fv);
    }
    while (i < size) : (i += 1) {
        const v: f64 = x[i];
        ss += v * v;
    }
    ss = 1.0 / @sqrt(ss * inv + eps);
    const ssv: @Vector(8, f32) = @splat(@as(f32, @floatCast(ss)));
    i = 0;
    while (i < size8) : (i += 8) {
        out[i..][0..8].* = w[i..][0..8].* * ssv * x[i..][0..8].*;
    }
    while (i < size) : (i += 1) {
        out[i] = @floatCast(w[i] * ss * x[i]);
    }
}

pub fn accum(a: []f32, b: []const f32, size: usize) void {
    const size8 = size & ~@as(usize, 7);
    var i: usize = 0;
    while (i < size8) : (i += 8) {
        var a_vec: @Vector(8, f32) = a[i..][0..8].*;
        const b_vec: @Vector(8, f32) = b[i..][0..8].*;
        a_vec += b_vec;
        a[i..][0..8].* = a_vec;
    }
    while (i < size) : (i += 1) {
        a[i] += b[i];
    }
}

fn skipGGUFValue(rs: *ReadState, gg_type: c.GGUF_TYPE) void {
    switch (gg_type) {
        .UINT8, .INT8, .BOOL => rs.pos += 1,
        .UINT16, .INT16 => rs.pos += 2,
        .UINT32, .INT32, .FLOAT32 => rs.pos += 4,
        .UINT64, .INT64, .FLOAT64 => rs.pos += 8,
        .STRING => {
            const slen: usize = @intCast(readU64(rs));
            rs.pos += slen;
        },
        .ARRAY => {
            const arr_type: c.GGUF_TYPE = @enumFromInt(readU32(rs));
            const arr_len: usize = @intCast(readU64(rs));
            switch (arr_type) {
                .UINT8, .INT8, .BOOL => rs.pos += arr_len,
                .UINT16, .INT16 => rs.pos += arr_len * 2,
                .UINT32, .INT32, .FLOAT32 => rs.pos += arr_len * 4,
                .UINT64, .INT64, .FLOAT64 => rs.pos += arr_len * 8,
                .STRING => {
                    for (0..arr_len) |_| {
                        const slen: usize = @intCast(readU64(rs));
                        rs.pos += slen;
                    }
                },
                else => {},
            }
        },
        else => {},
    }
}

fn readNumericGGUFValue(rs: *ReadState, gg_type: c.GGUF_TYPE) f64 {
    return switch (gg_type) {
        .UINT8 => @floatFromInt(readU8(rs)),
        .INT8 => @floatFromInt(readI8(rs)),
        .UINT16 => @floatFromInt(readU16(rs)),
        .INT16 => @floatFromInt(readI16(rs)),
        .UINT32 => @floatFromInt(readU32(rs)),
        .INT32 => @floatFromInt(readI32(rs)),
        .FLOAT32 => readF32(rs),
        .BOOL => if (readU8(rs) != 0) 1.0 else 0.0,
        .UINT64 => @floatFromInt(readU64(rs)),
        .INT64 => @floatFromInt(readI64(rs)),
        .FLOAT64 => readF64(rs),
        else => 0.0,
    };
}

pub fn parseGGUF(allocator: std.mem.Allocator, data: []const u8) !c.GGUFMetadata {
    var rs = ReadState{ .data = data };
    var gguf = c.GGUFMetadata.init(allocator);

    const magic = readU32(&rs);
    if (magic != c.GGUF_MAGIC) return error.InvalidGGUFMagic;

    _ = readU32(&rs);
    const n_tensors: usize = @intCast(readU64(&rs));
    const n_kv: usize = @intCast(readU64(&rs));

    var p: usize = 0;
    var si: usize = 0;
    while (si < n_kv) : (si += 1) {
        const key = try readString(&rs, allocator);
        const value_type: c.GGUF_TYPE = @enumFromInt(readU32(&rs));

        if (c.shouldSkipKey(key)) {
            skipGGUFValue(&rs, value_type);
            continue;
        }

        if (value_type == .STRING) {
            const val = try readString(&rs, allocator);
            try gguf.string_meta.put(key, val);
        } else if (value_type == .ARRAY) {
            const arr_type: c.GGUF_TYPE = @enumFromInt(readU32(&rs));
            const arr_len: usize = @intCast(readU64(&rs));
            if (arr_type == .STRING) {
                const sparse_step: usize = 256;
                const lengths = try allocator.alloc(u16, arr_len);
                const sparse_cum = try allocator.alloc(u32, ((arr_len - 1) >> 8) + 1);
                p = rs.pos;
                for (0..arr_len) |i| {
                    const slen: u32 = std.mem.readInt(u32, data[p..][0..4], .little);
                    p += 8;
                    if (i % sparse_step == 0) sparse_cum[i >> 8] = @intCast(p);
                    lengths[i] = @intCast(slen);
                    p += slen;
                }
                rs.pos = p;
                gguf.vocab_lengths = lengths;
                gguf.vocab_sparse_cum = sparse_cum;
                gguf.vocab_sparse_step = @intCast(sparse_step);
            } else if (arr_type == .FLOAT32 or arr_type == .UINT32) {
                rs.pos += arr_len * 4;
            } else {
                rs.pos += arr_len;
            }
        } else {
            const val = readNumericGGUFValue(&rs, value_type);
            try gguf.number_meta.put(key, val);
        }
    }

    for (0..n_tensors) |_| {
        const name = try readString(&rs, allocator);
        const n_dims = readU32(&rs);
        const dims = try allocator.alloc(u32, n_dims);
        for (0..n_dims) |d| {
            dims[d] = @intCast(readU64(&rs));
        }
        const quant_type: c.GGML_TYPE = @enumFromInt(readU32(&rs));
        const offset: usize = @intCast(readU64(&rs));

        var n_elements: usize = 1;
        for (dims) |d| n_elements *= d;

        try gguf.tensors.put(name, .{
            .dims = dims,
            .quant_type = quant_type,
            .offset = offset,
            .n_elements = n_elements,
        });
    }

    const alignment_val: f64 = gguf.number_meta.get("general.alignment") orelse 32.0;
    const alignment: usize = @intFromFloat(alignment_val);
    gguf.tensor_data_offset = ((rs.pos + alignment - 1) / alignment) * alignment;

    return gguf;
}

fn dequantizeTensor(ctx: *c.Context, offset: usize, n_elements: usize, ggml_type: c.GGML_TYPE, allocator: std.mem.Allocator) ![]f32 {
    const dst = try allocator.alloc(f32, n_elements);
    dm.dequantizeRow(ctx, dst, offset, n_elements, ggml_type);
    return dst;
}

fn loadFloatHelper(ctx: *c.Context, prefix: []const u8, suffix: []const u8, tensors: *std.StringHashMap(c.GGUFTensor), base_offset: usize, allocator: std.mem.Allocator) ?[]const f32 {
    const name = std.mem.concat(allocator, u8, &.{prefix, suffix}) catch return null;
    defer allocator.free(name);
    const t = tensors.get(name) orelse return null;
    const off = base_offset + t.offset;
    if (t.quant_type == .F32) {
        const data = ctx.gguf_uint8.?;
        const result = allocator.alloc(f32, t.n_elements) catch return null;
        for (0..t.n_elements) |i| {
            result[i] = @as(f32, @bitCast(std.mem.readInt(u32, data[off + i * 4 ..][0..4], .little)));
        }
        return result;
    }
    return dequantizeTensor(ctx, off, t.n_elements, t.quant_type, allocator) catch null;
}

fn needsLocalBuffers(quant_type: c.GGML_TYPE) bool {
    return switch (quant_type) {
        .Byte, .Nibble, .Word => true,
        else => false,
    };
}

fn loadQuantHelper(ctx: *c.Context, prefix: []const u8, suffix: []const u8, rows: usize, cols: usize, tensors: *std.StringHashMap(c.GGUFTensor), base_offset: usize) ?c.QuantizedTensor {
    var buf: [128]u8 = undefined;
    const name = std.fmt.bufPrint(&buf, "{s}{s}", .{prefix, suffix}) catch return null;
    const t = tensors.get(name) orelse return null;
    const row_sz = mt.getRowSize(cols, t.quant_type);
    const off = base_offset + t.offset;

    var qt = c.QuantizedTensor{
        .data_offset = off,
        .quant_type = t.quant_type,
        .rows = rows,
        .cols = cols,
        .row_size = row_sz,
        .dot_func = mt.getVecDotFunc(t.quant_type),
        .dot_q8_func = mt.getVecDotQ8Func(t.quant_type),
        .deq_row_func = dm.getDeqRowFunc(t.quant_type),
    };

    if (needsLocalBuffers(t.quant_type)) {
        const total_bytes = rows * row_sz;
        qt.local_u8 = ctx.gguf_uint8.?[off..][0..total_bytes];
        qt.local_i8 = @ptrCast(ctx.gguf_uint8.?[off..][0..total_bytes]);
        qt.data_ptr = @intCast(off);
    }

    return qt;
}

fn loadWeights(ctx: *c.Context, gguf: *c.GGUFMetadata, allocator: std.mem.Allocator) !c.Weights {
    const tensors = &gguf.tensors;
    const base_offset = gguf.tensor_data_offset;
    const config = ctx.config.?;

    const head_size = config.head_dim;
    const kv_dim = config.n_kv_heads * head_size;
    const q_dim = config.n_heads * head_size;

    var token_embedding: c.QuantizedTensor = .{};
    if (tensors.get("token_embd.weight")) |emb_tensor| {
        const row_sz = mt.getRowSize(config.dim, emb_tensor.quant_type);
        const off = base_offset + emb_tensor.offset;
        token_embedding = .{
            .data_offset = off,
            .quant_type = emb_tensor.quant_type,
            .rows = config.vocab_size,
            .cols = config.dim,
            .row_size = row_sz,
            .dot_func = mt.getVecDotFunc(emb_tensor.quant_type),
            .dot_q8_func = mt.getVecDotQ8Func(emb_tensor.quant_type),
            .deq_row_func = dm.getDeqRowFunc(emb_tensor.quant_type),
        };
        if (needsLocalBuffers(emb_tensor.quant_type)) {
            const total_bytes = config.vocab_size * row_sz;
            token_embedding.local_u8 = ctx.gguf_uint8.?[off..][0..total_bytes];
            token_embedding.local_i8 = @ptrCast(ctx.gguf_uint8.?[off..][0..total_bytes]);
            token_embedding.data_ptr = @intCast(off);
        }
    }

    const layers = try allocator.alloc(c.LayerWeights, config.n_layers);
    for (0..config.n_layers) |l| {
        var layer = c.LayerWeights{};
        const prefix = try std.fmt.allocPrint(allocator, "blk.{d}.", .{l});
        defer allocator.free(prefix);

        layer.rms_att_weight = loadFloatHelper(ctx, prefix, "attn_norm.weight", tensors, base_offset, allocator);
        layer.rms_ffn_weight = loadFloatHelper(ctx, prefix, "ffn_norm.weight", tensors, base_offset, allocator);
        if (layer.rms_ffn_weight == null) {
            layer.rms_ffn_weight = loadFloatHelper(ctx, prefix, "ffn_norm", tensors, base_offset, allocator);
        }
        layer.wq = loadQuantHelper(ctx, prefix, "attn_q.weight", q_dim, config.dim, tensors, base_offset);
        layer.wk = loadQuantHelper(ctx, prefix, "attn_k.weight", kv_dim, config.dim, tensors, base_offset);
        layer.wv = loadQuantHelper(ctx, prefix, "attn_v.weight", kv_dim, config.dim, tensors, base_offset);
        layer.wo = loadQuantHelper(ctx, prefix, "attn_output.weight", config.dim, q_dim, tensors, base_offset);
        layer.w1 = loadQuantHelper(ctx, prefix, "ffn_gate.weight", config.hidden_dim, config.dim, tensors, base_offset);
        layer.w2 = loadQuantHelper(ctx, prefix, "ffn_down.weight", config.dim, config.hidden_dim, tensors, base_offset);
        layer.w3 = loadQuantHelper(ctx, prefix, "ffn_up.weight", config.hidden_dim, config.dim, tensors, base_offset);

        {
            const d_inner = config.ssm_d_inner;
            const d_state = config.ssm_d_state;
            const n_group = config.ssm_n_group;
            const ssm_proj_size = 2 * d_inner + 2 * n_group * d_state + config.ssm_dt_rank;

            layer.ssm_in = loadQuantHelper(ctx, prefix, "ssm_in.weight", ssm_proj_size, config.dim, tensors, base_offset);
            layer.ssm_conv1d_w = loadFloatHelper(ctx, prefix, "ssm_conv1d.weight", tensors, base_offset, allocator);
            layer.ssm_conv1d_b = loadFloatHelper(ctx, prefix, "ssm_conv1d.bias", tensors, base_offset, allocator);
            layer.ssm_dt_b = loadFloatHelper(ctx, prefix, "ssm_dt.bias", tensors, base_offset, allocator);
            layer.ssm_a = loadFloatHelper(ctx, prefix, "ssm_a", tensors, base_offset, allocator);
            layer.ssm_d = loadFloatHelper(ctx, prefix, "ssm_d", tensors, base_offset, allocator);
            layer.ssm_out = loadQuantHelper(ctx, prefix, "ssm_out.weight", config.dim, d_inner, tensors, base_offset);
            layer.ssm_norm = loadFloatHelper(ctx, prefix, "ssm_norm.weight", tensors, base_offset, allocator);
            if (layer.ssm_norm == null) {
                const norm = try allocator.alloc(f32, d_inner);
                @memset(norm, 0.0);
                layer.ssm_norm = norm;
            }
        }

        layers[l] = layer;
    }

    var rms_final_weight: ?[]const f32 = null;
    if (tensors.get("output_norm.weight")) |t| {
        const off = base_offset + t.offset;
        if (t.quant_type == .F32) {
            const data = ctx.gguf_uint8.?;
            const result = allocator.alloc(f32, t.n_elements) catch null;
            if (result) |r| {
                for (0..t.n_elements) |i| {
                    r[i] = @as(f32, @bitCast(std.mem.readInt(u32, data[off + i * 4 ..][0..4], .little)));
                }
                rms_final_weight = r;
            }
        } else {
            rms_final_weight = try dequantizeTensor(ctx, off, t.n_elements, t.quant_type, allocator);
        }
    }

    var wcls: c.QuantizedTensor = .{};
    if (tensors.get("output.weight")) |t| {
        const row_sz = mt.getRowSize(config.dim, t.quant_type);
        const off = base_offset + t.offset;
        wcls = .{
            .data_offset = off,
            .quant_type = t.quant_type,
            .rows = config.vocab_size,
            .cols = config.dim,
            .row_size = row_sz,
            .dot_func = mt.getVecDotFunc(t.quant_type),
            .dot_q8_func = mt.getVecDotQ8Func(t.quant_type),
            .deq_row_func = dm.getDeqRowFunc(t.quant_type),
        };
        if (needsLocalBuffers(t.quant_type)) {
            const total_bytes = config.vocab_size * row_sz;
            wcls.local_u8 = ctx.gguf_uint8.?[off..][0..total_bytes];
            wcls.local_i8 = @ptrCast(ctx.gguf_uint8.?[off..][0..total_bytes]);
            wcls.data_ptr = @intCast(off);
        }
    } else {
        wcls = token_embedding;
    }

    return .{
        .token_embedding = token_embedding,
        .layers = layers,
        .rms_final_weight = rms_final_weight,
        .wcls = wcls,
    };
}
pub const FindSpecialTokenFn = *const fn (ctx: *c.Context, token_str: []const u8) i32;

fn createRunState(config: *c.Config, ctx_top_k: i32, allocator: std.mem.Allocator) !c.RunState {
    const head_size: usize = config.head_dim;
    const kv_dim: usize = config.n_kv_heads * head_size;
    const q_dim: usize = config.n_heads * head_size;
    const max_dim = if (config.dim > q_dim) config.dim else q_dim;

    const rope_size: usize = head_size / 2;
    const rope_freqs = try allocator.alloc(f32, rope_size);
    for (0..rope_size) |i| {
        const exp: f64 = (@as(f64, @floatFromInt(i)) * 2.0) / @as(f64, @floatFromInt(head_size));
        rope_freqs[i] = @floatCast(1.0 / std.math.pow(f64, config.rope_theta, exp));
    }

    const rope_scratch_size = c.PREFILL_BATCH_SIZE * rope_size;
    const rope_cos_all = try allocator.alloc(f32, rope_scratch_size);
    const rope_sin_all = try allocator.alloc(f32, rope_scratch_size);

    const head_bytes_q8 = (head_size >> 5) * c.Q8_0_BLOCK_SIZE;
    const max_cols = if (max_dim > config.hidden_dim) max_dim else config.hidden_dim;
    const x_q8_size = (max_cols >> 5) * 34;

    const kv_mul = config.n_heads / config.n_kv_heads;
    const head_q_offsets = try allocator.alloc(i32, config.n_heads);
    const head_kv_idx = try allocator.alloc(i32, config.n_heads);
    const head_att_offsets = try allocator.alloc(i32, config.n_heads);
    const head_kv_byte_offsets = try allocator.alloc(i32, config.n_heads);
    for (0..config.n_heads) |h| {
        head_q_offsets[h] = @intCast(h * head_size);
        head_kv_idx[h] = @intCast(h / kv_mul);
        head_att_offsets[h] = @intCast(h * config.seq_len);
        head_kv_byte_offsets[h] = 0;
    }

    const rope_cos_layer = try allocator.alloc(?[]f32, config.n_layers);
    const rope_sin_layer = try allocator.alloc(?[]f32, config.n_layers);
    for (0..config.n_layers) |l| {
        rope_cos_layer[l] = rope_cos_all;
        rope_sin_layer[l] = rope_sin_all;
    }

    const kv_cache_layer_size: usize = 0;
    const kv_capacity: usize = 0;

    const key_cache = try allocator.alloc(u8, kv_cache_layer_size * config.n_layers);
    const value_cache = try allocator.alloc(u8, kv_cache_layer_size * config.n_layers);
    const key_cache_int8: []i8 = @ptrCast(key_cache);
    const value_cache_int8: []i8 = @ptrCast(value_cache);

    const q_q8_total_bytes = config.n_heads * head_bytes_q8;
    const q_q8 = try allocator.alloc(u8, q_q8_total_bytes);
    const q_q8i8: []i8 = @ptrCast(q_q8);

    var ssm_conv_state: []?[]f32 = &.{};
    var ssm_state_arr: []?[]f32 = &.{};
    var ssm_proj_buf: ?[]f32 = null;
    var ssm_xbc_conv: ?[]f32 = null;
    var ssm_y_buf: ?[]f32 = null;

    {
        const d_conv = config.ssm_d_conv;
        const d_inner = config.ssm_d_inner;
        const d_state = config.ssm_d_state;
        const n_group = config.ssm_n_group;
        const conv_dim = d_inner + 2 * n_group * d_state;
        const ssm_proj_size = 2 * d_inner + 2 * n_group * d_state + config.ssm_dt_rank;

        ssm_conv_state = try allocator.alloc(?[]f32, config.n_layers);
        ssm_state_arr = try allocator.alloc(?[]f32, config.n_layers);
        for (0..config.n_layers) |l| {
            ssm_conv_state[l] = try allocator.alloc(f32, conv_dim * (d_conv - 1));
            ssm_state_arr[l] = try allocator.alloc(f32, d_state * d_inner);
        }
        ssm_proj_buf = try allocator.alloc(f32, ssm_proj_size);
        ssm_xbc_conv = try allocator.alloc(f32, conv_dim);
        ssm_y_buf = try allocator.alloc(f32, d_inner);
    }

    const top_k: usize = @intCast(ctx_top_k);
    const top_k_indices = try allocator.alloc(i32, top_k);
    const top_k_values = try allocator.alloc(f32, top_k);

    return .{
        .x = try allocator.alloc(f32, config.dim),
        .xb = try allocator.alloc(f32, max_dim),
        .xb2 = try allocator.alloc(f32, config.dim),
        .hb = try allocator.alloc(f32, config.hidden_dim),
        .hb2 = try allocator.alloc(f32, config.hidden_dim),
        .q = try allocator.alloc(f32, q_dim),
        .k = try allocator.alloc(f32, kv_dim),
        .v = try allocator.alloc(f32, kv_dim),
        .att = try allocator.alloc(f32, config.n_heads * config.seq_len),
        .logits = null,
        .batch_dim = config.dim,
        .batch_max_dim = max_dim,
        .batch_q_dim = q_dim,
        .batch_kv_dim = kv_dim,
        .batch_hidden_dim = config.hidden_dim,
        .batch_x_q8_size = x_q8_size,
        .batch_matmul_deq_cols = max_cols,
        .batch_buffers_ready = false,
        .key_cache = key_cache,
        .value_cache = value_cache,
        .key_cache_int8 = key_cache_int8,
        .value_cache_int8 = value_cache_int8,
        .q_q8 = q_q8,
        .q_q8i8 = q_q8i8,
        .head_seq_bytes = 0,
        .rope_cos_all = rope_cos_all,
        .rope_sin_all = rope_sin_all,
        .rope_freqs = rope_freqs,
        .rope_size = rope_size,
        .rope_cos_layer = rope_cos_layer,
        .rope_sin_layer = rope_sin_layer,
        .kv_mul = kv_mul,
        .kv_cache_layer_size = kv_cache_layer_size,
        .kv_capacity = kv_capacity,
        .attn_scale = 1.0 / @sqrt(@as(f64, @floatFromInt(head_size))),
        .dim = config.dim,
        .n_heads = config.n_heads,
        .n_kv_heads = config.n_kv_heads,
        .n_layers = config.n_layers,
        .seq_len = config.seq_len,
        .hidden_dim = config.hidden_dim,
        .vocab_size = config.vocab_size,
        .rms_norm_eps = config.rms_norm_eps,
        .inv_dim = 1.0 / @as(f64, @floatFromInt(config.dim)),
        .inv_head_size = 1.0 / @as(f64, @floatFromInt(head_size)),
        .top_k_indices = top_k_indices,
        .top_k_values = top_k_values,
        .head_q_offsets = head_q_offsets,
        .head_kv_idx = head_kv_idx,
        .head_att_offsets = head_att_offsets,
        .head_kv_byte_offsets = head_kv_byte_offsets,
        .head_bytes_q8 = head_bytes_q8,
        .batch_x = .{null} ** c.PREFILL_BATCH_SIZE,
        .batch_xb = .{null} ** c.PREFILL_BATCH_SIZE,
        .batch_xb2 = .{null} ** c.PREFILL_BATCH_SIZE,
        .batch_q_arr = .{null} ** c.PREFILL_BATCH_SIZE,
        .batch_k_arr = .{null} ** c.PREFILL_BATCH_SIZE,
        .batch_v_arr = .{null} ** c.PREFILL_BATCH_SIZE,
        .batch_hb = .{null} ** c.PREFILL_BATCH_SIZE,
        .batch_hb2 = .{null} ** c.PREFILL_BATCH_SIZE,
        .batch_q8 = .{null} ** c.PREFILL_BATCH_SIZE,
        .batch_q8i8 = .{null} ** c.PREFILL_BATCH_SIZE,
        .conv_state = ssm_conv_state,
        .ssm_state = ssm_state_arr,
        .ssm_proj_buf = ssm_proj_buf,
        .ssm_xbc_conv = ssm_xbc_conv,
        .ssm_y_buf = ssm_y_buf,
    };
}

pub fn loadModel(
    ctx: *c.Context,
    model_data: []const u8,
    find_special_token: FindSpecialTokenFn,
    on_progress: ?c.FnProgress,
    allocator: std.mem.Allocator,
) !void {
    ctx.gguf_uint8 = @constCast(model_data);
    ctx.gguf_int8 = @ptrCast(@constCast(model_data));
    ctx.gguf_base_offset = -1;

    var gguf = try parseGGUF(allocator, model_data);
    defer gguf.deinit();

    _ = gguf.string_meta.get("general.architecture");
    const meta_prefix = "falcon-h1";

    if (on_progress) |cb| cb("Loaded model now progressing...");

    const config = try allocator.create(c.Config);
    config.* = .{};

    config.seq_len = blk: {
        const key = try std.fmt.allocPrint(allocator, "{s}.context_length", .{meta_prefix});
        defer allocator.free(key);
        break :blk @intFromFloat(gguf.number_meta.get(key) orelse 2048.0);
    };

    config.dim = blk: {
        const key = try std.fmt.allocPrint(allocator, "{s}.embedding_length", .{meta_prefix});
        defer allocator.free(key);
        break :blk @intFromFloat(gguf.number_meta.get(key) orelse 4096.0);
    };

    config.hidden_dim = blk: {
        const key = try std.fmt.allocPrint(allocator, "{s}.feed_forward_length", .{meta_prefix});
        defer allocator.free(key);
        break :blk @intFromFloat(gguf.number_meta.get(key) orelse 11008.0);
    };

    config.n_layers = blk: {
        const key = try std.fmt.allocPrint(allocator, "{s}.block_count", .{meta_prefix});
        defer allocator.free(key);
        break :blk @intFromFloat(gguf.number_meta.get(key) orelse 32.0);
    };

    config.n_heads = blk: {
        const key = try std.fmt.allocPrint(allocator, "{s}.attention.head_count", .{meta_prefix});
        defer allocator.free(key);
        break :blk @intFromFloat(gguf.number_meta.get(key) orelse 32.0);
    };

    config.n_kv_heads = blk: {
        const key = try std.fmt.allocPrint(allocator, "{s}.attention.head_count_kv", .{meta_prefix});
        defer allocator.free(key);
        break :blk @intFromFloat(gguf.number_meta.get(key) orelse @as(f64, @floatFromInt(config.n_heads)));
    };

    config.vocab_size = blk: {
        const key = try std.fmt.allocPrint(allocator, "{s}.vocab_size", .{meta_prefix});
        defer allocator.free(key);
        break :blk @intFromFloat(gguf.number_meta.get(key) orelse 32000.0);
    };

    if (ctx.context_size > 0) {
        const cs = ctx.context_size;
        config.seq_len = if (config.seq_len < cs) config.seq_len else cs;
    }

    config.rope_theta = blk: {
        const key = try std.fmt.allocPrint(allocator, "{s}.rope.freq_base", .{meta_prefix});
        defer allocator.free(key);
        break :blk gguf.number_meta.get(key) orelse 500000.0;
    };

    config.head_dim = blk: {
        const key = try std.fmt.allocPrint(allocator, "{s}.attention.key_length", .{meta_prefix});
        defer allocator.free(key);
        break :blk @intFromFloat(gguf.number_meta.get(key) orelse 0.0);
    };

    config.rms_norm_eps = blk: {
        const key = try std.fmt.allocPrint(allocator, "{s}.attention.layer_norm_rms_epsilon", .{meta_prefix});
        defer allocator.free(key);
        break :blk gguf.number_meta.get(key) orelse 1e-6;
    };

    config.ssm_d_conv = blk: {
        const key = try std.fmt.allocPrint(allocator, "{s}.ssm.conv_kernel", .{meta_prefix});
        defer allocator.free(key);
        break :blk @intFromFloat(gguf.number_meta.get(key) orelse 4.0);
    };
    config.ssm_d_inner = blk: {
        const key = try std.fmt.allocPrint(allocator, "{s}.ssm.inner_size", .{meta_prefix});
        defer allocator.free(key);
        break :blk @intFromFloat(gguf.number_meta.get(key) orelse 768.0);
    };
    config.ssm_d_state = blk: {
        const key = try std.fmt.allocPrint(allocator, "{s}.ssm.state_size", .{meta_prefix});
        defer allocator.free(key);
        break :blk @intFromFloat(gguf.number_meta.get(key) orelse 64.0);
    };
    config.ssm_dt_rank = blk: {
        const key = try std.fmt.allocPrint(allocator, "{s}.ssm.time_step_rank", .{meta_prefix});
        defer allocator.free(key);
        break :blk @intFromFloat(gguf.number_meta.get(key) orelse 24.0);
    };
    config.ssm_n_group = blk: {
        const key = try std.fmt.allocPrint(allocator, "{s}.ssm.group_count", .{meta_prefix});
        defer allocator.free(key);
        break :blk @intFromFloat(gguf.number_meta.get(key) orelse 1.0);
    };

    if (config.head_dim == 0) config.head_dim = config.dim / config.n_heads;

    const tokenizer = try allocator.create(c.Tokenizer);
    tokenizer.* = .{};
    const vocab_size: usize = if (gguf.vocab_lengths) |vl| vl.len else 0;
    if (vocab_size > 0) {
        config.vocab_size = vocab_size;
        tokenizer.vocab_lengths = gguf.vocab_lengths.?;
        tokenizer.vocab_sparse_cum = gguf.vocab_sparse_cum.?;
        tokenizer.vocab_sparse_step = gguf.vocab_sparse_step;
        tokenizer.vocab_size = vocab_size;
    }
    tokenizer.bos_token = @intFromFloat(gguf.number_meta.get("tokenizer.ggml.bos_token_id") orelse 1.0);
    tokenizer.eos_token = @intFromFloat(gguf.number_meta.get("tokenizer.ggml.eos_token_id") orelse 2.0);
    tokenizer.eot_token = @bitCast(@as(i32, -1));

    ctx.config = config;
    ctx.tokenizer = tokenizer;

    {
        var im_end = find_special_token(ctx, "<|im_end|>");
        if (im_end < 0) im_end = tokenizer.eos_token;
        tokenizer.eot_token = im_end;
    }

    if (on_progress) |cb| cb("Loading weights...");
    const weights = try allocator.create(c.Weights);
    weights.* = try loadWeights(ctx, &gguf, allocator);
    ctx.weights = weights;

    if (on_progress) |cb| cb("Allocating scratch buffers...");

    const head_size = config.head_dim;
    const q_dim = config.n_heads * head_size;
    const max_dim = if (config.dim > q_dim) config.dim else q_dim;
    const max_cols = if (max_dim > config.hidden_dim) max_dim else config.hidden_dim;
    const x_q8_size = (max_cols >> 5) * 34;

    ctx.x_q8_buf = try allocator.alloc(u8, x_q8_size);
    ctx.x_q8_int8_buf = @ptrCast(ctx.x_q8_buf.?);
    ctx.matmul_deq_buf = try allocator.alloc(f32, 4 * max_cols);

    if (on_progress) |cb| cb("Creating run state...");
    const state = try allocator.create(c.RunState);
    state.* = try createRunState(config, ctx.top_k, allocator);
    ctx.state = state;

    if (on_progress) |cb| cb("Model loaded!");
}
