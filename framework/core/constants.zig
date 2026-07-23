const std = @import("std");

//MARK:- quantization types mapped to types in gguf file

pub const GGML_TYPE = enum(u32) {
    F32 = 0,
    Byte = 8,
    Nibble = 13,
    Word = 14,
    _,
};

pub const GGUF_TYPE = enum(u32) {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
    _,
};

//MARK:- GGUF quirks that need to be mapped somewhere idk

pub const GGUF_MAGIC: u32 = 0x46554747;

pub const QK8_0: usize = 32;
pub const QK_K: usize = 256;

pub const Q8_0_BLOCK_SIZE: usize = 34;
pub const PREFILL_BATCH_SIZE: usize = 80;

//MARK: skippables

pub const SKIP_METADATA_KEYS: [28][]const u8 = .{
    "general.author",
    "general.basename",
    "general.base_model.count",
    "general.datasets",
    "general.description",
    "general.file_type",
    "general.finetune",
    "general.languages",
    "general.name",
    "general.organization",
    "general.quantization_version",
    "general.size_label",
    "general.source.huggingface.repository",
    "general.source.url",
    "general.tags",
    "general.type",
    "general.url",
    "general.version",
    "tokenizer.chat_template",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token",
    "tokenizer.ggml.add_space_prefix",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.scores",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.unknown_token_id",
};

pub fn shouldSkipKey(key: []const u8) bool {
    for (SKIP_METADATA_KEYS) |k| {
        if (std.mem.eql(u8, k, key)) return true;
    }
    return false;
}

//MARK:- GGUF structs to map from the raw tensors to our functions

pub const GGUFTensor = struct {
    dims: []const u32 = &.{},
    quant_type: GGML_TYPE = .F32,
    offset: usize = 0,
    n_elements: usize = 0,
};

pub const GGUFMetadata = struct {
    version: u32 = 0,
    string_meta: std.StringHashMap([]const u8),
    number_meta: std.StringHashMap(f64),
    tensors: std.StringHashMap(GGUFTensor),
    tensor_data_offset: usize = 0,
    vocab_lengths: ?[]const u16 = null,
    vocab_sparse_cum: ?[]const u32 = null,
    vocab_sparse_step: u32 = 256,

    pub fn init(allocator: std.mem.Allocator) GGUFMetadata {
        return .{
            .string_meta = std.StringHashMap([]const u8).init(allocator),
            .number_meta = std.StringHashMap(f64).init(allocator),
            .tensors = std.StringHashMap(GGUFTensor).init(allocator),
        };
    }

    pub fn deinit(self: *GGUFMetadata) void {
        self.string_meta.deinit();
        self.number_meta.deinit();
        self.tensors.deinit();
    }
};

pub const QuantizedTensor = struct {
    data_offset: usize = 0,
    quant_type: GGML_TYPE = .F32,
    rows: usize = 0,
    cols: usize = 0,
    row_size: usize = 0,
    dot_func: ?VecDotFunc = null,
    dot_q8_func: ?VecDotQ8Func = null,
    deq_row_func: ?DeqRowFunc = null,
    local_u8: ?[]const u8 = null,
    local_i8: ?[]const i8 = null,
    data_ptr: i64 = -1,
};

//MARK: Internal model configuration

pub const Config = struct {
    dim: usize = 0,
    hidden_dim: usize = 0,
    n_layers: usize = 0,
    n_heads: usize = 0,
    n_kv_heads: usize = 0,
    vocab_size: usize = 0,
    seq_len: usize = 0,
    rope_theta: f64 = 0,
    head_dim: usize = 0,
    rms_norm_eps: f64 = 0,
    ssm_d_conv: usize = 0,
    ssm_d_inner: usize = 0,
    ssm_d_state: usize = 0,
    ssm_dt_rank: usize = 0,
    ssm_n_group: usize = 0,
};

pub const Tokenizer = struct {
    vocab_lengths: []const u16 = &.{},
    vocab_sparse_cum: []const u32 = &.{},
    vocab_sparse_step: u32 = 256,
    vocab_size: usize = 0,
    bos_token: i32 = 1,
    eos_token: i32 = 2,
    eot_token: i32 = -1,
};

pub const LayerWeights = struct {
    rms_att_weight: ?[]const f32 = null,
    rms_ffn_weight: ?[]const f32 = null,
    wq: ?QuantizedTensor = null,
    wk: ?QuantizedTensor = null,
    wv: ?QuantizedTensor = null,
    wo: ?QuantizedTensor = null,
    w1: ?QuantizedTensor = null,
    w2: ?QuantizedTensor = null,
    w3: ?QuantizedTensor = null,
    ssm_in: ?QuantizedTensor = null,
    ssm_conv1d_w: ?[]const f32 = null,
    ssm_conv1d_b: ?[]const f32 = null,
    ssm_dt_b: ?[]const f32 = null,
    ssm_a: ?[]const f32 = null,
    ssm_d: ?[]const f32 = null,
    ssm_out: ?QuantizedTensor = null,
    ssm_norm: ?[]const f32 = null,
};

pub const Weights = struct {
    token_embedding: ?QuantizedTensor = null,
    layers: []LayerWeights = &.{},
    rms_final_weight: ?[]const f32 = null,
    wcls: ?QuantizedTensor = null,
};

//NOTE: This is wehat gets passed around, we try to do a pass by reference on every function

pub const RunState = struct {
    x: ?[]f32 = null,
    xb: ?[]f32 = null,
    xb2: ?[]f32 = null,
    hb: ?[]f32 = null,
    hb2: ?[]f32 = null,
    q: ?[]f32 = null,
    k: ?[]f32 = null,
    v: ?[]f32 = null,
    att: ?[]f32 = null,
    logits: ?[]f32 = null,

    batch_dim: usize = 0,
    batch_max_dim: usize = 0,
    batch_q_dim: usize = 0,
    batch_kv_dim: usize = 0,
    batch_hidden_dim: usize = 0,
    batch_x_q8_size: usize = 0,
    batch_matmul_deq_cols: usize = 0,
    batch_buffers_ready: bool = false,

    key_cache: ?[]u8 = null,
    value_cache: ?[]u8 = null,
    key_cache_int8: ?[]i8 = null,
    value_cache_int8: ?[]i8 = null,
    q_q8: ?[]u8 = null,
    q_q8i8: ?[]i8 = null,
    head_seq_bytes: usize = 0,

    rope_cos_all: ?[]f32 = null,
    rope_sin_all: ?[]f32 = null,
    rope_freqs: ?[]f32 = null,
    rope_size: usize = 0,
    rope_cos_layer: []?[]f32 = &.{},
    rope_sin_layer: []?[]f32 = &.{
        // These will be allocated at runtime, hopefully
    },

    kv_mul: usize = 0,
    kv_cache_layer_size: usize = 0,
    kv_capacity: usize = 0,
    attn_scale: f64 = 0,
    dim: usize = 0,
    n_heads: usize = 0,
    n_kv_heads: usize = 0,
    n_layers: usize = 0,
    seq_len: usize = 0,
    hidden_dim: usize = 0,
    vocab_size: usize = 0,
    rms_norm_eps: f64 = 0,
    inv_dim: f64 = 0,
    inv_head_size: f64 = 0,

    top_k_indices: ?[]i32 = null,
    top_k_values: ?[]f32 = null,
    head_q_offsets: ?[]i32 = null,
    head_kv_idx: ?[]i32 = null,
    head_att_offsets: ?[]i32 = null,
    head_kv_byte_offsets: ?[]i32 = null,
    head_bytes_q8: usize = 0,

    batch_x: [PREFILL_BATCH_SIZE]?[]f32 = .{null} ** PREFILL_BATCH_SIZE,
    batch_xb: [PREFILL_BATCH_SIZE]?[]f32 = .{null} ** PREFILL_BATCH_SIZE,
    batch_xb2: [PREFILL_BATCH_SIZE]?[]f32 = .{null} ** PREFILL_BATCH_SIZE,
    batch_q_arr: [PREFILL_BATCH_SIZE]?[]f32 = .{null} ** PREFILL_BATCH_SIZE,
    batch_k_arr: [PREFILL_BATCH_SIZE]?[]f32 = .{null} ** PREFILL_BATCH_SIZE,
    batch_v_arr: [PREFILL_BATCH_SIZE]?[]f32 = .{null} ** PREFILL_BATCH_SIZE,
    batch_hb: [PREFILL_BATCH_SIZE]?[]f32 = .{null} ** PREFILL_BATCH_SIZE,
    batch_hb2: [PREFILL_BATCH_SIZE]?[]f32 = .{null} ** PREFILL_BATCH_SIZE,
    batch_q8: [PREFILL_BATCH_SIZE]?[]u8 = .{null} ** PREFILL_BATCH_SIZE,
    batch_q8i8: [PREFILL_BATCH_SIZE]?[]i8 = .{null} ** PREFILL_BATCH_SIZE,

    conv_state: []?[]f32 = &.{},
    ssm_state: []?[]f32 = &.{},
    ssm_proj_buf: ?[]f32 = null,
    ssm_xbc_conv: ?[]f32 = null,
    ssm_y_buf: ?[]f32 = null,
};

//MARK Chat Messages and System context

pub const ChatMessage = struct {
    role: []const u8 = "",
    content: []const u8 = "",
};

pub const FnRender = *const fn (token: []const u8) void;
pub const FnProgress = *const fn (msg: []const u8) void;

pub const LlamaInput = struct {
    type: []const u8 = "",
    model: ?[]const u8 = null,
    chat_history: ?[]const ChatMessage = null,
    cb_render: ?FnRender = null,
    on_progress: ?FnProgress = null,
    max_tokens: i32 = 0,
    context_size: usize = 0,
    system_prompt: []const u8 = "",
    temperature: f64 = 0,
    top_p: f64 = 0,
    top_k: i32 = 0,
};

pub const Context = struct {
    // Model data
    config: ?*Config = null,
    weights: ?*Weights = null,
    state: ?*RunState = null,
    tokenizer: ?*Tokenizer = null,

    // GGUF raw data
    gguf_data: ?[]const u8 = null,
    gguf_uint8: ?[]u8 = null,
    gguf_int8: ?[]i8 = null,
    data_view_offset: usize = 0,
    gguf_base_offset: i64 = -1,

    // Scratch buffers
    x_q8_buf: ?[]u8 = null,
    x_q8_int8_buf: ?[]i8 = null,
    matmul_deq_buf: ?[]f32 = null,

    // Sampling
    temperature: f64 = 0.9,
    top_p: f64 = 0.9,
    top_k: i32 = 40,
    system_prompt: []const u8 = "You are a helpful assistant.",
    max_tokens: i32 = -1,
    context_size: usize = 0,

    // Allocator
    allocator: ?std.mem.Allocator = null,

    // Render callback
    cb_render: ?FnRender = null,

    // Tokenizer trie
    trie_node_id: ?[]i32 = null,
    trie_child_start: ?[]i32 = null,
    trie_edge_char: ?[]u8 = null,
    trie_edge_target: ?[]i32 = null,

    // Tiktoken maps
    tiktoken_byte_to_unicode: ?std.AutoHashMap(i32, i32) = null,
    tiktoken_unicode_to_byte: ?std.AutoHashMap(i32, i32) = null,
    token_to_bytes_buf: [256]u8 = .{0} ** 256,

    // Perf measurement (persists across conversation calls)
    perf_baseline_rss: u64 = 0,
    perf_load_time_ns: u64 = 0,
    perf_load_rss: u64 = 0,
    perf_gen_time_ns: u64 = 0,
    perf_gen_rss: u64 = 0,
};

//MARK Zig oddity we do a function pointer instead of a normal callback like signature, I can only assume this is similar to a JS first class function passing, but we will find out

pub const VecDotFunc = *const fn (
    ctx: *Context,
    x: []const f32,
    src_offset: usize,
    n: usize,
) f64;

pub const VecDotQ8Func = *const fn (
    ctx: *Context,
    x_q8: []const u8,
    x_q8i8: []const i8,
    src_offset: usize,
    n: usize,
) f64;

pub const DeqRowFunc = *const fn (
    ctx: *Context,
    u8_data: []const u8,
    off: usize,
    dst: []f32,
    dst_off: usize,
    cols: usize,
    i8_data: []const i8,
) void;
