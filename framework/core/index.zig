const std = @import("std");
const c = @import("constants.zig");
const genny = @import("genny.zig");

const model_data = @embedFile("assets/gigantic.gguf");

pub export fn generate_conversation(prompt: [*:0]const u8, system_prompt: [*:0]const u8) ?[*:0]u8 {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var ctx = c.Context{};

    _ = genny.conversation(&ctx, .{
        .type = "load",
        .system_prompt = std.mem.sliceTo(system_prompt, 0),
        .max_tokens = 512,
        .context_size = 4096,
        .temperature = 0.3,
    }, model_data, null, allocator) catch return null;

    const result = genny.conversation(&ctx, .{
        .type = "generate",
        .chat_history = &.{
            .{ .role = "user", .content = std.mem.sliceTo(prompt, 0) },
        },
    }, model_data, null, allocator) catch return null;

    const out = std.heap.c_allocator.allocSentinel(u8, result.len, 0) catch return null;
    @memcpy(out, result);
    return out.ptr;
}

pub export fn generate_turnwise_conversation(
    messages: [*]const c.ChatMessageC,
    num_messages: usize,
    system_prompt: [*:0]const u8,
    max_tokens: i32,
    temperature: f64,
    context_size: usize,
) ?[*:0]u8 {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var ctx = c.Context{};

    _ = genny.conversation(&ctx, .{
        .type = "load",
        .system_prompt = std.mem.sliceTo(system_prompt, 0),
        .max_tokens = max_tokens,
        .context_size = context_size,
        .temperature = temperature,
    }, model_data, null, null, allocator) catch return null;

    const zig_messages = allocator.alloc(c.ChatMessage, num_messages) catch return null;
    for (0..num_messages) |i| {
        zig_messages[i] = .{
            .role = std.mem.sliceTo(messages[i].role, 0),
            .content = std.mem.sliceTo(messages[i].content, 0),
        };
    }

    const result = genny.conversation(&ctx, .{
        .type = "generate",
        .chat_history = zig_messages,
    }, model_data, null, null, allocator) catch return null;

    const out = std.heap.c_allocator.allocSentinel(u8, result.len, 0) catch return null;
    @memcpy(out, result);
    return out.ptr;
}

pub export fn free_string(s: [*:0]u8) void {
    const allocator = std.heap.c_allocator;
    const len = std.mem.len(s);
    allocator.free(s[0..len :0]); //NOTE I mean the OS should clean it up anyway but idk I dont wanna take any chances
}

// ---- Session streaming (additive; whole-answer exports above unchanged) ----
// Lifecycle: stream_start() (null = busy/failed) -> stream_poll() until -1
// (done) or -2 (error) -> stream_free() exactly once. poll() returns fresh
// byte counts, 0 when nothing is new yet. free() cancels and joins.
const stream = @import("stream.zig");

pub export fn stream_start(
    messages: [*]const c.ChatMessageC,
    num_messages: usize,
    system_prompt: [*:0]const u8,
    max_tokens: i32,
    temperature: f64,
    context_size: usize,
) ?*anyopaque {
    return stream.streamStart(messages, num_messages, system_prompt, max_tokens, temperature, context_size, model_data);
}

pub export fn stream_poll(handle: ?*anyopaque, out: [*]u8, cap: usize) isize {
    return stream.streamPoll(handle, out, cap);
}

pub export fn stream_free(handle: ?*anyopaque) void {
    stream.streamFree(handle);
}
