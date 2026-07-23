const std = @import("std");
const c = @import("constants.zig");
const genny = @import("genny.zig");

const model_data = @embedFile("assets/tiny.gguf");

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

pub export fn free_string(s: [*:0]u8) void {
    const allocator = std.heap.c_allocator;
    const len = std.mem.len(s);
    allocator.free(s[0..len :0]); //NOTE I mean the OS should clean it up anyway but idk I dont wanna take any chances
}
