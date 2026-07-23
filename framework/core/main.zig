const std = @import("std");
const index = @import("index.zig");

pub fn main() !void {
    var buf: [4096]u8 = undefined;
    const n = std.c.read(0, &buf, buf.len);
    if (n < 0) std.process.exit(1);
    const len = @min(@as(usize, @intCast(n)), buf.len - 1);
    buf[len] = 0;

    const prompt: [*:0]u8 = @ptrCast(&buf);
    const system_prompt: [*:0]const u8 = "You are a helpful assistant.";
    const result = index.generate_conversation(prompt, system_prompt) orelse {
        std.process.exit(1);
    };
    defer index.free_string(result);

    std.debug.print("Output: {s}\n", .{result});
}
