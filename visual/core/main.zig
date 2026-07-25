const std = @import("std");
const model = @import("model.zig");

pub fn main() !void {
    var input: [3136]f32 = undefined;
    const bytes = std.mem.sliceAsBytes(input[0..]);
    var offset: usize = 0;
    while (offset < bytes.len) {
        const n = try std.posix.read(0, bytes[offset..]);
        if (n == 0) break;
        offset += n;
    }

    const net = model.Net.load();
    var work_buf: [2 * 5 * 56 * 56 + 9]f32 = undefined;
    const logits = net.forward(&input, &work_buf);

    var buf: [512]u8 = undefined;
    var pos: usize = 0;
    for (logits, 0..) |v, i| {
        if (i > 0) {
            buf[pos] = ' ';
            pos += 1;
        }
        const s = try std.fmt.bufPrint(buf[pos..], "{d:.8}", .{v});
        pos += s.len;
    }
    buf[pos] = '\n';
    pos += 1;
    _ = std.c.write(1, buf[0..pos].ptr, pos);
}
