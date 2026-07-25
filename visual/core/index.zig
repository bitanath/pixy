const model = @import("model.zig");

export fn convnet_infer(image_bytes: [*]const u8, out_logits: [*]f32) void {
    const net = model.Net.load();
    var input: [56 * 56]f32 = undefined;
    for (0..56 * 56) |i| {
        input[i] = @as(f32, @floatFromInt(image_bytes[i])) / 255.0;
    }
    var work_buf: [2 * 5 * 56 * 56 + 9]f32 = undefined;
    const logits = net.forward(&input, &work_buf);
    for (0..9) |i| {
        out_logits[i] = logits[i];
    }
}
