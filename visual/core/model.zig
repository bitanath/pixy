const std = @import("std");
const ops = @import("ops.zig");

const weights_bytes = @embedFile("assets/weights.fw");

const max_intermediate = 5 * 56 * 56;

const layer_sizes = [_]usize{
    45, 5, 225, 5, 360, 8, 576, 8, 1152, 16, 313600, 100, 900, 9,
};

pub const Net = struct {
    layers: [14][]const f32,

    pub fn load() Net {
        var net = Net{ .layers = undefined };
        var offset: usize = 0;
        const base: [*]const u8 = weights_bytes.ptr;
        for (&layer_sizes, &net.layers) |numel, *layer| {
            const ptr: [*]const f32 = @ptrCast(@alignCast(base + offset));
            layer.* = ptr[0..numel];
            offset += numel * 4;
        }
        return net;
    }

    pub fn forward(self: *const Net, input: []const f32, work_buf: []f32) [9]f32 {
        var output: [9]f32 = undefined;
        const buf_a = work_buf[0..max_intermediate];
        const buf_b = work_buf[max_intermediate..][0..max_intermediate];

        const l = self.layers;

        ops.conv2d(input, l[0], l[1], 1, 56, 56, 5, 3, 3, 1, buf_a);
        ops.relu(buf_a[0..(5 * 56 * 56)]);

        ops.conv2d(buf_a[0..(5 * 56 * 56)], l[2], l[3], 5, 56, 56, 5, 3, 3, 1, buf_b);
        ops.relu(buf_b[0..(5 * 56 * 56)]);

        ops.maxPool2dSimd(buf_b[0..(5 * 56 * 56)], 5, 56, 56, buf_a);

        ops.conv2d(buf_a[0..(5 * 28 * 28)], l[4], l[5], 5, 28, 28, 8, 3, 3, 1, buf_b);
        ops.relu(buf_b[0..(8 * 28 * 28)]);

        ops.conv2d(buf_b[0..(8 * 28 * 28)], l[6], l[7], 8, 28, 28, 8, 3, 3, 1, buf_a);
        ops.relu(buf_a[0..(8 * 28 * 28)]);

        ops.maxPool2dSimd(buf_a[0..(8 * 28 * 28)], 8, 28, 28, buf_b);

        ops.conv2d(buf_b[0..(8 * 14 * 14)], l[8], l[9], 8, 14, 14, 16, 3, 3, 1, buf_a);
        ops.relu(buf_a[0..(16 * 14 * 14)]);

        ops.linear(buf_a[0..3136], l[10], l[11], 100, 3136, buf_b);
        ops.relu(buf_b[0..100]);

        ops.linear(buf_b[0..100], l[12], l[13], 9, 100, &output);

        return output;
    }
};
