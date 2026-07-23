const std = @import("std");
const build_opts = @import("build_opts");

pub const debugLogs = build_opts.debugLogs;

pub fn log(args: anytype) void {
    if (!debugLogs) return;

    var buffer: [4096]u8 = undefined;
    var w = std.Io.Writer.fixed(&buffer);

    const info = @typeInfo(@TypeOf(args));
    if (info != .@"struct" or !info.@"struct".is_tuple) {
        @compileError("log expects a tuple, e.g., log(.{ \"hello\", 42 })");
    }

    inline for (info.@"struct".fields, 0..) |field, i| {
        if (i > 0) {
            w.print(" ", .{}) catch return;
        }
        formatValue(&w, @field(args, field.name)) catch return;
    }
    w.print("\n", .{}) catch return;

    std.debug.print("{s}", .{w.buffered()});
}

fn formatValue(w: *std.Io.Writer, value: anytype) !void {
    const T = @TypeOf(value);
    const info = @typeInfo(T);

    switch (info) {
        .bool => {
            try w.writeAll(if (value) "true" else "false");
        },
        .int, .comptime_int => {
            try w.print("{d}", .{value});
        },
        .float, .comptime_float => {
            try w.print("{d}", .{value});
        },
        .pointer => {
            const ptr = info.pointer;
            if (ptr.size == .slice and ptr.child == u8) {
                try w.writeAll(value);
            } else if (ptr.size == .one) {
                const child_info = @typeInfo(ptr.child);
                if (child_info == .array and child_info.array.child == u8) {
                    try w.writeAll(value);
                } else {
                    try formatValue(w, value.*);
                }
            } else {
                try w.print("{any}", .{value});
            }
        },
        .array => {
            if (info.array.child == u8) {
                try w.writeAll(&value);
            } else {
                try w.print("{any}", .{value});
            }
        },
        .@"struct" => {
            if (info.@"struct".is_tuple) {
                inline for (info.@"struct".fields, 0..) |field, fi| {
                    if (fi > 0) try w.writeAll(" ");
                    try formatValue(w, @field(value, field.name));
                }
            } else {
                try w.print("{any}", .{value});
            }
        },
        .@"enum", .@"union" => {
            try w.print("{any}", .{value});
        },
        .optional => {
            if (value) |v| {
                try formatValue(w, v);
            } else {
                try w.writeAll("null");
            }
        },
        .void => try w.writeAll("void"),
        .null => try w.writeAll("null"),
        else => try w.print("{any}", .{value}),
    }
}
