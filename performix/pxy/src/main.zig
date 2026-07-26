const std = @import("std");

const types = @import("typefication.zig");
const renderer = @import("renderer.zig");
const parser = @import("parser.zig");
const Io = std.Io;
const Dir = Io.Dir;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.arena.allocator();

    const arg_slice = try init.minimal.args.toSlice(allocator);
    if (arg_slice.len < 2) {
        std.process.fatal("Usage: pxy <performix-run-dir-or-zip>", .{});
    }

    const input_path = arg_slice[1];

    var temp_dir_path: ?[]const u8 = null;
    var temp_alloc: ?std.heap.ArenaAllocator = null;

    const run_dir = openRunDir(input_path, io, &temp_dir_path, &temp_alloc) catch |err| {
        std.process.fatal("Error opening '{s}': {}", .{ input_path, err });
    };
    defer {
        run_dir.close(io);
        if (temp_alloc) |*ta| {
            if (temp_dir_path) |p| {
                const cwd = Dir.cwd();
                Dir.deleteTree(cwd, io, p) catch {};
            }
            ta.deinit();
        }
    }

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    const data = try parser.parseAll(run_dir, io, arena_allocator);

    var stdout_file = Io.File.stdout();
    var buf: [4096]u8 = undefined;
    var stdout_fw = stdout_file.writer(io, &buf);
    try renderer.renderAll(&stdout_fw.interface, data);
    try stdout_fw.flush();
}

fn openRunDir(
    path: []const u8,
    io: Io,
    out_temp_path: *?[]const u8,
    out_temp_arena: *?std.heap.ArenaAllocator,
) !Dir {
    if (std.mem.endsWith(u8, path, ".zip")) {
        const cwd = Dir.cwd();
        var temp_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        const ta = temp_arena.allocator();

        const list_result = try std.process.run(ta, io, .{
            .argv = &[_][]const u8{ "unzip", "-Z1", path },
        });
        var list_lines = std.mem.splitScalar(u8, list_result.stdout, '\n');
        const first_entry = list_lines.next() orelse return error.InvalidZip;
        const slash = std.mem.indexOfScalar(u8, first_entry, '/') orelse return error.InvalidZip;
        const top_dir = first_entry[0..slash];
        const temp_dir = try std.fmt.allocPrint(ta, "/tmp/pxtree/{s}", .{top_dir});
        Dir.createDirPath(cwd, io, "/tmp/pxtree") catch {};

        _ = try std.process.run(ta, io, .{
            .argv = &[_][]const u8{ "unzip", "-o", path, "-d", "/tmp/pxtree" },
        });

        out_temp_path.* = temp_dir;
        out_temp_arena.* = temp_arena;
        return Dir.openDir(cwd, io, temp_dir, .{});
    }

    const cwd = Dir.cwd();
    return Dir.openDir(cwd, io, path, .{});
}
