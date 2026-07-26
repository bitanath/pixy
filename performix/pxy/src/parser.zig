const std = @import("std");
const types = @import("types.zig");
const Io = std.Io;
const Dir = Io.Dir;

fn readFile(
    dir: Dir,
    io: Io,
    sub_path: []const u8,
    allocator: std.mem.Allocator,
    max_size: usize,
) !?[]const u8 {
    return Dir.readFileAlloc(dir, io, sub_path, allocator, Io.Limit.limited64(max_size)) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => |e| return e,
    };
}

fn parseCSVFields(line: []const u8, allocator: std.mem.Allocator) !std.ArrayList([]const u8) {
    var fields: std.ArrayList([]const u8) = .empty;
    var i: usize = 0;
    while (i < line.len) {
        if (line[i] == ',') {
            try fields.append(allocator, "");
            i += 1;
            continue;
        }
        if (line[i] == '"') {
            const end = std.mem.indexOfScalarPos(u8, line, i + 1, '"') orelse line.len;
            try fields.append(allocator, line[i + 1 .. end]);
            i = end + 1;
            if (i < line.len and line[i] == ',') i += 1;
        } else {
            const end = std.mem.indexOfScalarPos(u8, line, i, ',') orelse line.len;
            try fields.append(allocator, std.mem.trim(u8, line[i..end], " "));
            i = end + 1;
        }
    }
    return fields;
}

fn parseU64(field: []const u8) u64 {
    return std.fmt.parseUnsigned(u64, field, 10) catch 0;
}

fn parseU32(field: []const u8) u32 {
    return @intCast(parseU64(field));
}

fn openOutputDir(run_dir: Dir, io: Io) !Dir {
    const tool = try Dir.openDir(run_dir, io, "tool", .{});
    defer tool.close(io);
    const neoprof = try Dir.openDir(tool, io, "neoprof", .{});
    defer neoprof.close(io);
    const zero = try Dir.openDir(neoprof, io, "0", .{});
    defer zero.close(io);
    return Dir.openDir(zero, io, "output", .{ .iterate = true });
}

fn openNeoprofDir(run_dir: Dir, io: Io) !Dir {
    const tool = try Dir.openDir(run_dir, io, "tool", .{});
    defer tool.close(io);
    const neoprof = try Dir.openDir(tool, io, "neoprof", .{});
    defer neoprof.close(io);
    return Dir.openDir(neoprof, io, "0", .{});
}

pub fn parseMetadata(run_dir: Dir, io: Io, allocator: std.mem.Allocator) !types.RunMetadata {
    var result = types.RunMetadata{
        .workload = "unknown",
        .target = "unknown",
        .engine_version = "unknown",
        .recipe = "unknown",
        .start_time = "unknown",
        .end_time = "unknown",
        .total_samples = 0,
        .answer = "",
    };

    const content = (try readFile(run_dir, io, "metadata.json", allocator, 64 * 1024)) orelse return result;

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;

    if (obj.get("run.workload.cmdline")) |v| result.workload = v.string;
    if (obj.get("target.name")) |v| result.target = v.string;
    if (obj.get("engine.version")) |v| result.engine_version = v.string;
    if (obj.get("run.recipe_name")) |v| result.recipe = v.string;
    if (obj.get("run.start_time")) |v| result.start_time = v.string;
    if (obj.get("run.end_time")) |v| result.end_time = v.string;

    return result;
}

pub fn parseCaptureLogErr(run_dir: Dir, io: Io, allocator: std.mem.Allocator) ?[]const u8 {
    const neoprof = openNeoprofDir(run_dir, io) catch return null;
    defer neoprof.close(io);

    const text = readFile(neoprof, io, "capture_log_err.txt", allocator, 64 * 1024) catch return null;
    const content = text orelse return null;

    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \r");
        if (std.mem.indexOf(u8, trimmed, "STDOUT:")) |idx| {
            const rest = std.mem.trim(u8, trimmed[idx + "STDOUT:".len ..], " ");
            return rest;
        }
    }
    return null;
}

pub fn parseFunctions(run_dir: Dir, io: Io, allocator: std.mem.Allocator) ![]types.FunctionEntry {
    const out_dir = openOutputDir(run_dir, io) catch return allocator.alloc(types.FunctionEntry, 0);
    defer out_dir.close(io);

    const content = (try readFile(out_dir, io, "functions-capture-periodic_sampling.csv", allocator, 1024 * 1024)) orelse return allocator.alloc(types.FunctionEntry, 0);

    var entries: std.ArrayList(types.FunctionEntry) = .empty;
    var lines = std.mem.splitScalar(u8, content, '\n');
    _ = lines.next();

    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const fields = try parseCSVFields(line, allocator);
        if (fields.items.len < 4) continue;
        try entries.append(allocator, .{
            .samples = parseU64(fields.items[0]),
            .uid = parseU32(fields.items[1]),
            .image = fields.items[2],
            .name = fields.items[3],
        });
    }

    return entries.items;
}

pub fn parseCallPaths(run_dir: Dir, io: Io, allocator: std.mem.Allocator) ![]types.CallPathEntry {
    const out_dir = openOutputDir(run_dir, io) catch return allocator.alloc(types.CallPathEntry, 0);
    defer out_dir.close(io);

    const content = (try readFile(out_dir, io, "callpaths-capture-periodic_sampling.csv", allocator, 1024 * 1024)) orelse return allocator.alloc(types.CallPathEntry, 0);

    var entries: std.ArrayList(types.CallPathEntry) = .empty;
    var lines = std.mem.splitScalar(u8, content, '\n');
    _ = lines.next();

    while (lines.next()) |line| {
        if (line.len == 0) continue;
        const fields = try parseCSVFields(line, allocator);
        if (fields.items.len < 9) continue;
        try entries.append(allocator, .{
            .total_samples = parseU64(fields.items[0]),
            .self_samples = parseU64(fields.items[1]),
            .uid = parseU32(fields.items[2]),
            .image = fields.items[3],
            .name = fields.items[4],
            .depth = parseU32(fields.items[6]),
            .node_id = parseU32(fields.items[7]),
            .parent_node_id = parseU32(fields.items[8]),
        });
    }

    return entries.items;
}

pub fn parseSourceLines(run_dir: Dir, io: Io, allocator: std.mem.Allocator) ![]types.SourceLineEntry {
    const out_dir = openOutputDir(run_dir, io) catch return allocator.alloc(types.SourceLineEntry, 0);
    defer out_dir.close(io);

    var entries: std.ArrayList(types.SourceLineEntry) = .empty;

    var iter = Dir.iterate(out_dir);
    while (iter.next(io) catch null) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.startsWith(u8, entry.name, "sources-capture-periodic_sampling-")) continue;

        const content = (try readFile(out_dir, io, entry.name, allocator, 4 * 1024 * 1024)) orelse continue;
        var lines = std.mem.splitScalar(u8, content, '\n');
        _ = lines.next();

        while (lines.next()) |line| {
            if (line.len == 0) continue;
            const fields = try parseCSVFields(line, allocator);
            if (fields.items.len < 4) continue;
            const samples = parseU64(fields.items[3]);
            if (samples == 0) continue;
            try entries.append(allocator, .{
                .file = fields.items[0],
                .line = parseU32(fields.items[1]),
                .samples = samples,
                .function = if (fields.items.len > 4) fields.items[4] else "",
            });
        }
    }

    return entries.items;
}

pub fn parseAll(run_dir: Dir, io: Io, allocator: std.mem.Allocator) !types.ParsedData {
    var metadata = try parseMetadata(run_dir, io, allocator);
    const functions = try parseFunctions(run_dir, io, allocator);
    const callpath = try parseCallPaths(run_dir, io, allocator);
    const sources = try parseSourceLines(run_dir, io, allocator);

    if (callpath.len > 0) {
        metadata.total_samples = callpath[0].total_samples;
    }

    if (parseCaptureLogErr(run_dir, io, allocator)) |answer| {
        metadata.answer = answer;
    }

    return .{
        .metadata = metadata,
        .functions = functions,
        .callpath = callpath,
        .sources = sources,
    };
}
