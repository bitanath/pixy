const std = @import("std");
const types = @import("types.zig");

const R = "\x1b[0m";
const BOLD = "\x1b[1m";
const DIM = "\x1b[2m";
const CYAN = "\x1b[36m";
const RED_BOLD = "\x1b[1;31m";
const YELLOW_BOLD = "\x1b[1;33m";
const YELLOW = "\x1b[33m";

fn starRating(pct: f64) []const u8 {
    if (pct >= 40.0) return "*****";
    if (pct >= 20.0) return "****";
    if (pct >= 10.0) return "***";
    if (pct >= 5.0) return "**";
    if (pct >= 1.0) return "*";
    return "";
}

fn hotColor(pct: f64) []const u8 {
    if (pct >= 20.0) return RED_BOLD;
    if (pct >= 5.0) return YELLOW_BOLD;
    if (pct >= 1.0) return YELLOW;
    return "";
}

fn priorityLabel(pct: f64) []const u8 {
    if (pct >= 10.0) return "P0";
    if (pct >= 5.0) return "P1";
    if (pct >= 1.0) return "P2";
    if (pct >= 0.1) return "P3";
    return "P4";
}

fn parseIsoDuration(start: []const u8, end: []const u8) f64 {
    if (start.len < 19 or end.len < 19) return 0;
    const sh = std.fmt.parseInt(i64, start[11..13], 10) catch return 0;
    const sm = std.fmt.parseInt(i64, start[14..16], 10) catch return 0;
    const ss = std.fmt.parseInt(i64, start[17..19], 10) catch return 0;
    const eh = std.fmt.parseInt(i64, end[11..13], 10) catch return 0;
    const em = std.fmt.parseInt(i64, end[14..16], 10) catch return 0;
    const es = std.fmt.parseInt(i64, end[17..19], 10) catch return 0;
    const start_sec = sh * 3600 + sm * 60 + ss;
    const end_sec = eh * 3600 + em * 60 + es;
    return @as(f64, @floatFromInt(end_sec - start_sec));
}

pub fn renderAll(writer: anytype, data: types.ParsedData) !void {
    try renderHeader(writer, data.metadata);
    try writer.writeAll("\n");
    try renderFunctionTable(writer, data.functions, data.metadata.total_samples);
    try writer.writeAll("\n");
    try renderPriorityList(writer, data.functions, data.callpath, data.metadata);
}

fn renderHeader(writer: anytype, meta: types.RunMetadata) !void {
    const dur = parseIsoDuration(meta.start_time, meta.end_time);
    try writer.print("{s}{s}PXY - Performix Profiling Report{s}\n", .{ CYAN, BOLD, R });
    try writer.print("{s}═══════════════════════════{s}\n", .{ CYAN, R });
    try writer.print("{s}Workload:   {s}{s}{s}{s}\n", .{ CYAN, R, BOLD, meta.workload, R });
    try writer.print("{s}Target:     {s}{s}{s}\n", .{ CYAN, R, meta.target, R });
    try writer.print("{s}Engine:     {s}{s}{s}\n", .{ CYAN, R, meta.engine_version, R });
    try writer.print("{s}Recipe:     {s}{s}{s}\n", .{ CYAN, R, meta.recipe, R });
    try writer.print("{s}Duration:   {s}{s}  {s}  {s}{s}\n", .{ CYAN, R, meta.start_time, "→", meta.end_time, R });
    if (dur > 0) {
        var dbuf: [16]u8 = undefined;
        const ds = std.fmt.bufPrint(&dbuf, "{d:.0}", .{dur}) catch "";
        try writer.print("{s}            {s}{s}s total{s}\n", .{ CYAN, R, ds, R });
    }
    try writer.print("{s}Samples:    {s}{s}{d}{s}\n", .{ CYAN, R, BOLD, meta.total_samples, R });
    if (meta.answer.len > 0) try writer.print("{s}Answer:     {s}{s}{s}\n", .{ CYAN, R, meta.answer, R });
}

fn renderFunctionTable(writer: anytype, functions: []types.FunctionEntry, total: u64) !void {
    try writer.print("\n{s}{s}Flat Function Ranking{s}\n", .{ CYAN, BOLD, R });
    try writer.print("{s}────────────────────────────────────────────────────────────{s}\n", .{ CYAN, R });
    try writer.print("{s}{s:>2}  {s:>5}  {s:>5}  {s:>5}  {s:>6}  {s:<48}{s}\n", .{
        DIM, "#", "Samps", "    %", " Cum%", "Payoff", "Function", R
    });

    var cum: f64 = 0.0;
    var idx: u32 = 1;
    for (functions) |f| {
        if (!std.mem.eql(u8, f.image, "llm-linux")) continue;
        const pct = if (total > 0) @as(f64, @floatFromInt(f.samples)) / @as(f64, @floatFromInt(total)) * 100.0 else 0.0;
        cum += pct;
        const hc = hotColor(pct);
        const stars = starRating(pct);
        var pbuf: [16]u8 = undefined;
        var stars_buf: [32]u8 = undefined;
        const pct_s = std.fmt.bufPrint(&pbuf, "{d:5.1}", .{pct}) catch "???.?";
        const stars_s = if (stars.len > 0) (std.fmt.bufPrint(&stars_buf, "{s} ", .{stars}) catch "* ") else "";

        var payoff_buf: [6]u8 = undefined;
        var payoff_len: usize = 0;
        for (stars_s) |c| { if (payoff_len < 6) { payoff_buf[payoff_len] = c; payoff_len += 1; } }
        while (payoff_len < 6) : (payoff_len += 1) payoff_buf[payoff_len] = ' ';

        var fn_buf: [48]u8 = undefined;
        var fn_len: usize = 0;
        for (f.name) |c| { if (fn_len < 48) { fn_buf[fn_len] = c; fn_len += 1; } }
        while (fn_len < 48) : (fn_len += 1) fn_buf[fn_len] = ' ';

        try writer.print("{s}{d:>2}{s}  {d:>5}  {s}{s}{s}  {d:>5.1}  {s}{s}{s}  {s}{s}{s}\n", .{
            hc, idx, R,
            f.samples,
            hc, pct_s, R,
            cum,
            hc, payoff_buf[0..payoff_len], R,
            hc, fn_buf[0..fn_len], R,
        });
        idx += 1;
    }
}

fn cleanSymbolName(name: []const u8) []const u8 {
    var i: usize = 0;
    while (i < name.len) {
        const c = name[i];
        if (c >= 'a' and c <= 'z') break;
        if (c >= 'A' and c <= 'Z') break;
        if (c >= '0' and c <= '9') break;
        if (c == '_' or c == '.') break;
        i += 1;
    }
    return name[i..];
}

fn renderPriorityList(writer: anytype, functions: []types.FunctionEntry, callpath: []types.CallPathEntry, meta: types.RunMetadata) !void {
    try writer.print("\n{s}{s}Optimization Priority{s}\n", .{ CYAN, BOLD, R });
    try writer.print("{s}────────────────────────────────────────────────────────────{s}\n", .{ CYAN, R });

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var pt: std.ArrayList(*const types.FunctionEntry) = .empty;
    for (functions) |*f| {
        if (std.mem.eql(u8, f.image, "llm-linux")) try pt.append(alloc, f);
    }
    std.mem.sortUnstable(*const types.FunctionEntry, pt.items, {}, struct {
        fn lessThan(_: void, a: *const types.FunctionEntry, b: *const types.FunctionEntry) bool { return a.samples > b.samples; }
    }.lessThan);

    var uid_name: std.AutoArrayHashMapUnmanaged(u32, []const u8) = .empty;
    for (functions) |f| {
        uid_name.put(alloc, f.uid, f.name) catch {};
    }

    var node_map: std.AutoArrayHashMapUnmanaged(u32, *const types.CallPathEntry) = .empty;
    for (callpath) |*e| {
        node_map.put(alloc, e.node_id, e) catch {};
    }

    var uid_caller: std.AutoArrayHashMapUnmanaged(u32, []const u8) = .empty;
    var uid_depth: std.AutoArrayHashMapUnmanaged(u32, u32) = .empty;

    for (pt.items) |f| {
        var best_caller: []const u8 = "—";
        var best_samples: u64 = 0;
        var min_depth: u32 = 99;

        for (callpath) |*e| {
            if (e.uid != f.uid) continue;
            if (e.depth < min_depth) min_depth = e.depth;
            if (node_map.get(e.parent_node_id)) |parent| {
                const pname = uid_name.get(parent.uid) orelse cleanSymbolName(parent.name);
                if (e.total_samples > best_samples) {
                    best_samples = e.total_samples;
                    best_caller = pname;
                }
            }
        }

        if (min_depth == 99) min_depth = 0;
        uid_caller.put(alloc, f.uid, best_caller) catch {};
        uid_depth.put(alloc, f.uid, min_depth) catch {};
    }

    const duration_sec = parseIsoDuration(meta.start_time, meta.end_time);

    try writer.print("  {s}{s:>3}  {s:<48}  {s:>5}  {s:>4}  {s:<30}{s}\n", .{
        DIM, "Pri", "Function", "Time", "Dpth", "Pri Caller", R
    });

    for (pt.items) |f| {
        const pct = if (meta.total_samples > 0) @as(f64, @floatFromInt(f.samples)) / @as(f64, @floatFromInt(meta.total_samples)) * 100.0 else 0.0;
        const time_sec = @as(f64, @floatFromInt(f.samples)) / @as(f64, @floatFromInt(meta.total_samples)) * duration_sec;
        const depth = uid_depth.get(f.uid) orelse 0;
        const caller = uid_caller.get(f.uid) orelse "—";
        var time_buf: [8]u8 = undefined;
        const time_str = std.fmt.bufPrint(&time_buf, "{d:3.1}s", .{time_sec}) catch "0.0s";

        var fn_buf: [48]u8 = undefined;
        var fn_len: usize = 0;
        for (f.name) |c| { if (fn_len < 48) { fn_buf[fn_len] = c; fn_len += 1; } }
        while (fn_len < 48) : (fn_len += 1) fn_buf[fn_len] = ' ';

        var cl_buf: [30]u8 = undefined;
        var cl_len: usize = 0;
        for (caller) |c| { if (cl_len < 30) { cl_buf[cl_len] = c; cl_len += 1; } }
        while (cl_len < 30) : (cl_len += 1) cl_buf[cl_len] = ' ';

        const pc = if (pct >= 1.0) RED_BOLD else DIM;
        const hc = hotColor(pct);

        try writer.print("  {s}{s:>3}{s}  {s}{s}{s}  {s:>5}  {d:>4}  {s}{s}{s}\n", .{
            pc, priorityLabel(pct), R,
            hc, fn_buf[0..fn_len], R,
            time_str,
            depth,
            hc, cl_buf[0..cl_len], R,
        });
    }
}
