const std = @import("std");
const types = @import("typefication.zig");

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

fn shortFnName(name: []const u8) []const u8 {
    const paren = std.mem.indexOfScalar(u8, name, '(');
    if (paren) |p| {
        var end = p;
        while (end > 0 and (name[end - 1] == ' ' or name[end - 1] == '\t')) end -= 1;
        return name[0..end];
    }
    return name;
}

pub fn renderAll(writer: anytype, data: types.ParsedData) !void {
    try renderHeader(writer, data.metadata);
    try writer.writeAll("\n");

    switch (data.report_type) {
        .code_hotspots => {
            try renderFunctionTable(writer, data.functions, data.metadata.total_samples);
            try writer.writeAll("\n");
            try renderSourceHotspots(writer, data.functions, data.sources, data.metadata);
            try writer.writeAll("\n");
            try renderInsights(writer, data.functions, data.sources, data.metadata);
            try writer.writeAll("\n");
            try renderPriorityList(writer, data.functions, data.callpath, data.metadata);
        },
        .system_utilization => {
            try renderSystemUtil(writer, data.system_samples, data.metadata);
        },
    }
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
    if (meta.answer.len > 0) try writer.print("{s}Answer:     {s}{s}{s}\n", .{ CYAN, R, meta.answer, R });
}

fn renderFunctionTable(writer: anytype, functions: []types.FunctionEntry, total: u64) !void {
    try writer.print("{s}{s}Flat Function Ranking{s}\n", .{ CYAN, BOLD, R });
    try writer.print("{s}────────────────────────────────────────────────────────────{s}\n", .{ CYAN, R });
    try writer.print("{s}{s:>2}  {s:>5}  {s:>5}  {s:>5}  {s:>6}  {s:<48}{s}\n", .{ DIM, "#", "Samps", "    %", " Cum%", "Payoff", "Function", R });

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
        for (stars_s) |c| {
            if (payoff_len < 6) {
                payoff_buf[payoff_len] = c;
                payoff_len += 1;
            }
        }
        while (payoff_len < 6) : (payoff_len += 1) payoff_buf[payoff_len] = ' ';

        var fn_buf: [48]u8 = undefined;
        var fn_len: usize = 0;
        for (f.name) |c| {
            if (fn_len < 48) {
                fn_buf[fn_len] = c;
                fn_len += 1;
            }
        }
        while (fn_len < 48) : (fn_len += 1) fn_buf[fn_len] = ' ';

        try writer.print("{s}{d:>2}{s}  {d:>5}  {s}{s}{s}  {d:>5.1}  {s}{s}{s}  {s}{s}{s}\n", .{
            hc,                        idx, R,
            f.samples,                 hc,  pct_s,
            R,                         cum, hc,
            payoff_buf[0..payoff_len], R,   hc,
            fn_buf[0..fn_len],         R,
        });
        idx += 1;
    }
}

fn sourceEffectiveFunction(s: types.SourceLineEntry) []const u8 {
    const raw = if (s.parent_function.len > 0) s.parent_function else s.function;
    return shortFnName(raw);
}

fn renderSourceHotspots(writer: anytype, functions: []types.FunctionEntry, sources: []types.SourceLineEntry, meta: types.RunMetadata) !void {
    const total = meta.total_samples;
    if (total == 0) return;

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var llm_fns: std.ArrayList(*const types.FunctionEntry) = .empty;
    for (functions) |*f| {
        if (std.mem.eql(u8, f.image, "llm-linux")) try llm_fns.append(alloc, f);
    }
    std.mem.sortUnstable(*const types.FunctionEntry, llm_fns.items, {}, struct {
        fn lessThan(_: void, a: *const types.FunctionEntry, b: *const types.FunctionEntry) bool {
            return a.samples > b.samples;
        }
    }.lessThan);

    const duration_sec = parseIsoDuration(meta.start_time, meta.end_time);

    try writer.print("{s}{s}Source Hotspots{s}\n", .{ CYAN, BOLD, R });
    try writer.print("{s}────────────────────────────────────────────────────────────{s}\n", .{ CYAN, R });

    for (llm_fns.items) |f| {
        const pct = @as(f64, @floatFromInt(f.samples)) / @as(f64, @floatFromInt(total)) * 100.0;
        if (pct < 5.0) continue;

        const time_sec = @as(f64, @floatFromInt(f.samples)) / @as(f64, @floatFromInt(total)) * duration_sec;

        var time_buf: [8]u8 = undefined;
        const time_str = std.fmt.bufPrint(&time_buf, "{d:3.1}s", .{time_sec}) catch "0.0s";

        const pl = priorityLabel(pct);
        const hc = hotColor(pct);

        const sfn = shortFnName(f.name);

        try writer.print("  {s}{s}{s}{s}  [{d:5.1}%, {s}]  {s}{s}\n", .{ hc, BOLD, sfn, R, pct, time_str, DIM, pl });

        const SrcLine = struct { line: u32, file: []const u8, samples: u64 };
        var src_lines: std.ArrayList(SrcLine) = .empty;
        for (sources) |s| {
            const eff = sourceEffectiveFunction(s);
            if (!std.mem.eql(u8, eff, f.name)) continue;
            try src_lines.append(alloc, .{ .line = s.line, .file = s.file, .samples = s.samples });
        }

        std.mem.sortUnstable(SrcLine, src_lines.items, {}, struct {
            fn lessThan(_: void, a: SrcLine, b: SrcLine) bool {
                return a.samples > b.samples;
            }
        }.lessThan);

        const max_bar = 60;
        var shown: usize = 0;
        for (src_lines.items) |sl| {
            if (shown >= 8) break;
            shown += 1;

            const rel = blk: {
                var parts = std.mem.splitScalar(u8, sl.file, '/');
                var last: []const u8 = "";
                while (parts.next()) |p| last = p;
                break :blk last;
            };

            var bar_buf: [max_bar]u8 = undefined;
            const bar_n = if (src_lines.items[0].samples > 0)
                @min(@as(usize, @intCast(sl.samples * max_bar / src_lines.items[0].samples)), max_bar)
            else
                0;

            @memset(bar_buf[0..bar_n], '|');
            if (bar_n < max_bar) @memset(bar_buf[bar_n..max_bar], ' ');
            const bar = bar_buf[0..max_bar];

            var pbuf: [12]u8 = undefined;
            const pct_line = @as(f64, @floatFromInt(sl.samples)) / @as(f64, @floatFromInt(total)) * 100.0;
            const pct_s = std.fmt.bufPrint(&pbuf, "{d:5.1}%", .{pct_line}) catch "  ?.?%";

            try writer.print("    {s}{s}:{d:>4}{s}  {s:>6}  {s}{s}{s}\n", .{
                DIM,   rel, sl.line, R,
                pct_s, hc,  bar,     R,
            });
        }

        var total_shown: u64 = 0;
        for (src_lines.items[0..@min(src_lines.items.len, shown)]) |sl| total_shown += sl.samples;

        if (src_lines.items.len > shown) {
            try writer.print("    {s}... and {d} more lines{s}\n", .{ DIM, src_lines.items.len - shown, R });
        }
        try writer.print("    {s}{d} source line(s) in this function{s}\n", .{ DIM, src_lines.items.len, R });
    }
}

fn renderInsights(writer: anytype, functions: []types.FunctionEntry, sources: []types.SourceLineEntry, meta: types.RunMetadata) !void {
    const total = meta.total_samples;
    if (total == 0) return;

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var llm_fns: std.ArrayList(*const types.FunctionEntry) = .empty;
    for (functions) |*f| {
        if (std.mem.eql(u8, f.image, "llm-linux")) try llm_fns.append(alloc, f);
    }
    std.mem.sortUnstable(*const types.FunctionEntry, llm_fns.items, {}, struct {
        fn lessThan(_: void, a: *const types.FunctionEntry, b: *const types.FunctionEntry) bool {
            return a.samples > b.samples;
        }
    }.lessThan);

    const duration_sec = parseIsoDuration(meta.start_time, meta.end_time);

    try writer.print("{s}{s}Insights{s}\n", .{ CYAN, BOLD, R });
    try writer.print("{s}────────────────────────────────────────────────────────────{s}\n", .{ CYAN, R });

    const top_fn = if (llm_fns.items.len > 0) llm_fns.items[0] else return;
    const top_pct = @as(f64, @floatFromInt(top_fn.samples)) / @as(f64, @floatFromInt(total)) * 100.0;
    const top_sec = @as(f64, @floatFromInt(top_fn.samples)) / @as(f64, @floatFromInt(total)) * duration_sec;

    var tbuf: [16]u8 = undefined;
    const top_time = std.fmt.bufPrint(&tbuf, "{d:4.1}s", .{top_sec}) catch "?s";

    try writer.print("  {s}● {s}{s} ({d:.1}% / {s}) dominates — optimize first.{s}\n", .{
        RED_BOLD,
        BOLD,
        shortFnName(top_fn.name),
        top_pct,
        top_time,
        R,
    });

    var fn_src_lines: std.ArrayList(u64) = .empty;
    for (sources) |s| {
        const eff = sourceEffectiveFunction(s);
        if (!std.mem.eql(u8, eff, top_fn.name)) continue;
        try fn_src_lines.append(alloc, s.samples);
    }
    std.mem.sortUnstable(u64, fn_src_lines.items, {}, struct {
        fn lessThan(_: void, a: u64, b: u64) bool {
            return a > b;
        }
    }.lessThan);

    if (fn_src_lines.items.len > 0) {
        var top3: u64 = 0;
        var top5: u64 = 0;
        for (fn_src_lines.items[0..@min(fn_src_lines.items.len, 3)]) |v| top3 += v;
        if (fn_src_lines.items.len >= 5) {
            for (fn_src_lines.items[0..5]) |v| top5 += v;
        } else {
            top5 = top3;
        }

        const c3 = @as(f64, @floatFromInt(top3)) / @as(f64, @floatFromInt(total)) * 100.0;
        const c5 = @as(f64, @floatFromInt(top5)) / @as(f64, @floatFromInt(total)) * 100.0;

        try writer.print("    {s}Concentration: top 3 source lines = {d:.0}% of total; top 5 = {d:.0}%{s}\n", .{
            DIM, c3, c5, R,
        });
        try writer.print("    {s}Targeted micro-optimization of those lines yields highest ROI{s}\n", .{
            DIM, R,
        });
    }

    var p0_pct: f64 = 0;
    var p1_pct: f64 = 0;
    for (llm_fns.items) |f| {
        const p = @as(f64, @floatFromInt(f.samples)) / @as(f64, @floatFromInt(total)) * 100.0;
        if (p >= 10.0) p0_pct += p;
        if (p >= 5.0 and p < 10.0) p1_pct += p;
    }
    const block_pct = p0_pct + p1_pct;
    try writer.print("  {s}● {s}P0 + P1 functions account for {d:.0}% of total time — focus here{s}\n", .{
        YELLOW_BOLD, DIM, block_pct, R,
    });

    var src_line_hot_count: usize = 0;
    for (sources) |s| {
        if (s.samples >= 50) src_line_hot_count += 1;
    }
    try writer.print("  {s}● {s}{d} source lines have ≥50 samples — hot spots are concentrated{s}\n", .{
        YELLOW_BOLD, DIM, src_line_hot_count, R,
    });

    var light_lines: usize = 0;
    for (sources) |s| {
        if (s.samples < 10) light_lines += 1;
    }
    if (light_lines > 0) {
        try writer.print("  {s}● {s}{d} source lines have <10 samples — noise floor, ignore in optimization{s}\n", .{
            YELLOW_BOLD, DIM, light_lines, R,
        });
    }
}

fn renderSystemUtil(writer: anytype, samples: []types.SystemUtilSample, meta: types.RunMetadata) !void {
    if (samples.len == 0) {
        try writer.print("{s}(no timeline data){s}\n", .{ DIM, R });
        return;
    }

    const dur = parseIsoDuration(meta.start_time, meta.end_time);

    try writer.print("{s}{s}System Utilization Timeline{s}\n", .{ CYAN, BOLD, R });
    try writer.print("{s}────────────────────────────────────────────────────────────{s}\n", .{ CYAN, R });

    const last = samples[samples.len - 1];

    var bar: [40]u8 = undefined;
    const cpu_bars = @min(@as(usize, @intFromFloat(last.cpu_total_percent)) * 40 / 100, 40);
    @memset(bar[0..cpu_bars], '|');
    if (cpu_bars < 40) @memset(bar[cpu_bars..], ' ');
    var mbuf: [16]u8 = undefined;
    const mem_s = std.fmt.bufPrint(&mbuf, "{d:.0}MB", .{last.mem_used_mb}) catch "?MB";

    try writer.print("  {s}CPU:{s}      {d:5.1}%  {s}{s}{s}  {s}(pinned){s}\n", .{
        CYAN, R, last.cpu_total_percent, RED_BOLD, bar[0..cpu_bars], R, DIM, R,
    });
    try writer.print("  {s}Memory:{s}   {d:5.1}%  {s}{s}  {s}({s} / {d:.0}MB){s}\n", .{
        CYAN, R, last.mem_used_percent, YELLOW_BOLD, bar[0..@min(cpu_bars, 40)], R, mem_s, last.mem_total_mb, R,
    });
    try writer.print("  {s}I/O wait:{s}  {d:5.1}%  {s}no blocking{s}\n", .{
        CYAN, R, last.iowait_percent, DIM, R,
    });
    if (last.swap_used_kb > 0) {
        try writer.print("  {s}Swap:{s}     {d:.0}KB{s}\n", .{ CYAN, R, last.swap_used_kb, R });
    } else {
        try writer.print("  {s}Swap:{s}     {s}none{s}\n", .{ CYAN, R, DIM, R });
    }
    try writer.print("  {s}Procs:{s}    {} running, {} threads  {s}({d:.0} ctx/s){s}\n", .{
        CYAN, R, last.procs_running, last.threads_total, DIM, last.ctxt_per_s, R,
    });

    var disk_label: []const u8 = "~0";
    if (last.read_bps > 0 or last.write_bps > 0) {
        disk_label = "active";
    }
    try writer.print("  {s}Disk I/O:{s}  {s}  {s}(read during startup){s}\n", .{
        CYAN, R, disk_label, DIM, R,
    });

    try writer.writeAll("\n");

    try writer.print("{s}Timeline ({d} samples, {d:.0}s window){s}\n", .{
        DIM, samples.len, dur, R,
    });

    var ts: u64 = 0;
    for (samples) |s| {
        var cpu_bar: [16]u8 = undefined;
        const cpu_n = @min(@as(usize, @intCast(@as(u64, @intFromFloat(s.cpu0_percent)))) * 16 / 100, 16);
        @memset(cpu_bar[0..cpu_n], '#');
        if (cpu_n < 16) @memset(cpu_bar[cpu_n..], ' ');

        var pf_buf: [12]u8 = undefined;
        const pf_s = if (s.page_faults_per_s > 1000)
            (std.fmt.bufPrint(&pf_buf, "{d:.0}k", .{s.page_faults_per_s / 1000.0}) catch "?")
        else
            (std.fmt.bufPrint(&pf_buf, "{d:.0}", .{s.page_faults_per_s}) catch "?");

        const cpu_hc = if (s.cpu0_percent >= 99.0) RED_BOLD else DIM;

        try writer.print("  t={d:.0}s  CPU {s}{s}{s}  Mem {d:4.1}%  PF {s:>4}  IO {d:5.1}%\n", .{
            s.uptime_s,
            cpu_hc,
            cpu_bar[0..16],
            R,
            s.mem_used_percent,
            pf_s,
            s.iowait_percent,
        });

        ts += 1;
    }

    try writer.writeAll("\n");

    try writer.print("{s}{s}Insights{s}\n", .{ CYAN, BOLD, R });
    try writer.print("{s}────────────────────────────────────────────────────────────{s}\n", .{ CYAN, R });

    var max_pf: f64 = 0;
    var steady_min_pf: f64 = 999999;
    for (samples) |s| {
        if (s.page_faults_per_s > max_pf) max_pf = s.page_faults_per_s;
        if (s.uptime_s > 1 and s.page_faults_per_s < steady_min_pf) steady_min_pf = s.page_faults_per_s;
    }

    const is_cpu_bound = last.cpu_total_percent > 90;
    const has_startup = max_pf > steady_min_pf * 10;
    const has_io = last.iowait_percent > 5;
    const has_swap = last.swap_used_kb > 0;

    if (is_cpu_bound) {
        try writer.print("  {s}● {s}CPU-bound ({d:.0}%) throughout — no I/O bottlenecks{s}\n", .{
            RED_BOLD, DIM, last.cpu_total_percent, R,
        });
    }
    if (has_startup) {
        try writer.print("  {s}● {s}Startup phase: page faults peak at {d:.0}/s then settle to ~{d:.0}/s{s}\n", .{
            YELLOW_BOLD, DIM, max_pf, steady_min_pf, R,
        });
    }
    if (!has_io) {
        try writer.print("  {s}● {s}Zero I/O wait — model data fits in cache{s}\n", .{
            YELLOW_BOLD, DIM, R,
        });
    }
    if (!has_swap) {
        try writer.print("  {s}● {s}No swap usage — memory pressure is low{s}\n", .{
            YELLOW_BOLD, DIM, R,
        });
    }
    if (last.procs_running <= 2) {
        try writer.print("  {s}● {s}Single-threaded execution ({d} running process(es)){s}\n", .{
            YELLOW_BOLD, DIM, last.procs_running, R,
        });
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
    try writer.print("{s}{s}Optimization Priority{s}\n", .{ CYAN, BOLD, R });
    try writer.print("{s}────────────────────────────────────────────────────────────{s}\n", .{ CYAN, R });

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var pt: std.ArrayList(*const types.FunctionEntry) = .empty;
    for (functions) |*f| {
        if (std.mem.eql(u8, f.image, "llm-linux")) try pt.append(alloc, f);
    }
    std.mem.sortUnstable(*const types.FunctionEntry, pt.items, {}, struct {
        fn lessThan(_: void, a: *const types.FunctionEntry, b: *const types.FunctionEntry) bool {
            return a.samples > b.samples;
        }
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

    try writer.print("  {s}{s:>3}  {s:<48}  {s:>5}  {s:>4}  {s:<30}{s}\n", .{ DIM, "Pri", "Function", "Time", "Dpth", "Pri Caller", R });

    for (pt.items) |f| {
        const pct = if (meta.total_samples > 0) @as(f64, @floatFromInt(f.samples)) / @as(f64, @floatFromInt(meta.total_samples)) * 100.0 else 0.0;
        const time_sec = @as(f64, @floatFromInt(f.samples)) / @as(f64, @floatFromInt(meta.total_samples)) * duration_sec;
        const depth = uid_depth.get(f.uid) orelse 0;
        const caller = uid_caller.get(f.uid) orelse "—";
        var time_buf: [8]u8 = undefined;
        const time_str = std.fmt.bufPrint(&time_buf, "{d:3.1}s", .{time_sec}) catch "0.0s";

        var fn_buf: [48]u8 = undefined;
        var fn_len: usize = 0;
        for (f.name) |c| {
            if (fn_len < 48) {
                fn_buf[fn_len] = c;
                fn_len += 1;
            }
        }
        while (fn_len < 48) : (fn_len += 1) fn_buf[fn_len] = ' ';

        var cl_buf: [30]u8 = undefined;
        var cl_len: usize = 0;
        for (caller) |c| {
            if (cl_len < 30) {
                cl_buf[cl_len] = c;
                cl_len += 1;
            }
        }
        while (cl_len < 30) : (cl_len += 1) cl_buf[cl_len] = ' ';

        const pc = if (pct >= 1.0) RED_BOLD else DIM;
        const hc = hotColor(pct);

        try writer.print("  {s}{s:>3}{s}  {s}{s}{s}  {s:>5}  {d:>4}  {s}{s}{s}\n", .{
            pc,                priorityLabel(pct), R,
            hc,                fn_buf[0..fn_len],  R,
            time_str,          depth,              hc,
            cl_buf[0..cl_len], R,
        });
    }
}
