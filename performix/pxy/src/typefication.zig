const std = @import("std");

pub const ReportType = enum {
    code_hotspots,
    system_utilization,
};

pub const RunMetadata = struct {
    workload: []const u8,
    target: []const u8,
    engine_version: []const u8,
    recipe: []const u8,
    start_time: []const u8,
    end_time: []const u8,
    total_samples: u64,
    answer: []const u8,
};

pub const FunctionEntry = struct {
    samples: u64,
    uid: u32,
    image: []const u8,
    name: []const u8,
};

pub const CallPathEntry = struct {
    total_samples: u64,
    self_samples: u64,
    uid: u32,
    image: []const u8,
    name: []const u8,
    depth: u32,
    node_id: u32,
    parent_node_id: u32,
};

pub const SourceLineEntry = struct {
    file: []const u8,
    line: u32,
    samples: u64,
    function: []const u8,
    parent_function: []const u8,
};

pub const SystemUtilSample = struct {
    uptime_s: f64,
    cpu_total_percent: f64,
    cpu0_percent: f64,
    iowait_percent: f64,
    mem_used_percent: f64,
    mem_used_mb: f64,
    mem_total_mb: f64,
    swap_used_kb: f64,
    procs_running: u64,
    threads_total: u64,
    ctxt_per_s: f64,
    page_faults_per_s: f64,
    pgmajfaults_per_s: f64,
    read_bps: f64,
    write_bps: f64,
};

pub const ParsedData = struct {
    report_type: ReportType,
    metadata: RunMetadata,
    functions: []FunctionEntry,
    callpath: []CallPathEntry,
    sources: []SourceLineEntry,
    system_samples: []SystemUtilSample,
};
