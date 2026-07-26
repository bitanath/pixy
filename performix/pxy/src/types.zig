const std = @import("std");

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
};

pub const ParsedData = struct {
    metadata: RunMetadata,
    functions: []FunctionEntry,
    callpath: []CallPathEntry,
    sources: []SourceLineEntry,
};
