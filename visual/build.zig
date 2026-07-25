const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});

    b.resolveInstallPrefix("outputs", .{});

    const lib_mod = b.createModule(.{
        .root_source_file = b.path("core/index.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("core/main.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });

    const exe = b.addExecutable(.{ .name = "visual", .root_module = exe_mod });
    b.installArtifact(exe);

    const lib = b.addLibrary(.{ .name = "visual", .root_module = lib_mod, .linkage = .static });
    const install_lib = b.addInstallArtifact(lib, .{ .dest_sub_path = "visual.a" });
    b.getInstallStep().dependOn(&install_lib.step);

    const run_cmd = b.addRunArtifact(exe);
    const run_step = b.step("run", "Run the convnet");
    run_step.dependOn(&run_cmd.step);

    const watchos_type = b.option(
        []const u8,
        "watchos",
        "watchOS target: 'sim' or 'device' (default: 'sim')",
    ) orelse "sim";

    const watchos_tq: std.Target.Query = if (std.mem.eql(u8, watchos_type, "device"))
        .{ .cpu_arch = .aarch64, .os_tag = .watchos }
    else
        .{ .cpu_arch = .aarch64, .os_tag = .watchos, .abi = .simulator };

    const watchos_target = b.resolveTargetQuery(watchos_tq);
    const watchos_mod = b.createModule(.{
        .root_source_file = b.path("core/index.zig"),
        .target = watchos_target,
        .optimize = .ReleaseFast,
    });
    const watchos_lib = b.addLibrary(.{ .name = "visual", .root_module = watchos_mod, .linkage = .static });
    const install_watchos_lib = b.addInstallArtifact(watchos_lib, .{ .dest_sub_path = "visual-watchos.a" });
    b.getInstallStep().dependOn(&install_watchos_lib.step);
}
