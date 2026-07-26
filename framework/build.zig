const std = @import("std");

pub fn build(b: *std.Build) void {
    b.resolveInstallPrefix("outputs", .{});

    const debug_logs = b.option(bool, "debug-logs", "Enable debug output") orelse false;

    const options = b.addOptions();
    options.addOption(bool, "debugLogs", debug_logs);

    // macOS target (aarch64)
    const macos_target = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .macos });

    const macos_lib_mod = b.createModule(.{
        .root_source_file = b.path("core/index.zig"),
        .target = macos_target,
        .optimize = .ReleaseFast,
    });
    macos_lib_mod.addOptions("build_opts", options);

    const macos_exe_mod = b.createModule(.{
        .root_source_file = b.path("core/main.zig"),
        .target = macos_target,
        .optimize = .ReleaseFast,
    });
    macos_exe_mod.addOptions("build_opts", options);

    const macos_exe = b.addExecutable(.{ .name = "llm-macos", .root_module = macos_exe_mod });
    b.installArtifact(macos_exe);

    const macos_lib = b.addLibrary(.{ .name = "llm-macos", .root_module = macos_lib_mod, .linkage = .static });
    const install_macos_lib = b.addInstallArtifact(macos_lib, .{ .dest_sub_path = "llm-macos.a" });
    b.getInstallStep().dependOn(&install_macos_lib.step);

    // Linux target (aarch64, musl for fully static binary)
    const linux_target = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .linux, .abi = .musl });

    const linux_lib_mod = b.createModule(.{
        .root_source_file = b.path("core/index.zig"),
        .target = linux_target,
        .optimize = .ReleaseFast,
    });
    linux_lib_mod.addOptions("build_opts", options);
    linux_lib_mod.link_libc = true;

    const linux_exe_mod = b.createModule(.{
        .root_source_file = b.path("core/main.zig"),
        .target = linux_target,
        .optimize = .ReleaseFast,
    });
    linux_exe_mod.addOptions("build_opts", options);
    linux_exe_mod.link_libc = true;

    const linux_exe = b.addExecutable(.{ .name = "llm-linux", .root_module = linux_exe_mod });
    b.installArtifact(linux_exe);

    const linux_lib = b.addLibrary(.{ .name = "llm-linux", .root_module = linux_lib_mod, .linkage = .static });
    const install_linux_lib = b.addInstallArtifact(linux_lib, .{ .dest_sub_path = "llm-linux.a" });
    b.getInstallStep().dependOn(&install_linux_lib.step);

    // watchOS target (unchanged)
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
    watchos_mod.addOptions("build_opts", options);
    const watchos_lib = b.addLibrary(.{ .name = "llm", .root_module = watchos_mod, .linkage = .static });
    const install_watchos_lib = b.addInstallArtifact(watchos_lib, .{ .dest_sub_path = "llm-watchos.a" });
    b.getInstallStep().dependOn(&install_watchos_lib.step);
}
