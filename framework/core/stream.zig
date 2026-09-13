const std = @import("std");
const c = @import("constants.zig");
const genny = @import("genny.zig");

// Session-based streaming generation. Additive API: the whole-answer
// exports in index.zig are untouched. One live stream per process: start()
// claims a global slot and returns null while another stream runs, so callers
// fall back to whole-answer generation instead of interleaving tokens.

// Minimal spin mutex (std.Thread has no Mutex in 0.16). Hold times are
// microseconds (buffer append / bounded copy), contention negligible.
const SpinLock = struct {
    locked: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    fn lock(s: *SpinLock) void {
        while (s.locked.cmpxchgStrong(false, true, .acquire, .monotonic) != null) {
            std.atomic.spinLoopHint();
        }
    }

    fn unlock(s: *SpinLock) void {
        s.locked.store(false, .release);
    }
};
pub const Session = struct {
    thread: std.Thread,
    mutex: SpinLock = .{},
    arena: std.heap.ArenaAllocator,
    buffer: std.array_list.AlignedManaged(u8, null),
    read_off: usize = 0,
    done: bool = false,
    failed: bool = false,
    cancel: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    // Params owned by arena.
    messages: []c.ChatMessage = &.{},
    system: []const u8 = "",
    max_tokens: i32 = 0,
    temperature: f64 = 0.3,
    context_size: usize = 0,
    model_data: []const u8 = &.{},

    fn push(s: *Session, bytes: []const u8) void {
        s.mutex.lock();
        defer s.mutex.unlock();
        s.buffer.appendSlice(bytes) catch {
            s.failed = true;
        };
    }
};

// Global claim slot (single in-flight stream). Stored as usize so the
// claim is one atomic op: 0 means free.
var active_addr: std.atomic.Value(usize) = std.atomic.Value(usize).init(0);

fn renderCb(token: []const u8) void {
    const addr = active_addr.load(.acquire);
    if (addr == 0) return;
    const s: *Session = @ptrFromInt(addr);
    s.push(token);
}

fn run(s: *Session) void {
    defer {
        s.mutex.lock();
        s.done = true;
        s.mutex.unlock();
        _ = active_addr.cmpxchgStrong(@intFromPtr(s), 0, .release, .monotonic);
    }

    var ctx = c.Context{};
    _ = genny.conversation(&ctx, .{
        .type = "load",
        .system_prompt = s.system,
        .max_tokens = s.max_tokens,
        .context_size = s.context_size,
        .temperature = s.temperature,
    }, s.model_data, null, &s.cancel, s.arena.allocator()) catch {
        s.mutex.lock();
        s.failed = true;
        s.mutex.unlock();
        return;
    };

    _ = genny.conversation(&ctx, .{
        .type = "generate",
        .chat_history = s.messages,
        .cb_render = renderCb,
    }, s.model_data, null, &s.cancel, s.arena.allocator()) catch {
        s.mutex.lock();
        s.failed = true;
        s.mutex.unlock();
        return;
    };
}

/// Starts a streaming generation on a worker thread. Returns an opaque
/// handle, or null when a stream is already running or spawn fails.
/// The caller must pass the handle to stream_poll() until done, then call
/// stream_free() exactly once. Strings are copied; caller memory may be
/// released as soon as this returns.
pub fn streamStart(
    messages: [*]const c.ChatMessageC,
    num_messages: usize,
    system_prompt: [*:0]const u8,
    max_tokens: i32,
    temperature: f64,
    context_size: usize,
    model_data: []const u8,
) ?*anyopaque {
    return startInner(messages, num_messages, system_prompt, max_tokens, temperature, context_size, model_data) catch null;
}

fn startInner(
    messages: [*]const c.ChatMessageC,
    num_messages: usize,
    system_prompt: [*:0]const u8,
    max_tokens: i32,
    temperature: f64,
    context_size: usize,
    model_data: []const u8,
) !*anyopaque {
    const allocator = std.heap.c_allocator;
    const s = allocator.create(Session) catch return error.OutOfMemory;
    s.* = Session{
        .thread = undefined,
        .arena = std.heap.ArenaAllocator.init(std.heap.page_allocator),
        .buffer = std.array_list.AlignedManaged(u8, null).init(std.heap.page_allocator),
    };
    errdefer {
        s.buffer.deinit();
        s.arena.deinit();
        allocator.destroy(s);
    }
    const a = s.arena.allocator();

    const sys = std.mem.sliceTo(system_prompt, 0);
    s.system = a.dupe(u8, sys) catch return error.OutOfMemory;
    s.max_tokens = max_tokens;
    s.temperature = temperature;
    s.context_size = context_size;
    s.model_data = model_data;

    const msgs = a.alloc(c.ChatMessage, num_messages) catch return error.OutOfMemory;
    for (0..num_messages) |i| {
        const role = std.mem.sliceTo(messages[i].role, 0);
        const content = std.mem.sliceTo(messages[i].content, 0);
        msgs[i] = .{
            .role = a.dupe(u8, role) catch return error.OutOfMemory,
            .content = a.dupe(u8, content) catch return error.OutOfMemory,
        };
    }
    s.messages = msgs;

    // Claim the single-flight slot before spawning.
    const prev = active_addr.cmpxchgStrong(0, @intFromPtr(s), .acquire, .monotonic);
    if (prev != null) return error.StreamBusy;

    s.thread = std.Thread.spawn(.{}, run, .{s}) catch |err| {
        _ = active_addr.cmpxchgStrong(@intFromPtr(s), 0, .release, .monotonic);
        return err;
    };
    return @ptrCast(s);
}

/// Copies newly generated bytes into out (up to cap). Returns the byte
/// count, 0 when nothing is new yet, -1 when generation finished AND all
/// bytes were drained, -2 on engine failure or a null handle.
pub fn streamPoll(handle: ?*anyopaque, out: [*]u8, cap: usize) isize {
    const s: *Session = @ptrCast(@alignCast(handle orelse return -2));
    s.mutex.lock();
    defer s.mutex.unlock();
    if (s.failed) return -2;
    const avail = s.buffer.items.len - s.read_off;
    if (avail == 0) {
        return if (s.done) -1 else 0;
    }
    const n = @min(avail, cap);
    @memcpy(out[0..n], s.buffer.items[s.read_off..][0..n]);
    s.read_off += n;
    if (s.done and s.read_off == s.buffer.items.len) return -1;
    return @intCast(n);
}

/// Signals cancellation, waits for the worker, then frees the session.
/// Safe to call at any point after a non-null stream_start().
pub fn streamFree(handle: ?*anyopaque) void {
    const s: *Session = @ptrCast(@alignCast(handle orelse return));
    s.cancel.store(true, .unordered);
    s.thread.join();
    s.buffer.deinit();
    s.arena.deinit();
    std.heap.c_allocator.destroy(s);
}
