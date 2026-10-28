//! A side stream warms bounded immutable weight prefixes during decode all-gathers; prompts and capture skip it.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
pub const policy = @import("cuda_l2_policy.zig");
const Range = struct { ptr: u64 = 0, bytes: usize = 0 };

pub const Prefetch = struct {
    settings: policy.Settings,
    module: cuda.Module,
    kernel: cuda.Function,
    side: cuda.Stream,
    fork: cuda.Event,
    joined: cuda.Event,
    allocator: ?std.mem.Allocator = null,
    sites: []Range = &.{},

    pub fn init(d: *const cuda.Driver, settings: policy.Settings) !Prefetch {
        if (settings.mode == .disabled or settings.bytes == 0 or settings.bytes > 64 << 20 or
            settings.rows == 0 or settings.rows > 64) return error.BadGlmL2Prefetch;
        var module = try cuda.Module.load(d, cuda.kernels.glm_l2_prefetch);
        errdefer module.unload();
        const kernel = try module.function(switch (settings.mode) {
            .bulk => "tf_glm_l2_prefetch_bulk",
            .lines => "tf_glm_l2_prefetch",
            .touch => "tf_glm_l2_prefetch_touch",
            .disabled => unreachable,
        });
        var side = try cuda.Stream.init(d, true);
        errdefer side.deinit();
        var fork = try cuda.Event.init(d, false);
        errdefer fork.deinit();
        const joined = try cuda.Event.init(d, false);
        return .{ .settings = settings, .module = module, .kernel = kernel, .side = side, .fork = fork, .joined = joined };
    }

    fn range(w: *const weights.Weights, ptr: u64, budget: usize) Range {
        for (w.buffers.items) |b| {
            const n = policy.extent(b.ptr, b.len, ptr, budget);
            if (n > 0) return .{ .ptr = ptr, .bytes = n };
        }
        return .{};
    }

    /// Called once after the loader compacts weights; the table borrows only allocation-bounded immutable pointers.
    pub fn prepare(p: *Prefetch, a: std.mem.Allocator, w: *const weights.Weights) !void {
        if (p.allocator != null) return error.GlmL2AlreadyPrepared;
        const sites = try a.alloc(Range, w.layers.len);
        for (sites, 0..) |*entry, i| {
            const ptr: u64 = if (i + 1 < w.layers.len) switch (w.layers[i + 1].mixer) {
                .kda => |k| switch (k.proj) {
                    .bf16 => |v| v,
                    .split => |v| v.rest,
                },
                .mla => |m| switch (m.proj) {
                    .bf16 => |v| v,
                    else => 0,
                },
            } else switch (w.head) {
                .bf16 => |v| v,
                else => 0,
            };
            entry.* = range(w, ptr, p.settings.bytes);
        }
        p.allocator = a;
        p.sites = sites;
    }

    pub fn active(p: *const Prefetch, main: cuda.Stream, rows: usize, prompt: bool) !bool {
        if (!policy.eligible(p.settings, rows, prompt, false) or p.sites.len == 0) return false;
        return try cuda.graph.captureStatus(main) == .none;
    }

    /// One range per layer, forked after FFN and before its all-gather; no host synchronization or mutable scratch.
    pub fn site(p: *const Prefetch, main: cuda.Stream, layer: usize) !void {
        const r = p.sites[layer];
        if (r.bytes == 0) return;
        try p.fork.record(main);
        try p.side.wait(p.fork);
        var args: cuda.Args = .{};
        args.add(r.ptr);
        args.add(@as(u64, r.bytes));
        const lines = (r.bytes + 127) / 128;
        try cuda.launch.launch(p.kernel, .{ .grid = .{ .x = @intCast(@min(32, (lines + 127) / 128)) }, .block = .{ .x = 128 } }, p.side, &args);
    }

    /// Async join makes the caller's normal stream drain cover every borrowed weight read.
    pub fn finish(p: *const Prefetch, main: cuda.Stream) !void {
        try p.joined.record(p.side);
        try main.wait(p.joined);
    }

    /// Errors and weight teardown drain the side stream before any borrowed allocation can be released.
    pub fn drain(p: *const Prefetch) void {
        cuda.completion.drain(p.side.d, p.side, "GLM L2 prefetch stream did not drain");
    }

    pub fn deinit(p: *Prefetch) void {
        p.drain();
        p.side.deinit();
        p.fork.deinit();
        p.joined.deinit();
        p.module.unload();
        if (p.allocator) |a| a.free(p.sites);
        p.* = undefined;
    }
};
