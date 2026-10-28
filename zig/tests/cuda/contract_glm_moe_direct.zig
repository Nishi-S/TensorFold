//! Layout-only MoE replacements: exact old outputs, guarded buffers, non-finite classes and ordered FP32 sums.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const contract = @import("core").contract;
const c = contract.spec;
const g = contract.gates;
const Rig = @import("contract_rig.zig").Rig;
fn depth(_: c.Shape) usize {
    return 9;
}
pub const Combine = struct {
    rig: Rig,
    set: glm.glue.Set,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    values: []f32 = &.{},
    weights: []f32 = &.{},
    p: [6]u64 = @splat(0),
    pub fn deinit(s: *Combine) void {
        s.rig.deinit();
        s.arena.deinit();
    }
    pub fn kernel(s: *Combine) g.Kernel {
        return .{ .spec = .{
            .name = "cuda glm direct shared combine",
            .accum = .f32,
            .depth = depth,
            .out = .f32,
            .replaces = "tf_glm_combine over routed slots followed by shared slot",
            .served = &.{ .{ .rows = 8, .n = 4096, .k = 9 }, .{ .rows = 65, .n = 1032, .k = 9 } },
        }, .ctx = s, .vt = &g.hooks(Combine) };
    }
    pub fn prepare(s: *Combine, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n % 8 != 0 or shape.k != 9) return false;
        s.rig.reset();
        _ = s.arena.reset(.retain_capacity);
        s.shape = shape;
        const a = s.arena.allocator();
        s.values = try a.alloc(f32, shape.rows * 9 * shape.n);
        s.weights = try a.alloc(f32, shape.rows * 9);
        const routed = try a.alloc(f32, shape.rows * 8 * shape.n);
        const shared = try a.alloc(f32, shape.rows * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        for (s.values) |*v| v.* = random.float(f32) * 4 - 2;
        for (s.weights) |*v| v.* = if (fill == .stress) 1 else random.float(f32);
        if (fill == .stress) for (0..shape.rows) |r| {
            for (0..shape.n) |col| {
                for (0..9) |k| s.values[(r * 9 + k) * shape.n + col] = switch (k) {
                    0 => 1048576,
                    1 => 1,
                    2 => -1048576,
                    else => 0,
                };
            }
        };
        if (fill == .nonfinite) {
            s.values[0] = std.math.nan(f32);
            s.values[9 * shape.n] = std.math.inf(f32);
            s.values[18 * shape.n] = -std.math.inf(f32);
        }
        for (0..shape.rows) |r| {
            @memcpy(routed[r * 8 * shape.n ..][0 .. 8 * shape.n], s.values[r * 9 * shape.n ..][0 .. 8 * shape.n]);
            @memcpy(shared[r * shape.n ..][0..shape.n], s.values[(r * 9 + 8) * shape.n ..][0..shape.n]);
        }
        s.p = .{ try s.rig.input(std.mem.sliceAsBytes(routed)), try s.rig.input(std.mem.sliceAsBytes(shared)), try s.rig.input(std.mem.sliceAsBytes(s.weights)), try s.rig.output(s.count() * 4), try s.rig.input(std.mem.sliceAsBytes(s.values)), try s.rig.scratch(s.count() * 4) };
        return true;
    }
    pub fn run(s: *Combine) !void {
        try glm.glue.combineShared(s.set, s.rig.s, s.p[0], s.p[1], s.p[2], s.p[3], s.shape.rows, s.shape.n, 8);
    }
    pub fn replaced(s: *Combine) ![]const u8 {
        try glm.glue.combine(s.set, s.rig.s, s.p[4], true, s.p[2], s.p[5], s.shape.rows, s.shape.n, 9);
        return s.rig.bytes(s.p[5], s.count() * 4);
    }
    pub fn count(s: *Combine) usize {
        return s.shape.rows * s.shape.n;
    }
    pub fn reference(s: *Combine, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            w.* = 0;
            m.* = 0;
            for (0..9) |k| {
                const v = @as(f64, s.values[((i / s.shape.n) * 9 + k) * s.shape.n + i % s.shape.n]) * s.weights[(i / s.shape.n) * 9 + k];
                w.* += v;
                m.* += @abs(v);
            }
        }
    }
    pub fn simulate(s: *Combine, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (out, 0..) |*o, i| {
            var terms: [9]f64 = undefined;
            for (&terms, 0..) |*v, k| v.* = @as(f64, s.values[((i / s.shape.n) * 9 + k) * s.shape.n + i % s.shape.n]) * s.weights[(i / s.shape.n) * 9 + k];
            o.* = contract.rows.sumIn(acc, order, &terms);
        }
    }
};

pub const Topk = struct {
    rig: Rig,
    set: glm.glue.Set,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    p: [6]u64 = @splat(0),
    pub fn deinit(s: *Topk) void {
        s.rig.deinit();
        s.arena.deinit();
    }
    pub fn kernel(s: *Topk) g.Kernel {
        return .{ .spec = .{
            .name = "cuda glm compact routed topk",
            .accum = .none,
            .depth = depth,
            .out = .raw,
            .nan = .{ .stated = "same sigmoid, fmax and lower-id tie rule as tf_glm_topk; identical weights" },
            .served = &.{ .{ .rows = 8, .n = 512, .k = 8 }, .{ .rows = 65, .n = 9, .k = 8 }, .{ .rows = 1862, .n = 512, .k = 8 } },
        }, .ctx = s, .vt = &g.hooks(Topk) };
    }
    pub fn prepare(s: *Topk, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n < 9 or shape.n > 512 or shape.k != 8) return false;
        s.rig.reset();
        _ = s.arena.reset(.retain_capacity);
        s.shape = shape;
        const a = s.arena.allocator();
        const logits = try a.alloc(f32, shape.rows * shape.n);
        const bias = try a.alloc(f32, shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        for (logits) |*v| v.* = random.float(f32) * 4 - 2;
        @memset(logits[0..shape.n], 0);
        @memset(bias, 0);
        if (fill == .nonfinite) {
            logits[0] = std.math.nan(f32);
            logits[shape.n] = std.math.inf(f32);
            logits[2 * shape.n] = -std.math.inf(f32);
        }
        s.p = .{ try s.rig.input(std.mem.sliceAsBytes(logits)), try s.rig.input(std.mem.sliceAsBytes(bias)), try s.rig.output(shape.rows * 8 * 4), try s.rig.output(shape.rows * 9 * 4), try s.rig.scratch(shape.rows * 9 * 4), try s.rig.scratch(shape.rows * 9 * 4) };
        return true;
    }
    pub fn run(s: *Topk) !void {
        try glm.glue.selectRouted(s.set, s.rig.s, s.p[0], s.p[1], s.p[2], s.p[3], s.shape.rows, 8, s.shape.n, 2.5, true);
    }
    pub fn count(s: *Topk) usize {
        return s.shape.rows * 17;
    }
    pub fn reference(_: *Topk, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }
    pub fn check(s: *Topk, _: g.Gate) !?[]const u8 {
        try glm.glue.select(s.set, s.rig.s, s.p[0], s.p[1], s.p[4], s.p[5], s.shape.rows, 8, s.shape.n, 2.5, true);
        const a = s.arena.allocator();
        const old = try a.dupe(u8, try s.rig.bytes(s.p[4], s.shape.rows * 9 * 4));
        const compact = try s.rig.bytes(s.p[2], s.shape.rows * 8 * 4);
        for (0..s.shape.rows) |row| if (!std.mem.eql(u8, compact[row * 32 ..][0..32], old[row * 36 ..][0..32])) return "compact route differs from original selection";
        const weights = try a.dupe(u8, try s.rig.bytes(s.p[5], s.shape.rows * 9 * 4));
        if (!std.mem.eql(u8, weights, try s.rig.bytes(s.p[3], s.shape.rows * 9 * 4))) return "compact route changed weights";
        return null;
    }
};
