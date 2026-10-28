//! HC decode replacement and prompt MMA: independent FP64 dots, norms, guarded seams and low-precision controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const rows_ref = core.contract.rows;
const Dense = core.contract.dense.Dense;

fn depth(_: c.Shape) usize {
    return 1032;
}

pub const Case = struct {
    rig: rig_.Rig,
    candidate: cuda.Function,
    baseline: cuda.Function,
    mma: bool,
    drop_tail: bool = false,
    d: ?Dense = null,
    rows: usize = 0,
    x: u64 = 0,
    w: u64 = 0,
    out: u64 = 0,
    old: u64 = 0,
    compact: std.ArrayList(u8) = .empty,
    old_compact: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *Case) void {
        if (self.d) |d| d.deinit();
        self.compact.deinit(self.rig.gpu.gpa);
        self.old_compact.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    pub fn kernel(self: *Case) g.Kernel {
        return .{ .ctx = self, .vt = &vt, .control = if (self.drop_tail) .shapes else null, .spec = .{
            .name = if (self.drop_tail) "control: HC fast drops final row" else if (self.mma) "cuda GLM HC prompt MMA" else "cuda GLM HC decode partitioned",
            .accum = .f32,
            .depth = depth,
            .out = .f32,
            .replaces = if (self.mma) null else "tf_glm_hc_partial",
            .tiles = .{ .{ .size = if (self.mma) 64 else 4 }, .{ .size = 24, .step = 24, .min = 24 }, .{ .size = 1024, .step = 1024, .min = 1024 } },
            .served = &.{ .{ .rows = 1, .n = 24, .k = 1024 }, .{ .rows = 5, .n = 24, .k = 1024 }, .{ .rows = 63, .n = 24, .k = 1024 }, .{ .rows = 65, .n = 24, .k = 1024 }, .{ .rows = 129, .n = 24, .k = 1024 } },
        } };
    }
    const vt = blk: {
        var hooks = g.hooks(Case);
        hooks.output = output;
        break :blk hooks;
    };

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 129 or shape.n != 24 or shape.k != 1024) return false;
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        self.rows = shape.rows;
        const d = try Dense.init(self.rig.gpu.gpa, .{ .rows = shape.rows * 16, .n = 24, .k = 1024 }, 1);
        self.d = d;
        d.fillX(fill, seed, 2);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x4843);
        const rng = prng.random();
        for (0..24) |m| for (0..1024) |k| {
            const v: f32 = if (fill == .stress) @as(f32, @floatFromInt(1 + ((k % 512) * 7 + m) % 6)) * (if (k < 512) @as(f32, 1) else -1) else rng.float(f32) * 2 - 1;
            d.w[m * 1024 + k] = c.fromBf16(c.toBf16(v));
        };
        const weights = try self.rig.gpu.gpa.alloc(u16, 24 * 16384);
        defer self.rig.gpu.gpa.free(weights);
        for (0..24) |m| for (0..16) |b| for (0..1024) |k| {
            weights[m * 16384 + b * 1024 + k] = c.toBf16(@floatCast(d.w[m * 1024 + k]));
        };
        self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        self.w = try self.rig.input(std.mem.sliceAsBytes(weights));
        self.out = try self.rig.output(self.rows * 16 * 32 * 4);
        self.old = try self.rig.scratch(self.rows * 16 * 32 * 4);
        return true;
    }

    fn launch(self: *Case, f: cuda.Function, out: u64, mma: bool, legacy: bool, count_: usize) !void {
        var a: cuda.Args = .{};
        inline for (.{ self.x, self.w, out, @as(i64, 16384), @as(c_int, 16), @as(i64, @intCast(count_)) }) |v| a.add(v);
        const tile: usize = if (mma) 64 else 4;
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((count_ + tile - 1) / tile), .y = 16, .z = if (mma or legacy) 1 else 3 }, .block = .{ .x = if (mma or legacy) 128 else 288 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn run(self: *Case) !void {
        const n = self.rows - @as(usize, @intFromBool(self.drop_tail));
        if (n != 0) try self.launch(self.candidate, self.out, self.mma, false, n);
    }

    fn compactBytes(self: *Case, ptr: u64) ![]const u8 {
        const src = try self.rig.bytes(ptr, self.rows * 16 * 32 * 4);
        const dst = if (ptr == self.old) &self.old_compact else &self.compact;
        try dst.resize(self.rig.gpu.gpa, self.count() * 4);
        for (0..self.rows * 16) |r| @memcpy(dst.items[r * 25 * 4 ..][0 .. 25 * 4], src[r * 32 * 4 ..][0 .. 25 * 4]);
        return dst.items;
    }

    fn output(ctx: *anyopaque) ![]const u8 {
        const self: *Case = @ptrCast(@alignCast(ctx));
        return self.compactBytes(self.out);
    }

    pub fn replaced(self: *Case) ![]const u8 {
        try self.launch(self.baseline, self.old, false, true, self.rows);
        return self.compactBytes(self.old);
    }

    pub fn count(self: *Case) usize {
        return self.rows * 16 * 25;
    }

    fn terms(self: *Case, row: usize, col: usize, out: *[1024]f64) void {
        const d = self.d.?;
        for (out, 0..) |*v, k| {
            const x = c.fromBf16(d.x[row * 1024 + k]);
            v.* = x * (if (col == 24) x else d.w[col * 1024 + k]);
        }
    }

    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        var terms_: [1024]f64 = undefined;
        for (0..self.rows * 16) |r| for (0..25) |col| {
            self.terms(r, col, &terms_);
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (terms_) |v| {
                sum += v;
                abs += @abs(v);
            }
            want[r * 25 + col] = sum;
            mag[r * 25 + col] = abs;
        };
    }

    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        var terms_: [1024]f64 = undefined;
        for (0..self.rows * 16) |r| for (0..25) |col| {
            self.terms(r, col, &terms_);
            out[r * 25 + col] = rows_ref.sumIn(acc, order, &terms_);
        };
    }
};

pub const Group = struct {
    module: ?cuda.Module = null,
    cases: [4]Case = undefined,
    ready: bool = false,

    pub fn deinit(self: *Group) void {
        if (self.ready) for (&self.cases) |*item| item.deinit();
        if (self.module) |*m| m.unload();
    }

    pub fn add(self: *Group, gpu: @import("check.zig").Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !void {
        self.module = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
        const mma = try self.module.?.function("tf_glm_hc_partial_mma");
        const dec = try self.module.?.function("tf_glm_hc_partial_dec");
        const baseline = try self.module.?.function("tf_glm_hc_partial");
        for (&self.cases, 0..) |*item, i| item.* = .{ .rig = .{ .gpu = gpu, .s = s }, .candidate = if (i % 2 == 0) dec else mma, .baseline = baseline, .mma = i % 2 != 0, .drop_tail = i >= 2 };
        self.ready = true;
        for (&self.cases) |*item| try kernels.append(gpu.gpa, item.kernel());
    }
};
