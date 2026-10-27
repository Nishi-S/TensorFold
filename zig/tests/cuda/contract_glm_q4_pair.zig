//! Packed Q4 BF16 pairs: original product bytes and independent dense sums, including strided inputs and tails.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const Dense = core.contract.dense.Dense;

fn depth(s: c.Shape) usize {
    return s.k + 8;
}

pub const Case = struct {
    rig: rig_.Rig,
    k: *const glm.mm.Kernels,
    wide: bool,
    drop_tail: bool = false,
    d: ?Dense = null,
    x: u64 = 0,
    candidate_x: u64 = 0,
    w: glm.mm.Tiled = undefined,
    stride: usize = 0,
    out: u64 = 0,
    base: u64 = 0,
    part: u64 = 0,

    pub fn deinit(self: *Case) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Case) g.Kernel {
        return .{ .ctx = self, .vt = &vt, .control = if (self.drop_tail) .shapes else null, .spec = .{
            .name = if (self.drop_tail) "control: GLM Q4 pair drops final input" else if (self.wide) "cuda GLM Q4 pair FP32" else "cuda GLM Q4 pair BF16",
            .accum = .f32,
            .depth = depth,
            .out = if (self.wide) .f32 else .bf16,
            .replaces = "glm_mm4_16",
            .tiles = .{ .{ .size = 16 }, .{ .size = 64, .step = 8, .min = 8 }, .{ .size = 256, .step = 64, .min = 64 } },
            .served = &.{
                .{ .rows = 1, .n = 4096, .k = 2048 },  .{ .rows = 3, .n = 4096, .k = 1024 },
                .{ .rows = 8, .n = 72, .k = 512 },     .{ .rows = 16, .n = 128, .k = 2048 },
                .{ .rows = 1, .n = 12288, .k = 4096 }, .{ .rows = 3, .n = 4096, .k = 6144 },
            },
        } };
    }
    const vt = g.hooks(Case);

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 16 or shape.n == 0 or shape.n % 8 != 0 or
            shape.k == 0 or shape.k % 64 != 0) return false;
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 2);
        const words = try a.alloc(u32, shape.n * shape.k / 8);
        defer a.free(words);
        @memset(words, 0);
        const scales = try a.alloc(u16, shape.n * shape.k / 64);
        defer a.free(scales);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x4);
        const rng = prng.random();
        for (0..shape.n) |col| {
            for (0..shape.k / 64) |group| {
                const scale: f16 = if (fill == .stress) 1 else @floatCast(1.0 / @as(f32, @floatFromInt(@as(u32, 1) << @as(u5, @intCast((col + group) % 4)))));
                scales[col * (shape.k / 64) + group] = @bitCast(scale);
                for (0..64) |j| {
                    const at = group * 64 + j;
                    const magnitude: i32 = @intCast(1 + ((at % (shape.k / 2)) * 7 + col) % 6);
                    const code: i32 = if (fill == .stress) (if (at < shape.k / 2) magnitude else -magnitude) else @as(i32, rng.uintLessThan(u8, 16)) - 8;
                    d.w[col * shape.k + at] = @as(f64, @floatFromInt(code)) * @as(f64, scale);
                    words[(col * shape.k + at) / 8] |= @as(u32, @intCast(code + 8)) << @as(u5, @intCast(4 * (at % 8)));
                }
            }
        }
        self.stride = shape.k + 8;
        const x = try a.alloc(u16, shape.rows * self.stride);
        defer a.free(x);
        @memset(x, 0x7fc0);
        for (0..shape.rows) |r| @memcpy(x[r * self.stride ..][0..shape.k], d.x[r * shape.k ..][0..shape.k]);
        self.x = try self.rig.input(std.mem.sliceAsBytes(x));
        self.candidate_x = self.x;
        if (self.drop_tail) {
            for (0..shape.rows) |r| x[r * self.stride + shape.k - 1] = 0;
            self.candidate_x = try self.rig.input(std.mem.sliceAsBytes(x));
        }
        const src = try self.rig.input(std.mem.sliceAsBytes(words));
        const ss = try self.rig.input(std.mem.sliceAsBytes(scales));
        const tiled = try self.rig.scratch(words.len * 4);
        const ts = try self.rig.scratch(scales.len * 2);
        try glm.mm.tilePacked(self.k, self.rig.s, 4, src, ss, tiled, ts, shape.n, shape.k);
        const wt = try self.rig.input(try self.rig.bytes(tiled, words.len * 4));
        const st = try self.rig.input(try self.rig.bytes(ts, scales.len * 2));
        self.w = .{ .w = wt, .s = st, .n = shape.n, .k = shape.k, .bits = 4 };
        self.part = try self.rig.scratch(@max(16, glm.mm.partBytes(shape.rows, self.w)));
        self.out = try self.rig.output(self.count() * self.bytes());
        self.base = try self.rig.scratch(self.count() * self.bytes());
        return true;
    }

    fn bytes(self: *Case) usize {
        return if (self.wide) 4 else 2;
    }

    pub fn run(self: *Case) !void {
        try glm.mm.matmul(self.k, self.rig.s, self.candidate_x, self.stride, self.w, self.out, self.part, self.d.?.shape.rows, self.wide, false);
        try self.rig.s.synchronize();
    }

    pub fn replaced(self: *Case) ![]const u8 {
        var baseline = self.k.*;
        baseline.q4_pair = null;
        try glm.mm.matmul(&baseline, self.rig.s, self.x, self.stride, self.w, self.base, self.part, self.d.?.shape.rows, self.wide, false);
        return self.rig.bytes(self.base, self.count() * self.bytes());
    }

    pub fn count(self: *Case) usize {
        return self.d.?.outputs();
    }
    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        try self.d.?.reference(want, mag);
    }
    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.d.?.simulate(acc, order, out);
    }
};
