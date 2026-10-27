//! Weight-only FP8: independent FP64 dense formula, nonfinite classes, guards and accumulation stress.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const Dense = core.contract.dense.Dense;
const codec = core.exlr8.codec;

fn depth(s: c.Shape) usize {
    return s.k + s.k / 64 + 8;
}

pub const Case = struct {
    rig: rig_.Rig,
    kernels: *const glm.mm.Kernels,
    wide: bool,
    drop_tail: bool = false,
    d: ?Dense = null,
    x: u64 = 0,
    weight: glm.mm.Tiled = undefined,
    stride: usize = 0,
    out: u64 = 0,
    part: u64 = 0,

    pub fn deinit(self: *Case) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Case) g.Kernel {
        return .{ .ctx = self, .vt = &vt, .control = if (self.drop_tail) .shapes else null, .spec = .{
            .name = if (self.drop_tail) "control: GLM FP8 drops final input" else if (self.wide) "cuda GLM weight-only FP8 FP32" else "cuda GLM weight-only FP8 BF16",
            .accum = .f32,
            .depth = depth,
            .out = if (self.wide) .f32 else .bf16,
            .tiles = .{ .{ .size = 128 }, .{ .size = 64, .step = 8, .min = 8 }, .{ .size = 128, .step = 128, .min = 128 } },
            .served = &.{
                .{ .rows = 1, .n = 4096, .k = 4096 },  .{ .rows = 3, .n = 4096, .k = 4096 },
                .{ .rows = 16, .n = 72, .k = 512 },    .{ .rows = 32, .n = 128, .k = 2048 },
                .{ .rows = 64, .n = 128, .k = 128 },   .{ .rows = 128, .n = 128, .k = 256 },
                .{ .rows = 1, .n = 77440, .k = 4096 },
            },
        } };
    }
    const vt = g.hooks(Case);

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 8 != 0 or shape.k == 0 or shape.k % 128 != 0) return false;
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 2);
        const codes = try a.alloc(u8, shape.n * shape.k);
        defer a.free(codes);
        const scales = try a.alloc(f32, shape.n * shape.k / 128);
        defer a.free(scales);
        var prng = std.Random.DefaultPrng.init(seed ^ 0xf8);
        const rng = prng.random();
        for (0..shape.n) |col| {
            for (0..shape.k / 128) |group| {
                const scale: f32 = if (fill == .stress) 1 else if ((col + group) % 2 == 0) 0.125 else 0.25;
                scales[col * (shape.k / 128) + group] = scale;
                for (0..128) |j| {
                    const at = group * 128 + j;
                    const magnitude: i32 = @intCast(1 + ((at % (shape.k / 2)) * 7 + col) % 6);
                    const integer: i32 = if (fill == .stress) (if (at < shape.k / 2) magnitude else -magnitude) else @as(i32, rng.uintLessThan(u8, 16)) - 8;
                    const code = codec.e4m3FromF32(@floatFromInt(integer));
                    codes[col * shape.k + at] = code;
                    d.w[col * shape.k + at] = @as(f64, codec.e4m3Value(code)) * scale;
                }
            }
        }
        self.stride = shape.k + 8;
        const x = try a.alloc(u16, shape.rows * self.stride);
        defer a.free(x);
        @memset(x, 0x7fc0);
        for (0..shape.rows) |r| @memcpy(x[r * self.stride ..][0..shape.k], d.x[r * shape.k ..][0..shape.k]);
        if (self.drop_tail) for (0..shape.rows) |r| {
            x[r * self.stride + shape.k - 1] = 0;
        };
        self.x = try self.rig.input(std.mem.sliceAsBytes(x));
        self.weight = .{
            .w = try self.rig.input(codes),
            .s = try self.rig.input(std.mem.sliceAsBytes(scales)),
            .n = shape.n,
            .k = shape.k,
            .bits = 8,
            .format = .fp8,
        };
        self.part = try self.rig.scratch(@max(16, glm.mm.partBytes(shape.rows, self.weight)));
        self.out = try self.rig.output(self.count() * @as(usize, if (self.wide) 4 else 2));
        return true;
    }

    pub fn run(self: *Case) !void {
        try glm.mm.matmul(self.kernels, self.rig.s, self.x, self.stride, self.weight, self.out, self.part, self.d.?.shape.rows, self.wide, false);
        try self.rig.s.synchronize();
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
