//! Discrete affine packing is a raw-byte replacement of the frozen scalar oracle; generic numeric sum bounds are N/A.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const core = @import("core");
const c = core.contract.spec;
const g = core.contract.gates;
const Rig = @import("contract_rig.zig").Rig;
fn depth(_: c.Shape) usize {
    return 0;
}

pub const Case = struct {
    rig: Rig,
    q: *const glm.affine_quant.Quantizer,
    drop_tail: bool = false,
    groups: usize = 0,
    input: u64 = 0,
    output: u64 = 0,
    expected: std.ArrayList(u8) = .empty,
    nonfinite: bool = false,
    rejected: bool = false,

    pub fn deinit(self: *Case) void {
        self.rig.deinit();
        self.expected.deinit(self.rig.gpu.gpa);
    }
    pub fn kernel(self: *Case) g.Kernel {
        return .{ .ctx = self, .vt = &g.hooks(Case), .control = if (self.drop_tail) .shapes else null, .spec = .{
            .name = if (self.drop_tail) "control: GPU affine quantizer drops the last group" else "cuda GLM affine GPU quantizer scalar packed-byte replacement",
            .accum = .none,
            .depth = depth,
            .out = .raw,
            .nan = .{ .stated = "Reject nonfinite checkpoint weights; finite packed words and metadata equal the frozen native scalar oracle." },
            .replaces = "native scalar affine quantize",
            .tiles = .{ .{ .size = 4 }, .{}, .{ .size = 64, .step = 64, .min = 64 } },
            .served = &.{ .{ .rows = 1, .n = 1, .k = 64 }, .{ .rows = 5, .n = 1, .k = 64 }, .{ .rows = 127, .n = 1, .k = 64 }, .{ .rows = 4096, .n = 1, .k = 64 } },
        } };
    }
    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 4096 or shape.n != 1 or shape.k != 64) return false;
        self.rig.reset();
        self.groups = shape.rows;
        self.nonfinite = fill == .nonfinite;
        const a = self.rig.gpu.gpa;
        const src = try a.alloc(u16, self.groups * 64);
        defer a.free(src);
        var prng = std.Random.DefaultPrng.init(seed);
        for (src) |*v| v.* = c.toBf16((prng.random().float(f32) * 2 - 1) * 37);
        try self.expected.resize(a, self.groups * 36);
        const words = try a.alloc(u32, self.groups * 8);
        defer a.free(words);
        const scales = try a.alloc(u16, self.groups);
        defer a.free(scales);
        const biases = try a.alloc(u16, self.groups);
        defer a.free(biases);
        try glm.affine_pack_reference.quantize(src, self.groups, 64, words, scales, biases);
        @memcpy(self.expected.items[0 .. self.groups * 32], std.mem.sliceAsBytes(words));
        @memcpy(self.expected.items[self.groups * 32 ..][0 .. self.groups * 2], std.mem.sliceAsBytes(scales));
        @memcpy(self.expected.items[self.groups * 34 ..], std.mem.sliceAsBytes(biases));
        if (self.nonfinite) {
            src[0] = 0x7fc0;
            src[1] = 0x7f80;
            src[2] = 0xff80;
        }
        self.input = try self.rig.input(std.mem.sliceAsBytes(src));
        self.output = try self.rig.output(self.expected.items.len);
        return true;
    }
    pub fn run(self: *Case) !void {
        self.rejected = false;
        const groups = if (self.drop_tail and !self.nonfinite) self.groups - 1 else self.groups;
        if (groups == 0) return;
        self.q.run(self.rig.s, self.input, self.output, groups, 64) catch |err| {
            if (err != error.NonfiniteAffineWeight) return err;
            self.rejected = true;
        };
    }
    pub fn check(self: *Case, _: c.Gate) !?[]const u8 {
        if (self.nonfinite) return if (self.rejected) null else "nonfinite weights were accepted";
        if (self.rejected) return "finite weights were rejected";
        if (!std.mem.eql(u8, self.expected.items, try self.rig.outputBytes())) return "packed bytes differ from scalar oracle";
        return null;
    }
    pub fn replaced(self: *Case) ![]const u8 {
        return self.expected.items;
    }
    pub fn count(_: *Case) usize {
        return 0;
    }
    pub fn reference(_: *Case, _: []f64, _: []f64) !void {}
};

pub const Group = struct {
    q: ?glm.affine_quant.Quantizer = null,
    cases: [2]Case = undefined,
    ready: bool = false,
    pub fn deinit(self: *Group) void {
        if (self.ready) for (&self.cases) |*item| item.deinit();
        if (self.q) |*q| q.deinit();
    }
    pub fn add(self: *Group, gpu: @import("check.zig").Gpu, stream: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !void {
        self.q = try glm.affine_quant.Quantizer.init(gpu.d);
        for (&self.cases, 0..) |*item, i| item.* = .{ .rig = .{ .gpu = gpu, .s = stream }, .q = &self.q.?, .drop_tail = i == 1 };
        self.ready = true;
        for (&self.cases) |*item| try kernels.append(gpu.gpa, item.kernel());
    }
};
