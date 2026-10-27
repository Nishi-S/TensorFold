//! A cache hint has no numerical outputs: prove only immutable input bits and guards, with accumulation N/A.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;

fn depth(_: c.Shape) usize {
    return 0;
}

pub const Case = struct {
    rig: rig_.Rig,
    candidate: cuda.Function,
    mutate: bool = false,
    mode: u2 = 2,
    ptr: u64 = 0,
    len: usize = 0,
    expected: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *Case) void {
        self.rig.deinit();
        self.expected.deinit(self.rig.gpu.gpa);
    }

    pub fn kernel(self: *Case) g.Kernel {
        return .{ .ctx = self, .vt = &vt, .control = if (self.mutate) .repeat else null, .spec = .{
            .name = if (self.mutate) "control: L2 hint mutates a weight byte" else switch (self.mode) {
                1 => "cuda GLM L2 bulk immutable cache hints",
                3 => "cuda GLM L2 touch immutable reads",
                else => "cuda GLM L2 line immutable cache hints",
            },
            .accum = .none,
            .depth = depth,
            .out = .raw,
            .nan = .{ .stated = "Cache hints have no numeric output and preserve every input bit, including nonfinite payloads." },
            .tiles = .{ .{}, .{}, .{ .size = 128, .step = 1, .min = 1 } },
            .served = &.{ .{ .rows = 1, .n = 1, .k = 129 }, .{ .rows = 1, .n = 1, .k = 1 }, .{ .rows = 1, .n = 1, .k = 8 << 20 }, .{ .rows = 1, .n = 1, .k = 64 << 20 } },
        } };
    }
    const vt = g.hooks(Case);

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k == 0 or shape.k > 64 << 20) return false;
        self.rig.reset();
        self.len = shape.k;
        try self.expected.resize(self.rig.gpu.gpa, self.len);
        var prng = std.Random.DefaultPrng.init(seed);
        prng.random().bytes(self.expected.items);
        self.expected.items[0] = 0x5a;
        if (fill == .nonfinite and self.len >= 12) {
            for ([_]u32{ 0x7fc00000, 0x7f800000, 0xff800000 }, 0..) |v, i|
                std.mem.writeInt(u32, self.expected.items[i * 4 ..][0..4], v, .little);
        }
        self.ptr = try self.rig.input(self.expected.items);
        return true;
    }

    pub fn run(self: *Case) !void {
        var args: cuda.Args = .{};
        args.add(self.ptr);
        args.add(@as(u64, self.len));
        const lines = (self.len + 127) / 128;
        try cuda.launch.launch(self.candidate, .{
            .grid = .{ .x = @intCast(@min(32, (lines + 127) / 128)) },
            .block = .{ .x = 128 },
        }, self.rig.s, &args);
        if (self.mutate) try self.rig.zero(self.ptr, 1);
        try self.rig.s.synchronize();
    }

    pub fn check(self: *Case, _: c.Gate) !?[]const u8 {
        if (!std.mem.eql(u8, self.expected.items, try self.rig.inputBytes())) return "weight bytes changed";
        if (!try self.rig.intact()) return "allocation guard changed";
        return null;
    }

    pub fn count(_: *Case) usize {
        return 0;
    }
    pub fn reference(_: *Case, _: []f64, _: []f64) !void {}
};
