//! WY32 prompt arithmetic: independent serial FP64 reference, guarded inputs and outputs, and 32-row seams.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const contract = @import("core").contract;
const c = contract.spec;
const g = contract.gates;
const Rig = @import("contract_rig.zig").Rig;
fn depth(s: c.Shape) usize {
    return (s.rows + 1) * 512;
}
pub const Case = struct {
    rig: Rig,
    function: cuda.Function,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    q: []f32 = &.{},
    k: []f32 = &.{},
    decay: []f32 = &.{},
    v: []u16 = &.{},
    beta: []f32 = &.{},
    state: []f32 = &.{},
    p: [8]u64 = @splat(0),
    expected_state: [128 * 128]f64 = undefined,
    state_bound: [128 * 128]f64 = undefined,
    drop_tail: bool = false,
    pub fn deinit(s: *Case) void {
        s.rig.deinit();
        s.arena.deinit();
    }
    pub fn kernel(s: *Case) g.Kernel {
        return .{ .spec = .{
            .name = if (s.drop_tail) "control: WY32 drops final row" else "cuda GLM KDA WY32 prompt",
            .accum = .f32,
            .depth = depth,
            .out = .bf16,
            .tiles = .{ .{ .size = 32 }, .{ .size = 128, .min = 128, .step = 128 }, .{ .size = 128, .min = 128, .step = 128 } },
            .served = &.{ .{ .rows = 1, .n = 128, .k = 128 }, .{ .rows = 31, .n = 128, .k = 128 }, .{ .rows = 32, .n = 128, .k = 128 }, .{ .rows = 33, .n = 128, .k = 128 }, .{ .rows = 65, .n = 128, .k = 128 } },
        }, .ctx = s, .vt = &g.hooks(Case), .control = if (s.drop_tail) .shapes else null };
    }
    pub fn prepare(s: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != 128 or shape.k != 128) return false;
        s.rig.reset();
        _ = s.arena.reset(.retain_capacity);
        s.shape = shape;
        const a = s.arena.allocator();
        const n = shape.rows * 128;
        s.q = try a.alloc(f32, n);
        s.k = try a.alloc(f32, n);
        s.decay = try a.alloc(f32, n);
        s.v = try a.alloc(u16, n);
        s.beta = try a.alloc(f32, shape.rows);
        s.state = try a.alloc(f32, 128 * 128);
        var rng = std.Random.DefaultPrng.init(seed);
        const random = rng.random();
        for (s.q, s.k, s.decay, s.v, 0..) |*q, *k, *decay, *v, i| {
            q.* = (random.float(f32) - 0.5) / 16;
            k.* = (random.float(f32) - 0.5) / 8;
            decay.* = if (i % 31 == 0) 0 else 0.75 + random.float(f32) * 0.25;
            v.* = c.toBf16(random.float(f32) - 0.5);
        }
        for (s.beta) |*v| v.* = c.fromBf16(c.toBf16(random.float(f32)));
        for (s.state) |*v| v.* = random.float(f32) - 0.5;
        if (fill == .stress) {
            @memset(s.q, 1);
            @memset(s.k, 0);
            @memset(s.decay, 1);
            @memset(s.beta, 0);
            @memset(s.v, 0);
            for (s.state, 0..) |*v, i| {
                const col = i % 128;
                v.* = if ((i / 128) % 2 == 0)
                    (if (col % 2 == 1) @as(f32, 1) else if (col < 64) @as(f32, 65536) else -65536)
                else
                    (if (col < 32) @as(f32, 65536) else if (col < 64 or col >= 96) @as(f32, 1) else -65536);
            }
        }
        if (fill == .nonfinite) {
            @memset(s.q, 0.125);
            @memset(s.k, 0.125);
            @memset(s.decay, 0.875);
            @memset(s.beta, 0.5);
            s.state[0] = std.math.nan(f32);
            s.state[128] = std.math.inf(f32);
            s.state[256] = -std.math.inf(f32);
        }

        s.p = .{ try s.rig.input(std.mem.sliceAsBytes(s.state)), try s.rig.input(std.mem.sliceAsBytes(s.q)), try s.rig.input(std.mem.sliceAsBytes(s.k)), try s.rig.input(std.mem.sliceAsBytes(s.v)), try s.rig.input(std.mem.sliceAsBytes(s.decay)), try s.rig.input(std.mem.sliceAsBytes(s.beta)), try s.rig.output(n * 2), try s.rig.scratch(128 * 128 * 4) };
        return true;
    }
    pub fn run(s: *Case) !void {
        var a: cuda.Args = .{};
        a.add(@as(c_int, 1));
        for (s.p[0..6]) |p| a.add(p);
        a.add(@as(c_int, @intCast(s.shape.rows - @intFromBool(s.drop_tail))));
        a.add(s.p[6]);
        a.add(s.p[7]);
        a.add(@as(c_int, 0));
        try cuda.launch.launch(s.function, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 }, .shared = glm.kda.wy_shared_bytes }, s.rig.s, &a);
    }
    pub fn count(s: *Case) usize {
        return s.shape.rows * 128;
    }
    pub fn reference(s: *Case, want: []f64, mag: []f64) !void {
        var state: [128 * 128]f64 = undefined;
        var bound: [128 * 128]f64 = undefined;
        for (&state, &bound, s.state) |*v, *b, x| {
            v.* = x;
            b.* = @abs(x);
        }
        for (0..s.shape.rows) |r| for (0..128) |v| {
            var dot: f64 = 0;
            var dot_mag: f64 = 0;
            for (0..128) |k| {
                const i = v * 128 + k;
                const at = r * 128 + k;
                state[i] *= s.decay[at];
                bound[i] *= @abs(s.decay[at]);
                dot += state[i] * s.k[at];
                dot_mag += bound[i] * @abs(s.k[at]);
            }
            const value = c.fromBf16(s.v[r * 128 + v]);
            const delta = (value - dot) * s.beta[r];
            const dm = (@abs(value) + dot_mag) * @abs(s.beta[r]);
            var out: f64 = 0;
            var om: f64 = 0;
            for (0..128) |k| {
                const i = v * 128 + k;
                const at = r * 128 + k;
                state[i] += delta * s.k[at];
                bound[i] += dm * @abs(s.k[at]);
                out += state[i] * s.q[at];
                om += bound[i] * @abs(s.q[at]);
            }
            want[r * 128 + v] = out;
            mag[r * 128 + v] = om;
        };
        s.expected_state = state;
        s.state_bound = bound;
    }
    pub fn check(s: *Case, gate: g.Gate) !?[]const u8 {
        const a = s.arena.allocator();
        const want = try a.alloc(f64, s.count());
        const mag = try a.alloc(f64, s.count());
        try s.reference(want, mag);
        var spec = s.kernel().spec;
        const output = try s.rig.bytes(s.p[6], s.count() * 2);
        var nonfinite: usize = 0;
        for (want, mag, 0..) |w, m, i| {
            const got = c.read(.bf16, output, i);
            if (!std.math.isFinite(w)) nonfinite += 1;
            if (c.classOf(got) != c.classOf(w)) return "WY readout changed nonfinite class";
            if (std.math.isFinite(w) and @abs(got - w) > c.bound(spec, s.shape, w, m)) return "WY readout exceeds FP64 bound";
        }
        if (gate == .nonfinite and (nonfinite == 0 or nonfinite == want.len)) return "WY nonfinite fixture lacks finite and nonfinite readout controls";
        const bytes = try s.rig.bytes(s.p[7], 128 * 128 * 4);
        spec.out = .f32;
        for (s.expected_state, s.state_bound, 0..) |w, m, i| {
            const got = c.read(.f32, bytes, i);
            if (c.classOf(got) != c.classOf(w)) return "WY final recurrent state changed nonfinite class";
            if (std.math.isFinite(w) and @abs(got - w) > c.bound(spec, s.shape, w, m)) return "WY final recurrent state exceeds FP64 bound";
        }
        return null;
    }
    pub fn simulate(s: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        var terms: [128]f64 = undefined;
        for (out, 0..) |*o, i| {
            for (&terms, 0..) |*v, k| v.* = @as(f64, s.state[(i % 128) * 128 + k]) * s.q[(i / 128) * 128 + k];
            o.* = contract.rows.sumIn(acc, order, &terms);
        }
    }
};

pub const Group = struct {
    module: ?cuda.Module = null,
    cases: [2]Case = undefined,
    ready: bool = false,
    pub fn deinit(self: *Group) void {
        if (self.ready) for (&self.cases) |*item| item.deinit();
        if (self.module) |*m| m.unload();
    }
    pub fn add(self: *Group, gpu: @import("check.zig").Gpu, stream: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !void {
        self.module = try cuda.Module.load(gpu.d, cuda.kernels.glm_kda);
        const f = try self.module.?.function("tf_glm_kda_wy");
        try f.allowDynamicShared(glm.kda.wy_shared_bytes);
        for (&self.cases, 0..) |*item, i| item.* = .{ .rig = .{ .gpu = gpu, .s = stream }, .function = f, .arena = .init(gpu.gpa), .drop_tail = i == 1 };
        self.ready = true;
        for (&self.cases) |*item| try kernels.append(gpu.gpa, item.kernel());
    }
};
