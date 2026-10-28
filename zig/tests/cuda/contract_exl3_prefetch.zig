//! EXL3 register prefetch: the existing grouped projection's bytes, guards and independent rotated-domain sums.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const fmt = core.exl3_format;
const experts = 3;

fn depth(s: c.Shape) usize {
    return s.k + 4;
}

pub const Case = struct {
    rig: rig_.Rig,
    candidate: cuda.Function,
    baseline: cuda.Function,
    name: ?[]const u8 = null,
    down_fused: bool = false,
    down_baseline: ?cuda.Function = null,
    gateup_fused: bool = false,
    gateup_baseline: ?cuda.Function = null,
    drop_tail: bool = false,
    sk: usize,
    mats: usize,
    shape: c.Shape = undefined,
    arena: std.heap.ArenaAllocator,
    x: []f16 = &.{},
    weights: []f16 = &.{},
    dx: u64 = 0,
    candidate_x: u64 = 0,
    tables: [2]u64 = undefined,
    widths: u64 = 0,
    ids: u64 = 0,
    total: u64 = 0,
    members: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    raw_base: u64 = 0,
    scales: u64 = 0,
    picks: u64 = 0,
    done: u64 = 0,

    pub fn deinit(self: *Case) void {
        self.rig.deinit();
        self.arena.deinit();
    }

    pub fn kernel(self: *Case) g.Kernel {
        return .{ .ctx = self, .vt = &vt, .control = if (self.drop_tail) .shapes else null, .spec = .{
            .name = self.name orelse if (self.drop_tail) "control: EXL3 prefetch drops final K tile" else if (self.down_fused) "cuda EXL3 fused down epilogue" else if (self.sk == 4) "cuda EXL3 prefetch gate/up split4" else "cuda EXL3 prefetch down split1",
            .accum = .f32,
            .depth = depth,
            .ops = if (self.down_fused or self.gateup_fused) 9 else 0,
            .out = if (self.gateup_fused) .f16 else .f32,
            .replaces = if (self.gateup_fused) "tf_exl3_grouped_1 + tf_exl3_gateup_epilogue" else if (self.down_fused) "tf_exl3_grouped_1 + tf_exl3_down_combine" else "tf_exl3_grouped_1",
            .tiles = .{ .{ .size = 16 }, .{ .size = 128, .step = 128, .min = 128 }, .{ .size = 256, .step = 256, .min = 256 } },
            .served = if (self.sk == 4) &.{
                .{ .rows = 1, .n = 1024, .k = 4096 }, .{ .rows = 3, .n = 1024, .k = 4096 },
                .{ .rows = 8, .n = 128, .k = 256 },   .{ .rows = 16, .n = 128, .k = 512 },
            } else &.{
                .{ .rows = 1, .n = 4096, .k = 1024 }, .{ .rows = 3, .n = 4096, .k = 1024 },
                .{ .rows = 8, .n = 128, .k = 256 },   .{ .rows = 16, .n = 128, .k = 512 },
            },
        } };
    }

    const vt = g.hooks(Case);

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 16 or shape.n == 0 or shape.n % 128 != 0 or
            shape.k == 0 or shape.k % (64 * self.sk) != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.x = try a.alloc(f16, shape.rows * shape.k);
        for (self.x) |*v| v.* = @floatCast((rng.float(f32) * 2 - 1) * 0.5);
        const slice = shape.k / self.sk;
        if (fill == .stress) for (0..shape.rows) |r| {
            for (0..self.sk) |s| for (0..slice / 2) |i| {
                const v: f16 = @floatFromInt(256 + (i + 77) % 128);
                self.x[r * shape.k + s * slice + i] = v;
                self.x[r * shape.k + s * slice + slice / 2 + i] = -v + @as(f16, if (i < 8) 1 else 0);
            };
        };
        if (fill == .nonfinite) {
            const plants = [_]f16{ std.math.nan(f16), std.math.inf(f16), -std.math.inf(f16) };
            for (0..@min(shape.rows, plants.len)) |r| self.x[r * shape.k] = plants[r];
            if (shape.rows > 3) {
                self.x[3 * shape.k] = std.math.inf(f16);
                self.x[3 * shape.k + 1] = -std.math.inf(f16);
            }
        }
        self.dx = try self.rig.input(std.mem.sliceAsBytes(self.x));
        self.candidate_x = self.dx;
        if (self.drop_tail) {
            const altered = try a.dupe(f16, self.x);
            for (0..shape.rows) |r| for (0..self.sk) |s| {
                const at = r * shape.k + (s + 1) * slice - 16;
                @memset(altered[at..][0..16], 0);
            };
            self.candidate_x = try self.rig.input(std.mem.sliceAsBytes(altered));
        }
        self.weights = try a.alloc(f16, self.mats * experts * shape.k * shape.n);
        const raw = try a.alloc(i16, shape.k * shape.n / 4);
        for (0..self.mats) |mat| {
            var ptrs: [experts]u64 = undefined;
            for (&ptrs, 0..) |*ptr, e| {
                for (raw) |*v| v.* = if (fill == .stress) 0 else rng.int(i16);
                fmt.unpack(.{ .halves = 8 }, .mcg, raw, shape.k / 16, shape.n / 16, self.weights[(mat * experts + e) * shape.k * shape.n ..][0 .. shape.k * shape.n]);
                ptr.* = try self.rig.input(std.mem.sliceAsBytes(raw));
            }
            self.tables[mat] = try self.rig.input(std.mem.sliceAsBytes(&ptrs));
        }
        if (self.mats == 1) self.tables[1] = self.tables[0];
        const widths: [experts]i32 = @splat(8);
        self.widths = try self.rig.input(std.mem.sliceAsBytes(&widths));
        const ids = [_]i32{ 0, 1, 2 };
        self.ids = try self.rig.input(std.mem.sliceAsBytes(&ids));
        const total = [_]i32{experts};
        self.total = try self.rig.input(std.mem.sliceAsBytes(&total));
        const members = try a.alloc(i32, experts * shape.rows);
        @memset(members, -1);
        var used: [experts]usize = @splat(0);
        for (0..shape.rows) |r| {
            const e = r % experts;
            members[e * shape.rows + used[e]] = @intCast(r * 32);
            used[e] += 1;
        }
        self.members = try self.rig.input(std.mem.sliceAsBytes(members));
        self.out = try self.rig.output(self.count() * self.elemBytes());
        self.base = try self.rig.scratch(self.count() * self.elemBytes());
        if (self.down_fused or self.gateup_fused) {
            self.raw_base = try self.rig.scratch(self.rawCount() * 4);
            self.done = try self.rig.scratch(cuda.exl3_experts.fusion.counterBytes(experts, shape.n));
            const scales = try a.alloc(f16, experts * shape.n);
            for (scales, 0..) |*v, i| v.* = if (self.gateup_fused) 1 else @floatCast(1.0 + @as(f32, @floatFromInt(i / shape.n)) * 0.125);
            self.scales = try self.rig.input(std.mem.sliceAsBytes(scales));
            const picks = try a.alloc(i32, shape.rows);
            for (picks, 0..) |*v, i| v.* = @intCast(i % experts);
            self.picks = try self.rig.input(std.mem.sliceAsBytes(picks));
        }
        return true;
    }

    fn launch(self: *Case, f: cuda.Function, x: u64, out: u64, fused: bool) !void {
        var args: cuda.Args = .{};
        args.add(x);
        args.add(x);
        args.add(self.tables[0]);
        args.add(self.tables[1]);
        args.add(self.widths);
        args.add(self.widths);
        args.add(self.ids);
        args.add(self.total);
        args.add(self.members);
        args.add(if (fused and self.gateup_fused) self.raw_base else out);
        const sh = self.shape;
        for ([_]usize{ sh.k, sh.n, sh.rows, self.sk, sh.rows, 1, experts, self.mats, 0, 0 }) |v|
            args.add(@as(i32, @intCast(v)));
        args.add(@as(i32, -1));
        if (fused) {
            if (self.gateup_fused) try cuda.exl3_experts.fusion.clear(self.rig.s, self.done, experts, sh.n);
            cuda.exl3_experts.fusion.append(&args, self.scales, self.scales, self.scales, out, self.done, if (self.gateup_fused) 10 else 0, 0, 0);
        }
        try cuda.launch.launch(f, .{
            .grid = .{ .x = experts, .y = @intCast(sh.n / 128), .z = @intCast(self.mats * self.sk) },
            .block = .{ .x = 128 },
        }, self.rig.s, &args);
    }

    pub fn run(self: *Case) !void {
        try self.launch(self.candidate, self.candidate_x, self.out, self.down_fused or self.gateup_fused);
        try self.rig.s.synchronize();
    }

    pub fn replaced(self: *Case) ![]const u8 {
        try self.launch(self.baseline, self.dx, if (self.down_fused or self.gateup_fused) self.raw_base else self.base, false);
        if (self.down_fused) {
            var a: cuda.Args = .{};
            a.add(self.raw_base);
            a.add(self.picks);
            a.add(self.scales);
            a.add(self.base);
            a.add(@as(u64, 0));
            a.add(@as(u64, 0));
            for ([_]usize{ self.shape.rows, self.shape.n, 1, experts, 1 }) |v| a.add(@as(i32, @intCast(v)));
            try cuda.launch.launch(self.down_baseline.?, .{
                .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast(self.shape.n / 128) },
                .block = .{ .x = 32 },
            }, self.rig.s, &a);
        }
        if (self.gateup_fused) {
            var a: cuda.Args = .{};
            a.add(self.raw_base);
            a.add(self.picks);
            for (0..3) |_| a.add(self.scales);
            a.add(self.base);
            for ([_]usize{ self.shape.rows, self.shape.n, self.sk, experts }) |v| a.add(@as(i32, @intCast(v)));
            a.add(@as(f32, 10));
            a.add(@as(i32, 0));
            try cuda.launch.launch(self.gateup_baseline.?, .{
                .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast(self.shape.n / 128) },
                .block = .{ .x = 32 },
            }, self.rig.s, &a);
        }
        return self.rig.bytes(self.base, self.count() * self.elemBytes());
    }

    fn elemBytes(self: *Case) usize {
        return if (self.gateup_fused) 2 else 4;
    }

    pub fn count(self: *Case) usize {
        return if (self.gateup_fused) self.shape.rows * self.shape.n else self.rawCount();
    }

    fn rawCount(self: *Case) usize {
        return self.mats * self.sk * self.shape.rows * self.shape.n;
    }

    fn terms(self: *Case, index: usize, out: []f64) void {
        const sh = self.shape;
        const col = index % sh.n;
        const row = index / sh.n % sh.rows;
        const split = index / sh.n / sh.rows % self.sk;
        const mat = index / sh.n / sh.rows / self.sk;
        const e = row % experts;
        for (out, 0..) |*v, i| {
            const k = split * out.len + i;
            v.* = @as(f64, self.x[row * sh.k + k]) *
                @as(f64, self.weights[((mat * experts + e) * sh.k + k) * sh.n + col]);
        }
    }

    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        const buf = try self.rig.gpu.gpa.alloc(f64, self.shape.k / self.sk);
        defer self.rig.gpu.gpa.free(buf);
        const raw = if (self.gateup_fused) try self.rig.gpu.gpa.alloc(f64, self.rawCount()) else want;
        defer if (self.gateup_fused) self.rig.gpu.gpa.free(raw);
        for (raw, 0..) |*w, i| {
            var magnitude: f64 = 0;
            self.terms(i, buf);
            w.* = 0;

            for (buf) |v| {
                w.* += v;
                magnitude += @abs(v);
            }
            if (!self.gateup_fused) mag[i] = magnitude;
        }
        if (self.gateup_fused) {
            self.finishGateUp(raw, want);
            @memset(mag, 0); // Replacement gate checks bytes; the stress fixture checks its rounded exact sums.
        }
        if (self.down_fused) {
            self.rotate(want);
            for (0..mag.len / 128) |block| {
                var total: f64 = 0;
                for (mag[block * 128 ..][0..128]) |v| total += v;
                const row = block * 128 / self.shape.n;
                const scale = had_scale * (1.0 + @as(f64, @floatFromInt(row % experts)) * 0.125);
                @memset(mag[block * 128 ..][0..128], total * scale);
            }
        }
    }

    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        const buf = try self.rig.gpu.gpa.alloc(f64, self.shape.k / self.sk);
        defer self.rig.gpu.gpa.free(buf);
        const raw = if (self.gateup_fused) try self.rig.gpu.gpa.alloc(f64, self.rawCount()) else out;
        defer if (self.gateup_fused) self.rig.gpu.gpa.free(raw);
        for (raw, 0..) |*v, i| {
            self.terms(i, buf);
            v.* = core.contract.rows.sumIn(acc, order, buf);
        }
        if (self.gateup_fused) self.finishGateUp(raw, out);
        if (self.down_fused) self.rotate(out);
    }

    fn finishGateUp(self: *Case, raw: []const f64, out: []f64) void {
        const stride = self.shape.rows * self.shape.n;
        for (0..stride / 128) |block| {
            var gate: [128]f64 = @splat(0);
            var up: [128]f64 = @splat(0);
            for (0..self.sk) |split| for (0..128) |j| {
                gate[j] += raw[split * stride + block * 128 + j];
                up[j] += raw[(self.sk + split) * stride + block * 128 + j];
            };
            hadamard(&gate);
            hadamard(&up);
            for (&gate, up) |*gv, uv| {
                var gg = bf(gv.* * had_scale);
                var uu = bf(uv * had_scale);
                gg = if (gg > 10) 10 else gg;
                uu = if (uu > 10) 10 else if (uu < -10) -10 else uu;
                gv.* = bf(bf(gg / (1 + @exp(-gg))) * uu);
            }
            hadamard(&gate);
            for (out[block * 128 ..][0..128], gate) |*v, gv| v.* = gv * had_scale;
        }
    }

    fn rotate(self: *Case, out: []f64) void {
        for (0..out.len / 128) |block| {
            const values = out[block * 128 ..][0..128];
            hadamard(values);
            const row = block * 128 / self.shape.n;
            const scale = had_scale * (1.0 + @as(f64, @floatFromInt(row % experts)) * 0.125);
            for (values) |*v| v.* *= scale;
        }
    }
};

const had_scale: f64 = @as(f32, 0.08838834764831845);

fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

fn hadamard(values: []f64) void {
    var step: usize = 1;
    while (step < 128) : (step *= 2) {
        var start: usize = 0;
        while (start < 128) : (start += 2 * step) for (0..step) |i| {
            const a = values[start + i];
            const b = values[start + step + i];
            values[start + i] = a + b;
            values[start + step + i] = a - b;
        };
    }
}
