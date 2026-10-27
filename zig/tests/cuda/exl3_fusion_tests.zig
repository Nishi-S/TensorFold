//! Fused EXL3 decode: compare both epilogues, counter reuse and weighted fallbacks against the original pipeline.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const exl3 = cuda.exl3_experts;
const ref = @import("exl3_experts_f64_tests.zig");
const Rig = @import("quant_ref_tests.zig").RigT;

pub fn run(r: *Rig, loaded: *const exl3.Experts) !void {
    var baseline = loaded.*;
    baseline.decode_prefetch = false;
    baseline.decode_loads = .baseline;
    baseline.decode_fuse = 0;
    var candidate = baseline;
    candidate.fused = try exl3.fusion.Kernels.resolve(loaded.mod, 3);
    candidate.grouped_prefetch = try loaded.mod.function("tf_exl3_grouped_1_pf1");
    inline for (.{ "tf_exl3_grouped_1_nc1", "tf_exl3_grouped_1_nc2", "tf_exl3_grouped_1_nc4" }, 0..) |name, i|
        candidate.grouped_vector[i] = try loaded.mod.function(name);
    const actual = try r.gpu.gpa.alloc(u8, 17 * ref.SLOTS * ref.D * 4);
    defer r.gpu.gpa.free(actual);
    const variants = [_][ref.E]u32{
        .{ 4, 4, 4, 4, 4, 4, 4, 4, 4 },
        .{ 4, 5, 6, 4, 5, 6, 4, 5, 6 },
    };
    for (variants, 0..) |bits, variant| {
        var b = try ref.buildLayer(r, &bits);
        b.layer.codebook = .mcg;
        b.layer.uniform_k2 = if (variant == 0) 8 else 0;
        const scratch_base = try r.zeros(try exl3.scratchBytes(b.layer, 17, ref.SLOTS));
        const sc = try exl3.scratch(scratch_base, b.layer, 17, ref.SLOTS);
        const out = try r.zeros(17 * ref.D * 4);
        if (variant == 0) try isolateFp32GateUp(r, &baseline, &candidate, b, sc);
        for ([_]i32{ 0, 1 }) |act_mode| {
            b.layer.act_mode = act_mode;
            b.layer.limit = if (act_mode == 0) 10 else 0;
            for ([_]usize{ 1, 3, 8, 16, 17 }) |rows| {
                for ([_]bool{ false, true }) |weighted| {
                    const weights = if (weighted) b.dwts else 0;
                    try baseline.run(r.s, b.layer, b.dx, ref.D, .bf16, b.dpick, weights, rows, out, sc);
                    const old_xd = try r.back(sc.xd, rows * ref.SLOTS * b.layer.width * 2);
                    const old_slots = try r.back(sc.y, rows * ref.SLOTS * ref.D * 4);
                    const old_out = if (weighted) try r.back(out, rows * ref.D * 4) else &.{};
                    for ([_]u2{ 1, 2, 3 }) |mask| {
                        candidate.decode_fuse = mask;
                        for ([_]exl3.prefetch.Mode{ .baseline, .scalar, .nc1, .nc2, .nc4 }) |mode| {
                            candidate.decode_loads = mode;
                            for (0..5) |repeat| {
                                const context: Context = .{ .variant = variant, .act_mode = act_mode, .rows = rows, .weighted = weighted, .mask = mask, .mode = mode, .repeat = repeat };
                                try r.gpu.d.check(r.gpu.d.api.cuMemsetD32Async(sc.done, 0xa5a5a5a5, exl3.fusion.counterBytes(ref.E, b.layer.width) / 4, r.s.handle), "poison fusion counters");
                                try candidate.run(r.s, b.layer, b.dx, ref.D, .bf16, b.dpick, weights, rows, out, sc);
                                try context.compare("EXL3 gate/up down-input", try readInto(r, sc.xd, actual[0..old_xd.len]), old_xd);
                                try context.compare("EXL3 fused slot outputs", try readInto(r, sc.y, actual[0..old_slots.len]), old_slots);
                                if (weighted) try context.compare("EXL3 fused weighted outputs", try readInto(r, out, actual[0..old_out.len]), old_out);
                            }
                            if (variant == 0 and act_mode == 0 and (rows == 3 or rows == 16) and !weighted and mask == 3 and mode == .nc4) {
                                try cuda.graph.beginCapture(r.s, .thread_local);
                                candidate.run(r.s, b.layer, b.dx, ref.D, .bf16, b.dpick, 0, rows, out, sc) catch |err| {
                                    var discarded = cuda.graph.endCapture(r.s) catch return err;
                                    discarded.deinit();
                                    return err;
                                };
                                var graph = try cuda.graph.endCapture(r.s);
                                defer graph.deinit();
                                var exec = try graph.instantiate();
                                defer exec.deinit();
                                for (0..5) |repeat| {
                                    const context: Context = .{ .variant = variant, .act_mode = act_mode, .rows = rows, .weighted = weighted, .mask = mask, .mode = mode, .repeat = repeat, .captured = true };
                                    try r.gpu.d.check(r.gpu.d.api.cuMemsetD32Async(sc.done, 0xa5a5a5a5, exl3.fusion.counterBytes(ref.E, b.layer.width) / 4, r.s.handle), "poison graph counters");
                                    try exec.launchOn(r.s);
                                    try context.compare("EXL3 graph gate/up down-input", try readInto(r, sc.xd, actual[0..old_xd.len]), old_xd);
                                    try context.compare("EXL3 fused graph slots", try readInto(r, sc.y, actual[0..old_slots.len]), old_slots);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Issue #2's same random fixture: isolate projection/publication from the epilogue before the common down kernel.
fn isolateFp32GateUp(r: *Rig, baseline: *const exl3.Experts, configured: *const exl3.Experts, b: ref.Built, sc: exl3.Scratch) !void {
    var layer = b.layer;
    layer.act_mode = 1;
    layer.limit = 0;
    const rows = 16;
    const splits = try exl3.plan(layer.dims, layer.width, true);
    const partial_bytes = 2 * splits * rows * ref.SLOTS * layer.width * 4;
    const input_bytes = rows * ref.SLOTS * layer.width * 2;
    try baseline.runGateUpForTest(r.s, layer, b.dx, ref.D, .bf16, b.dpick, rows, sc);
    const old_z = try r.back(sc.z, partial_bytes);
    const old_xd = try r.back(sc.xd, input_bytes);
    var candidate = configured.*;
    candidate.decode_fuse = 2;
    candidate.decode_loads = .baseline;
    const context: Context = .{ .variant = 0, .act_mode = 1, .rows = rows, .weighted = false, .mask = 2, .mode = .baseline, .repeat = 0 };
    try r.gpu.d.check(r.gpu.d.api.cuMemsetD32Async(sc.done, 0xa5a5a5a5, exl3.fusion.counterBytes(ref.E, layer.width) / 4, r.s.handle), "poison isolation counters");
    try candidate.runGateUpForTest(r.s, layer, b.dx, ref.D, .bf16, b.dpick, rows, sc);
    try context.compare("EXL3 pre-down gate/up partials", try r.back(sc.z, partial_bytes), old_z);
    try context.compare("EXL3 pre-down FP16 input", try r.back(sc.xd, input_bytes), old_xd);
    check.pass("EXL3 pre-down FP32 isolation: partials and down inputs byte-equal", .{});
}

fn readInto(r: *Rig, ptr: u64, bytes: []u8) ![]const u8 {
    try r.s.synchronize();
    const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = bytes.len };
    try b.download(0, bytes);
    return bytes;
}

// Print the complete failing configuration without changing the byte-equality threshold.
const Context = struct {
    variant: usize,
    act_mode: i32,
    rows: usize,
    weighted: bool,
    mask: u2,
    mode: exl3.prefetch.Mode,
    repeat: usize,
    captured: bool = false,

    fn compare(c: Context, stage: []const u8, actual: []const u8, expected: []const u8) !void {
        check.sameBytes(stage, actual, expected) catch |err| {
            std.debug.print("EXL3 fusion failure: stage={s} variant={d} act_mode={d} rows={d} weighted={} mask={d} loads={t} repeat={d} captured={}\n", .{ stage, c.variant, c.act_mode, c.rows, c.weighted, c.mask, c.mode, c.repeat, c.captured });
            return err;
        };
    }
};
