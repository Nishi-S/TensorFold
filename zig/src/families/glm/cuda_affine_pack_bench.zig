//! Host-only affine Q4 packing comparison; output equality is checked before timings are reported.
const std = @import("std");
const current = @import("cuda_affine_pack.zig");
const reference = @import("cuda_affine_pack_reference_test.zig");

fn now() !u64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts) != 0) return error.ClockUnavailable;
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

pub fn main() !void {
    const a = std.heap.page_allocator;
    const n = 512;
    const k = 2048;
    const src = try a.alloc(u16, n * k);
    defer a.free(src);
    var rng = std.Random.DefaultPrng.init(0xa441_2026);
    for (src) |*v| {
        const x: f32 = (rng.random().float(f32) * 2 - 1) * 37;
        const bits: u32 = @bitCast(x);
        v.* = @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
    }
    const words = try a.alloc(u32, n * k / 8);
    defer a.free(words);
    const scales = try a.alloc(u16, n * k / 64);
    defer a.free(scales);
    const biases = try a.alloc(u16, n * k / 64);
    defer a.free(biases);
    const ref_words = try a.alloc(u32, words.len);
    defer a.free(ref_words);
    const ref_scales = try a.alloc(u16, scales.len);
    defer a.free(ref_scales);
    const ref_biases = try a.alloc(u16, biases.len);
    defer a.free(ref_biases);
    try reference.quantize(src, n, k, ref_words, ref_scales, ref_biases);
    try current.quantizePruned(src, n, k, words, scales, biases);
    var old_ms: [5]f64 = undefined;
    var new_ms: [5]f64 = undefined;
    for (0..5) |repeat| {
        for (0..2) |phase| {
            const old = (repeat + phase) % 2 == 0;
            const start = try now();
            if (old) try reference.quantize(src, n, k, ref_words, ref_scales, ref_biases) else try current.quantizePruned(src, n, k, words, scales, biases);
            const elapsed = @as(f64, @floatFromInt((try now()) - start)) / 1e6;
            if (old) old_ms[repeat] = elapsed else new_ms[repeat] = elapsed;
        }
        if (!std.mem.eql(u32, ref_words, words) or !std.mem.eql(u16, ref_scales, scales) or
            !std.mem.eql(u16, ref_biases, biases)) return error.PackedBytesDiffer;
        std.debug.print("repeat={d} floor_mod_ms={d:.3} pruned_ms={d:.3}\n", .{ repeat, old_ms[repeat], new_ms[repeat] });
    }
    std.sort.heap(f64, &old_ms, {}, std.sort.asc(f64));
    std.sort.heap(f64, &new_ms, {}, std.sort.asc(f64));
    std.debug.print("affine-pack elements={d} identical=true old_median_ms={d:.3} new_median_ms={d:.3} speedup={d:.3}\n", .{ src.len, old_ms[2], new_ms[2], old_ms[2] / new_ms[2] });
}
