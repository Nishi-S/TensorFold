//! Dense affine Q4: sixteen clipped ranges, scored with the stored BF16 scale and bias on each 64-value group.
const std = @import("std");

fn value(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn bf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn roundEven(x: f32) f32 {
    const lo = @floor(x);
    const d = x - lo;
    return if (d < 0.5) lo else if (d > 0.5 or @mod(lo, 2) != 0) lo + 1 else lo;
}

fn code(v: f32, scale: f32, bias: f32) u32 {
    return @intFromFloat(roundEven(@max(0, @min(15, (v - bias) / scale))));
}

const Candidate = struct { scale: u16, bias: u16, loss: f32 };

fn candidate(group: *const [64]f32, lo: f32, hi: f32, i: usize) !Candidate {
    const alpha: f32 = @floatCast(1.0 - 0.45 * @as(f64, @floatFromInt(i)) / 15.0);
    const low = lo * alpha;
    const high = hi * alpha;
    const s = bf16(@max((high - low) / 15, 1e-8));
    const b = bf16(low);
    const scale = value(s);
    const bias = value(b);
    if (!std.math.isFinite(scale) or !std.math.isFinite(bias) or scale <= 0) return error.AffineRangeOverflow;
    var loss: f32 = 0;
    for (group) |v| {
        const q: f32 = @floatFromInt(code(v, scale, bias));
        const product = q * scale;
        const residual = product + bias - v;
        loss += residual * residual;
    }
    if (!std.math.isFinite(loss)) return error.AffineRangeOverflow;
    return .{ .scale = s, .bias = b, .loss = loss };
}

/// Row-major BF16 [n,k] to low-first nibbles and BF16 [n,k/64] metadata; deterministic scalar-FP32 MSE scoring.
pub fn quantize(src: []const u16, n: usize, k: usize, words: []u32, scales: []u16, biases: []u16) !void {
    @setFloatMode(.strict);
    if (n == 0 or k == 0 or k % 64 != 0) return error.BadAffineShape;
    const elements = std.math.mul(usize, n, k) catch return error.BadAffineShape;
    if (src.len != elements or words.len != elements / 8 or scales.len != elements / 64 or
        biases.len != elements / 64) return error.BadAffineShape;
    for (src) |h| if (!std.math.isFinite(value(h))) return error.NonfiniteAffineWeight;
    for (0..elements / 64) |g| {
        var group: [64]f32 = undefined;
        var lo: f32 = std.math.inf(f32);
        var hi: f32 = -std.math.inf(f32);
        for (&group, src[g * 64 ..][0..64]) |*v, h| {
            v.* = value(h);
            lo = @min(lo, v.*);
            hi = @max(hi, v.*);
        }
        var best = try candidate(&group, lo, hi, 0);
        for (1..16) |i| {
            const trial = try candidate(&group, lo, hi, i);
            if (trial.loss < best.loss) best = trial;
        }
        scales[g] = best.scale;
        biases[g] = best.bias;
        const codes = words[g * 8 ..][0..8];
        @memset(codes, 0);
        for (group, 0..) |v, i| codes[i / 8] |= code(v, value(best.scale), value(best.bias)) << @intCast(4 * (i % 8));
    }
}

test "affine Q4 packs an exact unsigned grid with BF16 metadata across rows" {
    var src: [256]u16 = undefined;
    for (&src, 0..) |*v, i| v.* = bf16(@as(f32, @floatFromInt(i % 16)) + @as(f32, @floatFromInt(i / 64)) * 16);
    var words: [32]u32 = undefined;
    var scales: [4]u16 = undefined;
    var biases: [4]u16 = undefined;
    try quantize(&src, 2, 128, &words, &scales, &biases);
    for (scales, biases, 0..) |s, b, i| {
        try std.testing.expectEqual(bf16(1), s);
        try std.testing.expectEqual(bf16(@as(f32, @floatFromInt(i)) * 16), b);
        try std.testing.expectEqual(@as(u32, 0x76543210), words[i * 8]);
        try std.testing.expectEqual(@as(u32, 0xfedcba98), words[i * 8 + 1]);
    }
}

test "constant and zero groups keep first tied minimum; clipping improves outlier fixture" {
    var src: [64]u16 = @splat(bf16(0));
    var words: [8]u32 = undefined;
    var scales: [1]u16 = undefined;
    var biases: [1]u16 = undefined;
    for ([_]f32{ 0, 3, -3 }) |v| {
        @memset(&src, bf16(v));
        try quantize(&src, 1, 64, &words, &scales, &biases);
        try std.testing.expectEqual(bf16(1e-8), scales[0]);
        try std.testing.expectEqual(bf16(v), biases[0]);
        for (words) |w| try std.testing.expectEqual(@as(u32, 0), w);
    }
    @memset(&src, 0);
    src[0] = bf16(-1);
    src[63] = bf16(1);
    try quantize(&src, 1, 64, &words, &scales, &biases);
    var group: [64]f32 = undefined;
    for (&group, src) |*v, h| v.* = value(h);
    const full = try candidate(&group, -1, 1, 0);
    var got: f32 = 0;
    for (group, 0..) |v, i| {
        const q = (words[i / 8] >> @as(u5, @intCast(4 * (i % 8)))) & 15;
        const residual = @as(f32, @floatFromInt(q)) * value(scales[0]) + value(biases[0]) - v;
        got += residual * residual;
    }
    try std.testing.expect(got < full.loss);
    for (0..16) |i| try std.testing.expect(got <= (try candidate(&group, -1, 1, i)).loss);
}

test "ties round to even and unsupported geometry or weights are refused" {
    for ([_]f32{ 0.5, 1.5, 2.5, 3.5 }, [_]f32{ 0, 2, 2, 4 }) |v, want|
        try std.testing.expectEqual(want, roundEven(v));
    var src: [64]u16 = @splat(0);
    var words: [8]u32 = undefined;
    var scales: [1]u16 = undefined;
    var biases: [1]u16 = undefined;
    try std.testing.expectError(error.BadAffineShape, quantize(&src, 1, 63, &words, &scales, &biases));
    try std.testing.expectError(error.BadAffineShape, quantize(&src, 1, 64, words[0..7], &scales, &biases));
    try std.testing.expectError(error.BadAffineShape, quantize(&src, std.math.maxInt(usize), 64, &words, &scales, &biases));
    for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }) |h| {
        src[0] = h;
        try std.testing.expectError(error.NonfiniteAffineWeight, quantize(&src, 1, 64, &words, &scales, &biases));
    }
}
