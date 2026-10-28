//! Frozen pre-optimization scalar oracle: floor/mod tie rounding and ordered candidate MSE sums.
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
