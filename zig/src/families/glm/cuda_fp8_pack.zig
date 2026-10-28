//! Mia-style weight-only E4M3FN: BF16 rows, FP32 absmax scales per 128 columns, BF16 activations at execution.
const std = @import("std");
const codec = @import("core").exlr8.codec;

pub const group_size = 128;
pub const decode = codec.e4m3Value;
pub const encode = codec.e4m3FromF32;

fn value(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// Reject malformed geometry and non-finite checkpoint weights before mutating the output arrays.
pub fn quantize(src: []const u16, n: usize, k: usize, codes: []u8, scales: []f32) !void {
    if (n == 0 or k == 0 or k % group_size != 0) return error.BadFp8Shape;
    const count = std.math.mul(usize, n, k) catch return error.BadFp8Shape;
    if (src.len != count or codes.len != count or scales.len != count / group_size) return error.BadFp8Shape;
    for (src) |h| if (!std.math.isFinite(value(h))) return error.NonFiniteFp8Weight;
    for (scales, 0..) |*scale, g| {
        const values = src[g * group_size ..][0..group_size];
        var amax: f32 = 0;
        for (values) |h| amax = @max(amax, @abs(value(h)));
        scale.* = @max(amax / 448.0, 1e-30);
        for (values, 0..) |h, i| codes[g * group_size + i] = encode(@max(-448.0, @min(448.0, value(h) / scale.*)));
    }
}

test "FP8 uses a separate FP32 scale for every row and block of 128" {
    var src: [512]u16 = undefined;
    for (&src, 0..) |*h, i| {
        const v: f32 = if (i / 128 == 0) 0 else if (i / 128 == 1) 448 else if (i / 128 == 2) -224 else 1;
        h.* = @truncate(@as(u32, @bitCast(v)) >> 16);
    }
    var codes: [512]u8 = undefined;
    var scales: [4]f32 = undefined;
    try quantize(&src, 2, 256, &codes, &scales);
    try std.testing.expectEqual(@as(f32, 1e-30), scales[0]);
    try std.testing.expectEqual(@as(f32, 1), scales[1]);
    try std.testing.expectEqual(@as(f32, 0.5), scales[2]);
    for (src, codes, 0..) |h, q, i| try std.testing.expectApproxEqAbs(value(h), decode(q) * scales[i / 128], 1e-6);
}

test "FP8 conversion ties, subnormals, signs and saturated endpoints" {
    for (0..256) |i| {
        const code: u8 = @intCast(i);
        if (code & 0x7f == 0x7f) continue;
        try std.testing.expectEqual(code, encode(decode(code)));
    }
    try std.testing.expectEqual(@as(u8, 0x38), encode(1.0625));
    try std.testing.expectEqual(@as(u8, 0x3a), encode(1.1875));
    try std.testing.expectEqual(@as(u8, 0), encode(0x1p-10));
    try std.testing.expectEqual(@as(u8, 2), encode(3 * 0x1p-10));
    try std.testing.expectEqual(@as(u8, 0x7e), encode(1e20));
    try std.testing.expectEqual(@as(u8, 0xfe), encode(-1e20));
}

test "FP8 refuses non-finite weights and malformed geometry without publishing partial packs" {
    var src: [128]u16 = @splat(0);
    var codes: [128]u8 = @splat(99);
    var scales: [1]f32 = .{123};
    for ([_]u16{ 0x7f80, 0xff80, 0x7fc0 }) |bad| {
        src[127] = bad;
        try std.testing.expectError(error.NonFiniteFp8Weight, quantize(&src, 1, 128, &codes, &scales));
        for (codes) |q| try std.testing.expectEqual(@as(u8, 99), q);
        try std.testing.expectEqual(@as(f32, 123), scales[0]);
    }
    try std.testing.expectError(error.BadFp8Shape, quantize(src[0..64], 1, 64, codes[0..64], &scales));
    try std.testing.expectError(error.BadFp8Shape, quantize(&src, 1, 128, codes[0..127], &scales));
    try std.testing.expectError(error.BadFp8Shape, quantize(&src, std.math.maxInt(usize), 128, &codes, &scales));
}
