//! Independent host witness that the HC contract's dot and square inputs distinguish accumulator precision.
const std = @import("std");
const Dense = @import("dense.zig").Dense;
const rows = @import("rows.zig");
const c = @import("contract.zig");

test "HC fast stress dots and squares stay exact in FP32 and reject reduced accumulators" {
    const d = try Dense.init(std.testing.allocator, .{ .rows = 16, .n = 24, .k = 1024 }, 1);
    defer d.deinit();
    d.fillX(.stress, 7, 2);
    var term: [1024]f64 = undefined;
    for ([_]rows.Order{ .sequential, .pairwise, .strided }) |order| {
        var worst = [2]f64{ 0, 0 };
        for (0..16) |r| for (0..25) |m| {
            for (&term, 0..) |*v, k| {
                const x = c.fromBf16(d.x[r * 1024 + k]);
                const w: f64 = @as(f64, @floatFromInt(1 + ((k % 512) * 7 + m) % 6)) * (if (k < 512) @as(f64, 1) else -1);
                v.* = x * (if (m == 24) x else w);
            }
            var exact: f64 = 0;
            for (term) |v| exact += v;
            try std.testing.expectEqual(exact, rows.sumIn(.f32, order, &term));
            for ([_]c.Accum{ .f16, .bf16 }, &worst) |acc, *bad| {
                const got = rows.sumIn(acc, order, &term);
                bad.* = @max(bad.*, if (std.math.isFinite(got)) @abs(got - exact) / (@abs(exact) * 0x1p-24 + 0x1p-149) else std.math.inf(f64));
            }
        };
        try std.testing.expect(worst[0] >= 100 and worst[1] >= 100);
    }
}
