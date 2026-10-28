//! Explicit provenance for compatible eight-lane affine input sums, restricted to whole-row views.
const std = @import("std");
pub const Input = struct {
    x: u64 = 0,
    xs: u64 = 0,
    stride: usize = 0,
    k: usize = 0,
    rows: usize = 0,

    pub fn sums(in: Input, x: u64, stride: usize, k: usize, rows: usize) u64 {
        if (in.x == 0 or in.xs == 0 or in.stride == 0 or stride != in.stride or k != in.k or x < in.x) return 0;
        const row_bytes = std.math.mul(u64, stride, 2) catch return 0;
        const offset = x - in.x;
        if (offset % row_bytes != 0) return 0;
        const first = offset / row_bytes;
        if (first > in.rows or rows > in.rows - first) return 0;
        const bytes = std.math.mul(u64, k / 64, 4) catch return 0;
        const at = std.math.mul(u64, first, bytes) catch return 0;
        return std.math.add(u64, in.xs, at) catch 0;
    }
};

test "affine sums provenance permits row views and rejects stale geometry and overflow" {
    const in: Input = .{ .x = 4096, .xs = 65536, .stride = 4104, .k = 4096, .rows = 17 };
    try std.testing.expectEqual(@as(u64, 65536 + 7 * 64 * 4), in.sums(4096 + 7 * 4104 * 2, 4104, 4096, 10));
    try std.testing.expectEqual(@as(u64, 0), in.sums(4096 + 7 * 4104 * 2, 4104, 4096, 11));
    try std.testing.expectEqual(@as(u64, 0), in.sums(4098, 4104, 4096, 1));
    try std.testing.expectEqual(@as(u64, 0), in.sums(4096, 4096, 4096, 1));
    try std.testing.expectEqual(@as(u64, 0), in.sums(4096, 4104, 2048, 1));
    var overflow = in;
    overflow.xs = std.math.maxInt(u64);
    try std.testing.expectEqual(@as(u64, 0), overflow.sums(4096 + 4104 * 2, 4104, 4096, 1));
}
