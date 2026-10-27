//! Opt-in ordered decode epilogues; counters belong to a call's scratch and are cleared on its consuming stream.
const std = @import("std");
const module = @import("module.zig");
const launch = @import("launch.zig");
const Stream = @import("stream.zig").Stream;

pub const Selection = struct { down: bool = false, gateup: bool = false };

/// Bit 0 fuses down for per-slot output; bit 1 fuses gate/up after every ordered split is published.
pub fn parse(value: ?[]const u8) !u2 {
    const v = value orelse return 0;
    if (v.len != 1 or v[0] < '0' or v[0] > '3') return error.BadExl3DecodeFuse;
    return @intCast(v[0] - '0');
}

pub fn select(mask: u2, mcg: bool, uniform_k2: u32, rows: usize, down_sk: usize, weighted: bool) Selection {
    if (!mcg or uniform_k2 != 8 or rows == 0 or rows > 16) return .{};
    return .{ .down = mask & 1 != 0 and down_sk == 1 and !weighted, .gateup = mask & 2 != 0 };
}

pub const Kernels = struct {
    down: ?module.Function = null,
    gateup: ?module.Function = null,

    pub fn resolve(m: module.Module, mask: u2) !Kernels {
        return .{
            .down = if (mask & 1 != 0) try m.function("tf_exl3_down_fused") else null,
            .gateup = if (mask & 2 != 0) try m.function("tf_exl3_gateup_fused") else null,
        };
    }
};

/// At most one 16-member tile for admitted decode rows; a larger capacity still reserves enough expert counters.
pub fn counterBytes(max_experts: usize, width: usize) usize {
    return max_experts * (width / 128) * 4;
}

pub fn clear(s: Stream, ptr: u64, max_experts: usize, width: usize) !void {
    try s.d.check(s.d.api.cuMemsetD32Async(ptr, 0, counterBytes(max_experts, width) / 4, s.handle), "EXL3 fuse counters");
}

/// Appended after GroupedArgs: its 124 payload bytes pad to 128 at the first pointer of this embedded struct.
pub fn append(a: *launch.Args, sv0: u64, sv1: u64, su2: u64, out: u64, done: u64, limit: f32, act_mode: i32, load_mode: i32) void {
    a.add(sv0);
    a.add(sv1);
    a.add(su2);
    a.add(out);
    a.add(done);
    a.add(limit);
    a.add(act_mode);
    a.add(load_mode);
}

test "fusion bits refuse unknown values and preserve unsupported combinations" {
    try std.testing.expectEqual(@as(u2, 0), try parse(null));
    for ([_][]const u8{ "0", "1", "2", "3" }, 0..) |v, i|
        try std.testing.expectEqual(@as(u2, @intCast(i)), try parse(v));
    for ([_][]const u8{ "", "4", "-1", "on", "01" }) |v|
        try std.testing.expectError(error.BadExl3DecodeFuse, parse(v));
    const both = select(3, true, 8, 16, 1, false);
    try std.testing.expect(both.down and both.gateup);
    try std.testing.expect(!select(3, true, 8, 1, 1, true).down);
    try std.testing.expect(select(3, true, 8, 1, 1, true).gateup);
    try std.testing.expect(!select(3, true, 8, 1, 2, false).down);
    for ([_]usize{ 0, 17, 128 }) |r| try std.testing.expectEqual(Selection{}, select(3, true, 8, r, 1, false));
    try std.testing.expectEqual(Selection{}, select(3, false, 8, 1, 1, false));
    try std.testing.expectEqual(Selection{}, select(3, true, 0, 1, 1, false));
}

test "fused argument tail starts at the C struct boundary" {
    var a: launch.Args = .{};
    for (0..10) |_| a.add(@as(u64, 0));
    for (0..11) |_| a.add(@as(i32, 0));
    try std.testing.expectEqual(@as(usize, 124), a.used);
    append(&a, 11, 12, 13, 14, 15, 1.5, 0, 4);
    try std.testing.expectEqual(@as(u16, 128), a.offsets[21]);
    try std.testing.expectEqual(@as(u16, 176), a.offsets[28]);
    try std.testing.expectEqual(@as(usize, 180), a.used);
}
