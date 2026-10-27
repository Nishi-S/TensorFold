//! Opt-in policy for the native EXL3 decode prefetch experiment.
const std = @import("std");

pub fn parse(value: ?[]const u8) !bool {
    const v = value orelse return false;
    if (std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.BadExl3DecodePrefetch;
}

pub const Mode = enum(u3) { baseline, scalar, nc1, nc2, nc4 };

/// Explicit load mode overrides the legacy boolean after both spellings are validated.
pub fn resolve(value: ?[]const u8, legacy: ?[]const u8) !Mode {
    const old = try parse(legacy);
    const v = value orelse return if (old) .scalar else .baseline;
    if (std.mem.eql(u8, v, "baseline") or std.mem.eql(u8, v, "0")) return .baseline;
    if (std.mem.eql(u8, v, "scalar") or std.mem.eql(u8, v, "ldg")) return .scalar;
    if (std.mem.eql(u8, v, "nc1") or std.mem.eql(u8, v, "nc")) return .nc1;
    if (std.mem.eql(u8, v, "nc2")) return .nc2;
    if (std.mem.eql(u8, v, "nc4")) return .nc4;
    return error.BadExl3DecodeLoads;
}

pub fn symbol(mode: Mode) ?[:0]const u8 {
    return switch (mode) {
        .baseline => null,
        .scalar => "tf_exl3_grouped_1_pf1",
        .nc1 => "tf_exl3_grouped_1_nc1",
        .nc2 => "tf_exl3_grouped_1_nc2",
        .nc4 => "tf_exl3_grouped_1_nc4",
    };
}

pub fn eligible(enabled: bool, mcg: bool, uniform_k2: u32, rows: usize) bool {
    return enabled and mcg and uniform_k2 == 8 and rows >= 1 and rows <= 16;
}

test "decode prefetch is explicit and restricted to verified mcg four-bit decode windows" {
    try std.testing.expect(!try parse(null));
    try std.testing.expect(!try parse("0"));
    try std.testing.expect(try parse("1"));
    try std.testing.expectError(error.BadExl3DecodePrefetch, parse("nc"));
    for ([_]usize{ 1, 3, 8, 16 }) |rows| try std.testing.expect(eligible(true, true, 8, rows));
    for ([_]usize{ 0, 17, 64, 128 }) |rows| try std.testing.expect(!eligible(true, true, 8, rows));
    try std.testing.expect(!eligible(false, true, 8, 1));
    try std.testing.expect(!eligible(true, false, 8, 1));
    for ([_]u32{ 0, 2, 7, 9, 16 }) |k2| try std.testing.expect(!eligible(true, true, k2, 1));
}

test "load modes preserve the scalar alias and reject ambiguous spellings" {
    try std.testing.expectEqual(Mode.baseline, try resolve(null, null));
    try std.testing.expectEqual(Mode.scalar, try resolve(null, "1"));
    try std.testing.expectEqual(Mode.baseline, try resolve("baseline", "1"));
    try std.testing.expectEqual(Mode.nc4, try resolve("nc4", "0"));
    inline for (.{ "scalar", "nc1", "nc2", "nc4" }) |name|
        try std.testing.expectEqual(@field(Mode, name), try resolve(name, null));
    try std.testing.expectEqual(Mode.nc1, try resolve("nc", null));
    try std.testing.expectEqual(Mode.scalar, try resolve("ldg", null));
    try std.testing.expectError(error.BadExl3DecodeLoads, resolve("nc3", null));
    try std.testing.expectError(error.BadExl3DecodePrefetch, resolve("nc1", "true"));
}
