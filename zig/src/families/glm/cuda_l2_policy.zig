//! Bounded, opt-in GLM weight-prefetch settings and allocation-range checks.
const std = @import("std");
pub const Mode = enum(u2) { disabled = 0, bulk = 1, lines = 2, touch = 3 };
pub const Settings = struct {
    mode: Mode = .disabled,
    bytes: usize = 8 << 20,
    rows: usize = 64,

    pub fn parse(mode: ?[]const u8, mb: ?[]const u8, rows: ?[]const u8) !Settings {
        var s: Settings = .{};
        if (mode) |v| {
            if (std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "bulk")) s.mode = .bulk else if (std.mem.eql(u8, v, "lines")) s.mode = .lines else if (std.mem.eql(u8, v, "touch")) s.mode = .touch else if (!std.mem.eql(u8, v, "0") and !std.mem.eql(u8, v, "off")) return error.BadGlmL2Prefetch;
        }
        if (mb) |v| {
            const n = std.fmt.parseInt(usize, v, 10) catch return error.BadGlmL2Budget;
            if (n == 0 or n > 64) return error.BadGlmL2Budget;
            s.bytes = n << 20;
        }
        if (rows) |v| {
            s.rows = std.fmt.parseInt(usize, v, 10) catch return error.BadGlmL2Rows;
            if (s.rows == 0 or s.rows > 64) return error.BadGlmL2Rows;
        }
        return s;
    }

    pub fn fromEnv() !Settings {
        return parse(env("TF_GLM_L2PF"), env("TF_GLM_L2PF_MB"), env("TF_GLM_L2PF_ROWS"));
    }
    fn env(name: [*:0]const u8) ?[]const u8 {
        return if (std.c.getenv(name)) |p| std.mem.span(p) else null;
    }
};

pub fn extent(base: u64, len: usize, ptr: u64, budget: usize) usize {
    if (ptr == 0 or ptr < base or ptr - base >= len) return 0;
    return @min(len - @as(usize, @intCast(ptr - base)), budget);
}

pub fn eligible(s: Settings, rows: usize, prompt: bool, capturing: bool) bool {
    return s.mode != .disabled and rows > 0 and rows <= s.rows and !prompt and !capturing;
}

test "prefetch settings are explicit, bounded and normalize aliases" {
    try std.testing.expectEqual(Mode.disabled, (try Settings.parse(null, null, null)).mode);
    try std.testing.expectEqualDeep(try Settings.parse("1", "8", "64"), try Settings.parse("bulk", null, null));
    try std.testing.expectEqual(Mode.disabled, (try Settings.parse("off", null, null)).mode);
    try std.testing.expectEqual(Mode.lines, (try Settings.parse("lines", null, null)).mode);
    try std.testing.expectEqual(Mode.touch, (try Settings.parse("touch", null, null)).mode);
    try std.testing.expectError(error.BadGlmL2Prefetch, Settings.parse("unknown", null, null));
    for ([_][]const u8{ "0", "65", "-1", "1.5", "" }) |v| {
        try std.testing.expectError(error.BadGlmL2Budget, Settings.parse("0", v, null));
        try std.testing.expectError(error.BadGlmL2Rows, Settings.parse("0", null, v));
    }
}

test "prefetch never exceeds owned allocation or budget and skips prompt/capture" {
    try std.testing.expectEqual(@as(usize, 8), extent(256, 16, 264, 64));
    try std.testing.expectEqual(@as(usize, 4), extent(256, 16, 264, 4));
    for ([_]u64{ 0, 255, 272, std.math.maxInt(u64) }) |p|
        try std.testing.expectEqual(@as(usize, 0), extent(256, 16, p, 64));
    const s = try Settings.parse("1", null, "16");
    try std.testing.expect(eligible(s, 16, false, false));
    try std.testing.expect(!eligible(s, 17, false, false));
    try std.testing.expect(!eligible(s, 1, true, false));
    try std.testing.expect(!eligible(s, 1, false, true));
}
