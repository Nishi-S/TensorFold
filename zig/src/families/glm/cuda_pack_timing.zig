//! Opt-in startup packing wall times; disabled mode reads no clock and changes no stream synchronization.
const std = @import("std");
const platform = @import("sys");

pub const Phase = enum { setup, readback, affine_quant, symmetric_quant, fp8_quant, upload_tile, compact, bf16_tile };

pub const Timing = struct {
    enabled: bool = false,
    valid: bool = true,
    last: u64 = 0,
    ns: [@typeInfo(Phase).@"enum".field_names.len]u64 = @splat(0),

    pub fn init() Timing {
        const value = platform.getenv("TF_GLM_PACK_TIMING") orelse return .{};
        if (!std.mem.eql(u8, value, "1")) return .{};
        const start = now() orelse return .{ .enabled = true, .valid = false };
        return .{ .enabled = true, .last = start };
    }

    pub fn mark(t: *Timing, phase: Phase) void {
        if (!t.enabled or !t.valid) return;
        t.record(phase, now());
    }

    fn record(t: *Timing, phase: Phase, end_: ?u64) void {
        const end = end_ orelse {
            t.valid = false;
            return;
        };
        if (end < t.last) {
            t.valid = false;
            return;
        }
        t.ns[@backingInt(phase)] +|= end - t.last;
        t.last = end;
    }

    pub fn report(t: *const Timing, rank: usize, linears: usize) void {
        if (!t.enabled) return;
        if (!t.valid) {
            std.log.warn("glm pack timing: rank={d} unavailable monotonic clock", .{rank});
            return;
        }
        var total: u64 = 0;
        for (t.ns) |n| total +|= n;
        std.log.info("glm pack timing: rank={d} linears={d} total_ms={d:.3}", .{ rank, linears, ms(total) });
        inline for (@typeInfo(Phase).@"enum".field_names, 0..) |name, i|
            std.log.info("glm pack timing: rank={d} phase={s} wall_ms={d:.3}", .{ rank, name, ms(t.ns[i]) });
    }
};

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

fn now() ?u64 {
    var ts: platform.c.timespec = undefined;
    if (platform.c.clock_gettime(platform.c.CLOCK.MONOTONIC, &ts) != 0 or ts.sec < 0 or ts.nsec < 0) return null;
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

test "packing phases accumulate elapsed time and reject broken clocks" {
    var timing: Timing = .{ .enabled = true, .last = 10 };
    timing.record(.readback, 20);
    timing.record(.affine_quant, 50);
    timing.record(.readback, 70);
    try std.testing.expectEqual(@as(u64, 30), timing.ns[@backingInt(Phase.readback)]);
    try std.testing.expectEqual(@as(u64, 30), timing.ns[@backingInt(Phase.affine_quant)]);
    timing.record(.compact, 69);
    try std.testing.expect(!timing.valid);
    var failed: Timing = .{ .enabled = true, .last = 10 };
    failed.record(.setup, null);
    try std.testing.expect(!failed.valid);
    var disabled: Timing = .{};
    disabled.mark(.setup);
    try std.testing.expectEqualDeep(Timing{}, disabled);
}
