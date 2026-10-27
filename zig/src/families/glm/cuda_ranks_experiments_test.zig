//! Experimental GLM settings are validated and agreed before loading on every rank.
const std = @import("std");
const ranks = @import("cuda_ranks.zig");
const Settings = ranks.Settings;
const Fingerprint = ranks.Fingerprint;
const Mismatch = ranks.Mismatch;
const mismatch = ranks.mismatch;

test "prompt K experiment is explicit and all ranks must agree" {
    const Reader = struct {
        fn absent(_: [:0]const u8) ?[]const u8 {
            return null;
        }
        fn one(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PROMPT_K")) "one" else null;
        }
        fn untiled(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PROMPT_K")) "shape" else if (std.mem.eql(u8, key, "TF_GLM_MM")) "0" else null;
        }

        fn shape(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PROMPT_K")) "shape" else null;
        }
        fn bad(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_PROMPT_K")) "auto" else null;
        }
    };
    var problem: []const u8 = "";
    const selected = try Settings.read(Reader.shape, &problem);
    try std.testing.expect(selected.prompt_fixed_k);
    try std.testing.expect(!(try Settings.read(Reader.absent, &problem)).prompt_fixed_k);
    try std.testing.expect(!(try Settings.read(Reader.one, &problem)).prompt_fixed_k);
    try std.testing.expectError(error.BadSetting, Settings.read(Reader.untiled, &problem));
    try std.testing.expectError(error.BadSetting, Settings.read(Reader.bad, &problem));
    var all = [_]Fingerprint{ Fingerprint.of(2, 0, .{}, 8192, true, @splat(0), @splat(0), @splat(0)), Fingerprint.of(2, 1, selected, 8192, true, @splat(0), @splat(0), @splat(0)) };
    try std.testing.expectEqual(Fingerprint.Field.prompt_fixed_k, mismatch(&all).?.field);
    all[0].prompt_fixed_k = 1;
    try std.testing.expectEqual(@as(?Mismatch, null), mismatch(&all));
}

test "decode experiment flags reject typos and disagree across ranks in either direction" {
    inline for (.{ .{ "TF_EXL3_DECODE_PREFETCH", "exl3_prefetch" }, .{ "TF_GLM_Q4_PAIR_DECODE", "q4_pair_decode" }, .{ "TF_GLM_AFFINE_PRUNE", "affine_prune" }, .{ "TF_GLM_PACKED_PROMPT_BATCH", "packed_prompt_batch" } }) |entry| {
        const Reader = struct {
            var value: ?[]const u8 = null;
            fn get(key: [:0]const u8) ?[]const u8 {
                return if (std.mem.eql(u8, key, entry[0])) value else null;
            }
        };
        var problem: []const u8 = "";
        for ([_]?[]const u8{ null, "0" }) |value| {
            Reader.value = value;
            try std.testing.expect(!@field(try Settings.read(Reader.get, &problem), entry[1]));
        }
        for ([_][]const u8{ "", "true", "nc", "2", "01" }) |value| {
            Reader.value = value;
            try std.testing.expectError(error.BadSetting, Settings.read(Reader.get, &problem));
            try std.testing.expectEqualStrings(entry[0] ++ ": 0 or 1", problem);
        }
        Reader.value = "1";
        const selected = try Settings.read(Reader.get, &problem);
        try std.testing.expect(@field(selected, entry[1]));
        const base = Fingerprint.of(2, 0, .{}, 8192, true, @splat(0), @splat(0), @splat(0));
        const chosen = Fingerprint.of(2, 1, selected, 8192, true, @splat(0), @splat(0), @splat(0));
        try std.testing.expectEqual(@as(u32, 16 << 8 | 16), base.bits);
        try std.testing.expectEqual(base.bits, chosen.bits & 0xffff);
        try std.testing.expectEqual(base.layout, chosen.layout);
        var all = [_]Fingerprint{ base, chosen };
        try std.testing.expectEqual(Mismatch{ .field = .bits, .rank = 1 }, mismatch(&all).?);
        all[0].bits = chosen.bits;
        try std.testing.expect(mismatch(&all) == null);
        all[1].bits = base.bits;
        try std.testing.expectEqual(Mismatch{ .field = .bits, .rank = 1 }, mismatch(&all).?);
    }
}

test "the two experiment choices have independent fingerprint bits" {
    var all = [_]Fingerprint{
        Fingerprint.of(2, 0, .{ .exl3_prefetch = true }, 8192, true, @splat(0), @splat(0), @splat(0)),
        Fingerprint.of(2, 1, .{ .q4_pair_decode = true }, 8192, true, @splat(0), @splat(0), @splat(0)),
    };
    try std.testing.expectEqual(Mismatch{ .field = .bits, .rank = 1 }, mismatch(&all).?);
    const both = Fingerprint.of(2, 0, .{ .exl3_prefetch = true, .q4_pair_decode = true, .dense_bits = 4, .head_bits = 8 }, 8192, true, @splat(0), @splat(0), @splat(0));
    try std.testing.expectEqual(@as(u32, 4 << 8 | 8), both.bits & 0xffff);
    all[0] = both;
    all[1] = both;
    all[1].rank = 1;
    try std.testing.expect(mismatch(&all) == null);
}

const MiaEnv = struct {
    var pairs: []const struct { []const u8, []const u8 } = &.{};
    fn get(key: [:0]const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], key)) return p[1];
        return null;
    }
};

test "Mia formats require matching widths and tiled execution before any allocation" {
    var problem: []const u8 = "";
    defer MiaEnv.pairs = &.{};
    MiaEnv.pairs = &.{.{ "TF_GLM_DENSE_FORMAT", "affine4" }};
    try std.testing.expectError(error.BadSetting, Settings.read(MiaEnv.get, &problem));
    MiaEnv.pairs = &.{.{ "TF_GLM_HEAD_FORMAT", "fp8" }};
    try std.testing.expectError(error.BadSetting, Settings.read(MiaEnv.get, &problem));
    MiaEnv.pairs = &.{ .{ "TF_GLM_DENSE_FORMAT", "affine4" }, .{ "TF_GLM_DENSE_BITS", "4" }, .{ "TF_GLM_MM", "0" } };
    try std.testing.expectError(error.BadSetting, Settings.read(MiaEnv.get, &problem));
    MiaEnv.pairs = &.{ .{ "TF_GLM_DENSE_FORMAT", "fp8" }, .{ "TF_GLM_DENSE_BITS", "8" } };
    try std.testing.expectError(error.BadSetting, Settings.read(MiaEnv.get, &problem));
    MiaEnv.pairs = &.{ .{ "TF_GLM_HEAD_FORMAT", "affine4" }, .{ "TF_GLM_HEAD_BITS", "4" } };
    try std.testing.expectError(error.BadSetting, Settings.read(MiaEnv.get, &problem));
    MiaEnv.pairs = &.{
        .{ "TF_GLM_DENSE_FORMAT", "affine4" }, .{ "TF_GLM_DENSE_BITS", "4" },
        .{ "TF_GLM_HEAD_FORMAT", "fp8" },      .{ "TF_GLM_HEAD_BITS", "8" },
        .{ "TF_EXL3_DECODE_LOADS", "nc4" },    .{ "TF_EXL3_DECODE_FUSE", "3" },
    };
    const selected = try Settings.read(MiaEnv.get, &problem);
    try std.testing.expect(selected.dense_format == .affine4 and selected.head_format == .fp8);
    try std.testing.expectEqual(@as(u3, 4), selected.exl3_loads);
    try std.testing.expectEqual(@as(u2, 3), selected.exl3_fuse);
    var all = [_]Fingerprint{
        Fingerprint.of(2, 0, selected, 8192, true, @splat(0), @splat(0), @splat(0)),
        Fingerprint.of(2, 1, selected, 8192, true, @splat(0), @splat(0), @splat(0)),
    };
    try std.testing.expect(mismatch(&all) == null);
    inline for (.{ "dense_format", "head_format", "exl3_loads", "exl3_fuse" }) |field| {
        var other = selected;
        @field(other, field) = @field(@as(Settings, .{}), field);
        all[1] = Fingerprint.of(2, 1, other, 8192, true, @splat(0), @splat(0), @splat(0));
        try std.testing.expectEqual(Fingerprint.Field.bits, mismatch(&all).?.field);
    }
}

test "load aliases normalize to the same rank fingerprint and explicit baseline overrides legacy" {
    var problem: []const u8 = "";
    defer MiaEnv.pairs = &.{};
    MiaEnv.pairs = &.{.{ "TF_EXL3_DECODE_PREFETCH", "1" }};
    const legacy = try Settings.read(MiaEnv.get, &problem);
    MiaEnv.pairs = &.{.{ "TF_EXL3_DECODE_LOADS", "scalar" }};
    const scalar = try Settings.read(MiaEnv.get, &problem);
    var all = [_]Fingerprint{
        Fingerprint.of(2, 0, legacy, 8192, true, @splat(0), @splat(0), @splat(0)),
        Fingerprint.of(2, 1, scalar, 8192, true, @splat(0), @splat(0), @splat(0)),
    };
    try std.testing.expect(mismatch(&all) == null);
    MiaEnv.pairs = &.{ .{ "TF_EXL3_DECODE_LOADS", "baseline" }, .{ "TF_EXL3_DECODE_PREFETCH", "1" } };
    const explicit = try Settings.read(MiaEnv.get, &problem);
    try std.testing.expect(!explicit.exl3_prefetch and explicit.exl3_loads == 0);
    for ([_][]const u8{ "4", "-1", "03", "yes" }) |bad| {
        MiaEnv.pairs = &.{.{ "TF_EXL3_DECODE_FUSE", bad }};
        try std.testing.expectError(error.BadSetting, Settings.read(MiaEnv.get, &problem));
    }
}

test "L2 budgets are validated even disabled and every rank agrees on bytes and row limit" {
    defer MiaEnv.pairs = &.{};
    var problem: []const u8 = "";
    for ([_][]const u8{ "0", "65", "-1", "128", "abc" }) |bad| {
        MiaEnv.pairs = &.{.{ "TF_GLM_L2PF_MB", bad }};
        try std.testing.expectError(error.BadSetting, Settings.read(MiaEnv.get, &problem));
        MiaEnv.pairs = &.{.{ "TF_GLM_L2PF_ROWS", bad }};
        try std.testing.expectError(error.BadSetting, Settings.read(MiaEnv.get, &problem));
    }
    MiaEnv.pairs = &.{ .{ "TF_GLM_L2PF", "lines" }, .{ "TF_GLM_L2PF_MB", "16" }, .{ "TF_GLM_L2PF_ROWS", "8" } };
    const s = try Settings.read(MiaEnv.get, &problem);
    try std.testing.expect(s.l2_prefetch and s.l2_mb == 16 and s.l2_rows == 8);
    var all = [_]Fingerprint{
        Fingerprint.of(2, 0, s, 8192, true, @splat(0), @splat(0), @splat(0)),
        Fingerprint.of(2, 1, s, 8192, true, @splat(0), @splat(0), @splat(0)),
    };
    try std.testing.expect(mismatch(&all) == null);
    inline for (.{ "l2_prefetch", "l2_mb", "l2_rows" }) |field| {
        var other = s;
        @field(other, field) = @field(@as(Settings, .{}), field);
        all[1] = Fingerprint.of(2, 1, other, 8192, true, @splat(0), @splat(0), @splat(0));
        try std.testing.expectEqual(Fingerprint.Field.prefetch, mismatch(&all).?.field);
    }
}

test "packing and prompt scheduling experiments use independent rank bits" {
    const a = Fingerprint.of(2, 0, .{ .affine_prune = true }, 8192, true, @splat(0), @splat(0), @splat(0));
    const b = Fingerprint.of(2, 1, .{ .packed_prompt_batch = true }, 8192, true, @splat(0), @splat(0), @splat(0));
    const all = [_]Fingerprint{ a, b };
    try std.testing.expectEqual(Fingerprint.Field.bits, mismatch(&all).?.field);
    try std.testing.expectEqual(@as(u32, 16 << 8 | 16), a.bits & 0x07ffffff);
    try std.testing.expectEqual(@as(u32, 16 << 8 | 16), b.bits & 0x07ffffff);
}
