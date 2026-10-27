//! Explicit Mia-port formats and decode experiments, validated before rank agreement and weight allocation.
const std = @import("std");
const cuda = @import("cuda");
const mm = @import("cuda_mm.zig");

pub fn read(s: anytype, get: *const fn ([:0]const u8) ?[]const u8, problem: *[]const u8) !void {
    inline for (.{ .{ "TF_EXL3_DECODE_PREFETCH", "exl3_prefetch" }, .{ "TF_GLM_Q4_PAIR_DECODE", "q4_pair_decode" }, .{ "TF_GLM_AFFINE_PRUNE", "affine_prune" }, .{ "TF_GLM_PACKED_PROMPT_BATCH", "packed_prompt_batch" } }) |e| {
        if (get(e[0])) |v| {
            if (std.mem.eql(u8, v, "1")) @field(s, e[1]) = true else if (!std.mem.eql(u8, v, "0")) {
                problem.* = e[0] ++ ": 0 or 1";
                return error.BadSetting;
            }
        }
    }
    const mode = cuda.exl3_experts.prefetch.resolve(get("TF_EXL3_DECODE_LOADS"), get("TF_EXL3_DECODE_PREFETCH")) catch {
        problem.* = "TF_EXL3_DECODE_LOADS: baseline, scalar, nc1, nc2 or nc4";
        return error.BadSetting;
    };
    s.exl3_loads = @backingInt(mode);
    s.exl3_prefetch = mode == .scalar;
    if (get("TF_GLM_DENSE_FORMAT")) |v| {
        s.dense_format = if (std.mem.eql(u8, v, "symmetric")) .symmetric else if (std.mem.eql(u8, v, "affine4")) .affine4 else {
            problem.* = "TF_GLM_DENSE_FORMAT: symmetric or affine4";
            return error.BadSetting;
        };
    }
    if (get("TF_GLM_HEAD_FORMAT")) |v| {
        s.head_format = if (std.mem.eql(u8, v, "symmetric")) .symmetric else if (std.mem.eql(u8, v, "fp8")) .fp8 else {
            problem.* = "TF_GLM_HEAD_FORMAT: symmetric or fp8";
            return error.BadSetting;
        };
    }
    if ((s.dense_format != .symmetric or s.head_format != .symmetric) and !s.tiled) {
        problem.* = "Mia packed formats require TF_GLM_MM=1";
        return error.BadSetting;
    }
    if (s.dense_format == .affine4 and s.dense_bits != 4) {
        problem.* = "TF_GLM_DENSE_FORMAT=affine4 requires TF_GLM_DENSE_BITS=4";
        return error.BadSetting;
    }
    if (s.head_format == .fp8 and s.head_bits != 8) {
        problem.* = "TF_GLM_HEAD_FORMAT=fp8 requires TF_GLM_HEAD_BITS=8";
        return error.BadSetting;
    }
    if (get("TF_EXL3_DECODE_FUSE")) |v| {
        s.exl3_fuse = std.fmt.parseInt(u2, v, 10) catch {
            problem.* = "TF_EXL3_DECODE_FUSE: 0, 1, 2 or 3";
            return error.BadSetting;
        };
        if (v.len != 1) {
            problem.* = "TF_EXL3_DECODE_FUSE: 0, 1, 2 or 3";
            return error.BadSetting;
        }
    }
    if (get("TF_GLM_L2PF")) |v| {
        s.l2_mode = if (std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off")) 0 else if (std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "bulk")) 1 else if (std.mem.eql(u8, v, "lines")) 2 else if (std.mem.eql(u8, v, "touch")) 3 else {
            problem.* = "TF_GLM_L2PF: 0, off, 1, bulk, lines or touch";
            return error.BadSetting;
        };
        s.l2_prefetch = s.l2_mode != 0;
    }
    inline for (.{ .{ "TF_GLM_L2PF_MB", "l2_mb" }, .{ "TF_GLM_L2PF_ROWS", "l2_rows" } }) |e| {
        if (get(e[0])) |v| {
            const n = std.fmt.parseInt(u7, v, 10) catch 0;
            if (n < 1 or n > 64) {
                problem.* = e[0] ++ ": 1 to 64";
                return error.BadSetting;
            }
            @field(s, e[1]) = n;
        }
    }
}

pub fn bits(s: anytype) u32 {
    const mode: u32 = if (s.exl3_loads == 0 and s.exl3_prefetch) 1 else s.exl3_loads;
    return @as(u32, s.dense_bits) << 8 | s.head_bits |
        (if (mode == 1) @as(u32, 1) << 16 else mode << 18) |
        @as(u32, @intFromBool(s.q4_pair_decode)) << 17 |
        @as(u32, @backingInt(s.dense_format)) << 21 |
        @as(u32, @backingInt(s.head_format)) << 23 |
        @as(u32, s.exl3_fuse) << 25 |
        @as(u32, @intFromBool(s.affine_prune)) << 27 |
        @as(u32, @intFromBool(s.packed_prompt_batch)) << 28;
}

pub const Format = mm.Format;

fn l2Mode(s: anytype) u32 {
    return if (!s.l2_prefetch) 0 else if (s.l2_mode == 0) 1 else s.l2_mode;
}

pub fn l2Fingerprint(s: anytype) u32 {
    return @as(u32, s.l2_rows) << 9 | @as(u32, s.l2_mb) << 2 | l2Mode(s);
}

pub fn l2Settings(s: anytype) @import("cuda_l2_policy.zig").Settings {
    return .{ .mode = @fromBackingInt(@intCast(l2Mode(s))), .bytes = @as(usize, s.l2_mb) << 20, .rows = s.l2_rows };
}
