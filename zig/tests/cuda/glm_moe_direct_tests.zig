//! Compact route selection and direct shared combine retain the original slot arithmetic and guard bytes.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Rig = @import("quant_ref_tests.zig").RigT;

pub fn run(r: *Rig, set: glm.glue.Set) !void {
    for ([_]usize{ 1, 17, 65 }) |rows| {
        const ne = 512;
        const logits = try r.a.alloc(f32, rows * ne);
        const bias = try r.a.alloc(f32, ne);
        for (logits, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 29)) / 13 - 1;
        @memset(bias, 0);
        const dl = try r.dev(std.mem.sliceAsBytes(logits));
        const db = try r.dev(std.mem.sliceAsBytes(bias));
        const old_pick = try r.zeros(rows * 9 * 4);
        const old_wts = try r.zeros(rows * 9 * 4);
        const new_pick = try r.zeros(rows * 8 * 4 + 16);
        const new_wts = try r.zeros(rows * 9 * 4 + 16);
        for ([_]bool{ false, true }) |norm| {
            try glm.glue.select(set, r.s, dl, db, old_pick, old_wts, rows, 8, ne, 2.5, norm);
            const old = try r.back(old_pick, rows * 9 * 4);
            for (0..5) |_| {
                try glm.glue.selectRouted(set, r.s, dl, db, new_pick, new_wts, rows, 8, ne, 2.5, norm);
                const compact = try r.back(new_pick, rows * 8 * 4 + 16);
                for (0..rows) |row| try check.sameBytes("compact routed picks", compact[row * 32 ..][0..32], old[row * 36 ..][0..32]);
                try check.sameBytes("compact route weights", try r.back(new_wts, rows * 9 * 4), try r.back(old_wts, rows * 9 * 4));
                try check.sameBytes("compact pick guard", compact[rows * 8 * 4 ..], &@as([16]u8, @splat(0xff)));
                try check.sameBytes("compact weight guard", try r.back(new_wts + rows * 9 * 4, 16), &@as([16]u8, @splat(0xff)));
            }
        }
        for ([_]usize{ 8, 1024, 1032 }) |d| {
            const routed = try r.a.alloc(f32, rows * 8 * d);
            const shared = try r.a.alloc(f32, rows * d);
            const expanded = try r.a.alloc(f32, rows * 9 * d);
            for (routed, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 127)) - 63)) / 32;
            for (shared, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 11)) / 7;
            for (0..rows) |row| {
                @memcpy(expanded[row * 9 * d ..][0 .. 8 * d], routed[row * 8 * d ..][0 .. 8 * d]);
                @memcpy(expanded[(row * 9 + 8) * d ..][0..d], shared[row * d ..][0..d]);
            }
            const dr = try r.dev(std.mem.sliceAsBytes(routed));
            const ds = try r.dev(std.mem.sliceAsBytes(shared));
            const de = try r.dev(std.mem.sliceAsBytes(expanded));
            const old_out = try r.zeros(rows * d * 4);
            const new_out = try r.zeros(rows * d * 4 + 16);
            try glm.glue.combine(set, r.s, de, true, old_wts, old_out, rows, d, 9);
            for (0..5) |_| {
                try glm.glue.combineShared(set, r.s, dr, ds, old_wts, new_out, rows, d, 8);
                try check.sameBytes("direct shared combine", try r.back(new_out, rows * d * 4), try r.back(old_out, rows * d * 4));
                try check.sameBytes("direct combine guard", try r.back(new_out + rows * d * 4, 16), &@as([16]u8, @splat(0xff)));
            }
            try check.sameBytes("direct routed input retained", try r.back(dr, routed.len * 4), std.mem.sliceAsBytes(routed));
            try check.sameBytes("direct shared input retained", try r.back(ds, shared.len * 4), std.mem.sliceAsBytes(shared));
        }
    }
    check.pass("glm-moe-direct compact picks and shared combine byte-equal", .{});
}
