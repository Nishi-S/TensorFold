//! Side-stream L2 owner: bounded prepared pointers, repeated joins, capture skipping and teardown with pending reads.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Prefetch = glm.forward.l2_prefetch.Prefetch;

pub fn run(gpu: check.Gpu) !void {
    var main = try cuda.Stream.init(gpu.d, true);
    defer main.deinit();
    var original: [4355]u8 = undefined;
    for (&original, 0..) |*v, i| v.* = @truncate(i *% 13 +% 7);
    var buffer = try cuda.DeviceBuffer.fromHost(gpu.d, &original);
    defer buffer.free();
    var layers: [1]glm.weights.Layer = undefined;
    var w: glm.weights.Weights = undefined;
    w.layers = &layers;
    w.head = .{ .bf16 = buffer.ptr + original.len - 17 };
    w.buffers = .empty;
    defer w.buffers.deinit(gpu.gpa);
    try w.buffers.append(gpu.gpa, buffer);
    for ([_]glm.forward.l2_prefetch.policy.Mode{ .bulk, .lines, .touch }) |mode| {
        var p = try Prefetch.init(gpu.d, .{ .mode = mode, .bytes = 256, .rows = 16 });
        defer p.deinit();
        try p.prepare(gpu.gpa, &w);
        try check.expect(p.sites.len == 1 and p.sites[0].bytes == 17, "glm-l2: range ends inside owned allocation", .{});
        try std.testing.expectError(error.GlmL2AlreadyPrepared, p.prepare(gpu.gpa, &w));
        try check.expect(try p.active(main, 16, false), "glm-l2: eligible decode", .{});
        try check.expect(!try p.active(main, 17, false), "glm-l2: oversized window skipped", .{});
        try check.expect(!try p.active(main, 1, true), "glm-l2: prompt skipped", .{});
        for (0..5) |_| {
            try p.site(main, 0);
            try p.finish(main);
            try main.synchronize();
            var got: [original.len]u8 = undefined;
            try buffer.download(0, &got);
            try check.sameBytes("glm-l2: joined read-only range", &got, &original);
        }
        try cuda.graph.beginCapture(main, .thread_local);
        const during_capture = p.active(main, 1, false);
        var graph = try cuda.graph.endCapture(main);
        defer graph.deinit();
        try check.expect(!try during_capture, "glm-l2: capture skips side work", .{});
        try check.expect(try graph.nodeCount() == 0, "glm-l2: skipped capture contains no side event", .{});
        try p.site(main, 0); // deinit must drain this last unjoined borrow before buffer.free above
    }
    var got: [original.len]u8 = undefined;
    try buffer.download(0, &got);
    try check.sameBytes("glm-l2: owner teardown preserves weights", &got, &original);
    check.pass("glm-l2-prefetch: bounded ranges, five async joins, prompt/capture skips and pending-work teardown", .{});
}
