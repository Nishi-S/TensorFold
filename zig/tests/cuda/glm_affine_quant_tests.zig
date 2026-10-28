//! GPU startup packing against the frozen scalar oracle, with repeats, offset guards and explicit error propagation.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const guard = 128;
fn bf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}
fn fixture(gpu: check.Gpu, q: *const glm.affine_quant.Quantizer, s: cuda.Stream, groups: usize) !void {
    const a = gpu.gpa;
    const src = try a.alloc(u16, groups * 64);
    defer a.free(src);
    var random = std.Random.DefaultPrng.init(0xa441_2026);
    for (src) |*x| x.* = bf((random.random().float(f32) * 2 - 1) * 37);
    for (0..groups) |g| switch (g % 13) {
        0 => @memset(src[g * 64 ..][0..64], 0),
        1 => @memset(src[g * 64 ..][0..64], bf(-3)),
        2 => for (src[g * 64 ..][0..64], 0..) |*x, j| {
            x.* = bf(@floatFromInt(j % 16));
        },
        3 => for (src[g * 64 ..][0..64], 0..) |*x, j| {
            x.* = if (j % 2 == 0) 0x8000 else 0;
        },
        4 => for (src[g * 64 ..][0..64], 0..) |*x, j| {
            x.* = if (j % 2 == 0) 0 else 0x8000;
        },
        5 => for (src[g * 64 ..][0..64], 0..) |*x, j| {
            x.* = bf((@as(f32, @floatFromInt(j)) - 31) * 1e11);
        },
        6 => for (src[g * 64 ..][0..64], 0..) |*x, j| {
            x.* = @intCast(1 + j);
        },
        else => {},
    };
    const words = try a.alloc(u32, groups * 8);
    defer a.free(words);
    const scales = try a.alloc(u16, groups);
    defer a.free(scales);
    const biases = try a.alloc(u16, groups);
    defer a.free(biases);
    try glm.affine_pack_reference.quantize(src, groups, 64, words, scales, biases);
    var input = try cuda.DeviceBuffer.alloc(gpu.d, 2 * guard + src.len * 2);
    defer input.free();
    var output = try cuda.DeviceBuffer.alloc(gpu.d, 2 * guard + groups * 36);
    defer output.free();
    defer s.synchronize() catch @panic("affine GPU pack test drain failed");
    try input.fill8(0xa5, s.handle);
    try s.synchronize();
    try input.upload(guard, std.mem.sliceAsBytes(src));
    const got = try a.alloc(u8, output.len);
    defer a.free(got);
    const original = try a.alloc(u8, input.len);
    defer a.free(original);
    try input.download(0, original);
    const after = try a.alloc(u8, input.len);
    defer a.free(after);
    for (0..5) |_| {
        try output.fill8(0xa5, s.handle);
        try q.run(s, input.ptr + guard, output.ptr + guard, groups, 64);
        try output.download(0, got);
        for (got[0..guard]) |v| try check.expect(v == 0xa5, "GPU affine prefix guard", .{});
        for (got[got.len - guard ..]) |v| try check.expect(v == 0xa5, "GPU affine suffix guard", .{});
        const packed_bytes = got[guard .. got.len - guard];
        try check.sameBytes("GPU affine words", packed_bytes[0 .. groups * 32], std.mem.sliceAsBytes(words));
        try check.sameBytes("GPU affine scales", packed_bytes[groups * 32 ..][0 .. groups * 2], std.mem.sliceAsBytes(scales));
        try check.sameBytes("GPU affine biases", packed_bytes[groups * 34 ..], std.mem.sliceAsBytes(biases));
        try input.download(0, after);
        try check.sameBytes("GPU affine immutable input", after, original);
    }
}
fn errors(gpu: check.Gpu, q: *const glm.affine_quant.Quantizer, s: cuda.Stream) !void {
    var src: [128]u16 = @splat(0);
    var input = try cuda.DeviceBuffer.alloc(gpu.d, src.len * 2);
    defer input.free();
    var output = try cuda.DeviceBuffer.alloc(gpu.d, 72);
    defer output.free();
    defer s.synchronize() catch @panic("affine GPU error test drain failed");
    for ([_]u16{ 0x7fc0, 0x7f80, 0xff80 }) |value| {
        src[127] = value;
        try input.upload(0, std.mem.sliceAsBytes(&src));
        const got = q.run(s, input.ptr, output.ptr, 2, 64);
        if (got) |_| return error.TestFailed else |err| try check.expect(err == error.NonfiniteAffineWeight, "GPU affine rejects nonfinite", .{});
    }
    for (&src, 0..) |*v, j| v.* = bf(1e30 * (@as(f32, @floatFromInt(j % 64)) - 31));
    try input.upload(0, std.mem.sliceAsBytes(&src));
    const got = q.run(s, input.ptr, output.ptr, 2, 64);
    if (got) |_| return error.TestFailed else |err| try check.expect(err == error.AffineRangeOverflow, "GPU affine rejects overflowing loss", .{});
    src[127] = 0x7fc0;
    try input.upload(0, std.mem.sliceAsBytes(&src));
    const priority = q.run(s, input.ptr, output.ptr, 2, 64);
    if (priority) |_| return error.TestFailed else |err| try check.expect(err == error.NonfiniteAffineWeight, "GPU affine nonfinite error takes priority", .{});
}
pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var q = try glm.affine_quant.Quantizer.init(gpu.d);
    defer q.deinit();
    for ([_]usize{ 1, 3, 4, 5, 127, 257, 4096 }) |groups| try fixture(gpu, &q, s, groups);
    try errors(gpu, &q, s);
    check.pass("glm-affine-quant: frozen scalar bytes, five repeats, guards, nonfinite and overflow errors", .{});
}
