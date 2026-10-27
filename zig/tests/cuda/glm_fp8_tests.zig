//! Weight-only FP8: independent exact dyadic products and row/stride/prompt consistency through native dispatch.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const codec = @import("core").exlr8.codec;
const check = @import("check.zig");
const capacity = 261;
const guard = 128;
const marker = 0xa5;

fn bf(value: f32) u16 {
    const u: u32 = @bitCast(value);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn scalar(code: u16) f64 {
    return @as(f32, @bitCast(@as(u32, code) << 16));
}

fn shape(gpu: check.Gpu, module: cuda.Module, n: usize, k: usize) !void {
    const a = gpu.gpa;
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var kernels = try glm.mm.Kernels.load(module);
    try kernels.loadFp8(module);
    kernels.packed_prompt_batch = true;
    const stride = k + 8;
    const x = try a.alloc(u16, capacity * stride);
    defer a.free(x);
    @memset(x, 0x7fc0);
    for (0..capacity) |r| for (0..k) |i| {
        x[r * stride + i] = bf(@as(f32, @floatFromInt(@as(i32, @intCast((r * 5 + i * 3) % 17)) - 8)) / 16);
    };
    const codes = try a.alloc(u8, n * k);
    defer a.free(codes);
    const scales = try a.alloc(f32, n * k / 128);
    defer a.free(scales);
    for (0..n) |c| {
        for (0..k) |i| codes[c * k + i] = codec.e4m3FromF32(@floatFromInt(@as(i32, @intCast((c * 7 + i * 11) % 17)) - 8));
        for (0..k / 128) |g| scales[c * (k / 128) + g] = if ((c + g) % 2 == 0) 0.125 else 0.0625;
    }
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
    defer dx.free();
    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, codes);
    defer dw.free();
    var ds = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(scales));
    defer ds.free();
    const weight: glm.mm.Tiled = .{ .w = dw.ptr, .s = ds.ptr, .n = n, .k = k, .bits = 8, .format = .fp8 };
    var part = try cuda.DeviceBuffer.alloc(gpu.d, @max(16, glm.mm.partBytes(capacity, weight)));
    defer part.free();
    var output = try cuda.DeviceBuffer.alloc(gpu.d, 2 * guard + capacity * n * 4);
    defer output.free();
    defer stream.synchronize() catch @panic("FP8 fixture keeps device buffers until completion");
    const got = try a.alloc(u8, output.len);
    defer a.free(got);
    const expected = try a.alloc(f64, capacity * n);
    defer a.free(expected);
    for (0..capacity) |r| for (0..n) |c| {
        var sum: f64 = 0;
        for (0..k) |i| sum += scalar(x[r * stride + i]) * @as(f64, codec.e4m3Value(codes[c * k + i])) * @as(f64, scales[c * (k / 128) + i / 128]);
        expected[r * n + c] = sum;
    };
    for ([_]bool{ false, true }) |pdl| {
        kernels.pdl = pdl;
        for ([_]bool{ false, true }) |wide| {
            const size: usize = if (wide) 4 else 2;
            for ([_]bool{ false, true }) |prompt| for ([_]usize{ 1, 3, 16, 17, 33, 64, 126, 127, 128, 129, 254, 255, 256, 257 }) |rows| {
                for ([_]usize{ 0, 1 }) |offset| for (0..5) |_| {
                    try output.fill8(marker, stream.handle);
                    try glm.mm.matmul(&kernels, stream, dx.ptr + offset * stride * 2, stride, weight, output.ptr + guard, part.ptr, rows, wide, prompt);
                    try stream.synchronize();
                    try output.download(0, got);
                    for (got[0..guard]) |v| try check.expect(v == marker, "FP8 output prefix guard", .{});
                    for (got[guard + rows * n * size ..]) |v| try check.expect(v == marker, "FP8 output tail guard", .{});
                    for (0..rows * n) |i| {
                        const want: f32 = @floatCast(expected[offset * n + i]);
                        const at = guard + i * size;
                        if (wide) {
                            const bits = std.mem.readInt(u32, got[at..][0..4], .little);
                            try check.expect(bits == @as(u32, @bitCast(want)), "FP8 FP32 equals independent exact dyadic dot", .{});
                        } else {
                            const bits = std.mem.readInt(u16, got[at..][0..2], .little);
                            try check.expect(bits == bf(want), "FP8 BF16 equals rounded independent exact dyadic dot", .{});
                        }
                    }
                };
            };
        }
    }
    const copied = try a.alloc(u8, dx.len);
    defer a.free(copied);
    try dx.download(0, copied);
    try check.sameBytes("FP8 input and NaN stride padding unchanged", copied, std.mem.sliceAsBytes(x));
}

// Rounded sums expose prompt split-K and bucket-dependent MMA order changes.
fn rowInvariant(gpu: check.Gpu, module: cuda.Module, k: usize) !void {
    const a = gpu.gpa;
    const n = 72;
    const stride = k + 8;
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var kernels = try glm.mm.Kernels.load(module);
    try kernels.loadFp8(module);
    kernels.packed_prompt_batch = true;
    var prng = std.Random.DefaultPrng.init(0xf8a771 + k);
    const random = prng.random();
    const x = try a.alloc(u16, capacity * stride);
    defer a.free(x);
    @memset(x, 0x7fc0);
    for (0..capacity) |r| for (0..k) |i| {
        const exponent: i32 = @as(i32, random.uintLessThan(u8, 20)) - 10;
        x[r * stride + i] = bf(std.math.ldexp(random.float(f32) * 4 - 2, exponent));
    };
    const codes = try a.alloc(u8, n * k);
    defer a.free(codes);
    for (codes) |*v| {
        v.* = random.int(u8);
        if (v.* & 0x7f == 0x7f) v.* -= 1;
    }
    const scales = try a.alloc(f32, n * k / 128);
    defer a.free(scales);
    for (scales) |*v| v.* = random.float(f32) * 0.3 + 0.001;
    var requires_rounding = false;
    for (0..n) |c| {
        var sum: f64 = 0;
        for (0..k) |i| sum += scalar(x[i]) * @as(f64, codec.e4m3Value(codes[c * k + i])) * @as(f64, scales[c * (k / 128) + i / 128]);
        const rounded: f32 = @floatCast(sum);
        requires_rounding = requires_rounding or sum != @as(f64, rounded);
    }
    try check.expect(requires_rounding, "FP8 row fixture requires rounded FP32 sums", .{});
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
    defer dx.free();
    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, codes);
    defer dw.free();
    var ds = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(scales));
    defer ds.free();
    const weight: glm.mm.Tiled = .{ .w = dw.ptr, .s = ds.ptr, .n = n, .k = k, .bits = 8, .format = .fp8 };
    var part = try cuda.DeviceBuffer.alloc(gpu.d, @max(16, glm.mm.partBytes(capacity, weight)));
    defer part.free();
    var output = try cuda.DeviceBuffer.alloc(gpu.d, capacity * n * 4);
    defer output.free();
    defer stream.synchronize() catch @panic("FP8 row fixture keeps buffers until completion");
    const baseline = try a.alloc(u8, output.len);
    defer a.free(baseline);
    const got = try a.alloc(u8, output.len);
    defer a.free(got);
    for ([_]bool{ false, true }) |wide| {
        const size: usize = if (wide) 4 else 2;
        kernels.pdl = false;
        for (0..capacity) |r| try glm.mm.matmul(&kernels, stream, dx.ptr + r * stride * 2, stride, weight, output.ptr + r * n * size, part.ptr, 1, wide, false);
        try stream.synchronize();
        try output.download(0, baseline);
        for ([_]bool{ false, true }) |pdl| {
            kernels.pdl = pdl;
            for ([_]bool{ false, true }) |prompt| for ([_]usize{ 1, 3, 16, 17, 33, 64, 126, 127, 128, 129, 254, 255, 256, 257 }) |rows| {
                for ([_]usize{ 0, 1 }) |offset| {
                    try glm.mm.matmul(&kernels, stream, dx.ptr + offset * stride * 2, stride, weight, output.ptr, part.ptr, rows, wide, prompt);
                    try stream.synchronize();
                    try output.download(0, got);
                    try check.sameBytes("FP8 random row/prompt bit invariance", baseline[offset * n * size ..][0 .. rows * n * size], got[0 .. rows * n * size]);
                }
            };
        }
    }
}

pub fn run(gpu: check.Gpu) !void {
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm_mm);
    defer module.unload();
    for ([_][2]usize{ .{ 8, 128 }, .{ 72, 512 }, .{ 128, 2048 } }) |dims| try shape(gpu, module, dims[0], dims[1]);
    for ([_]usize{ 128, 4096 }) |k| try rowInvariant(gpu, module, k);
    check.pass("glm-fp8: exact dots, random row invariance, BF16/FP32, five repeats, PDL on/off, prompt/decode, offsets, tails and guards", .{});
}
