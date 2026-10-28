//! GPU affine packing retains native scalar candidate arithmetic and returns only a four-byte error status to the host.
const std = @import("std");
const cuda = @import("cuda");

pub const Quantizer = struct {
    module: cuda.Module,
    function: cuda.Function,
    status: cuda.DeviceBuffer,

    pub fn init(d: *const cuda.Driver) !Quantizer {
        var module = try cuda.Module.load(d, cuda.kernels.glm_affine_quant);
        errdefer module.unload();
        const function = try module.function("tf_glm_affine_quant");
        return .{ .module = module, .function = function, .status = try cuda.DeviceBuffer.alloc(d, 4) };
    }

    pub fn deinit(q: *Quantizer) void {
        q.status.free();
        q.module.unload();
    }

    /// Distinct BF16 source and packed destination buffers remain owned until this synchronized call returns.
    pub fn run(q: *const Quantizer, s: cuda.Stream, src: u64, dst: u64, n: usize, k: usize) !void {
        if (n == 0 or k == 0 or k % 64 != 0) return error.BadAffineShape;
        const elements = std.math.mul(usize, n, k) catch return error.BadAffineShape;
        const groups = elements / 64;
        if ((groups + 3) / 4 > std.math.maxInt(u32)) return error.BadAffineShape;
        const words = elements / 2;
        const metadata = groups * 2;
        try q.status.fill8(0, s.handle);
        var args: cuda.Args = .{};
        args.add(src);
        args.add(dst);
        args.add(dst + words);
        args.add(dst + words + metadata);
        args.add(q.status.ptr);
        args.add(@as(u64, groups));
        try cuda.launch.launch(q.function, .{ .grid = .{ .x = @intCast((groups + 3) / 4) }, .block = .{ .x = 128 } }, s, &args);
        try s.synchronize();
        var status: [4]u8 = undefined;
        try q.status.download(0, &status);
        const bits = std.mem.readInt(u32, &status, .little);
        if (bits & 1 != 0) return error.NonfiniteAffineWeight;
        if (bits & 2 != 0) return error.AffineRangeOverflow;
    }
};
