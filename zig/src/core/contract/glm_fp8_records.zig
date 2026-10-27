//! Weight-only FP8 launches measured with nvcc 13.0.88; later compiler receipts remain pending.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{
        .launch = .{ .name = "glm_mm glm_fp8_16", .image = "glm_mm", .symbol = "glm_fp8_16", .threads = 128, .dynamic = 25088, .opt_in = true, .min_sm = 80 },
        .usage = .{
            .{ .regs = 77, .static = 0, .bound = 128 },
            .{ .regs = 72, .static = 0, .bound = 128 },
            .{ .regs = 72, .static = 0, .bound = 128 },
            .{ .regs = 56, .static = 0, .bound = 128 },
            .{ .regs = 56, .static = 0, .bound = 128 },
        },
        .later_pending = true,
    },
    .{
        .launch = .{ .name = "glm_mm glm_fp8_32", .image = "glm_mm", .symbol = "glm_fp8_32", .threads = 128, .dynamic = 33280, .opt_in = true, .min_sm = 80 },
        .usage = .{
            .{ .regs = 104, .static = 0, .bound = 128 },
            .{ .regs = 104, .static = 0, .bound = 128 },
            .{ .regs = 104, .static = 0, .bound = 128 },
            .{ .regs = 72, .static = 0, .bound = 128 },
            .{ .regs = 72, .static = 0, .bound = 128 },
        },
        .later_pending = true,
    },
    .{
        .launch = .{ .name = "glm_mm glm_fp8_64", .image = "glm_mm", .symbol = "glm_fp8_64", .threads = 256, .dynamic = 49664, .opt_in = true, .min_sm = 80 },
        .usage = .{
            .{ .regs = 80, .static = 0, .bound = 256 },
            .{ .regs = 80, .static = 0, .bound = 256 },
            .{ .regs = 80, .static = 0, .bound = 256 },
            .{ .regs = 69, .static = 0, .bound = 256 },
            .{ .regs = 69, .static = 0, .bound = 256 },
        },
        .later_pending = true,
    },
    .{
        .launch = .{ .name = "glm_mm glm_fp8_128", .image = "glm_mm", .symbol = "glm_fp8_128", .threads = 256, .dynamic = 82432, .opt_in = true, .min_sm = 80 },
        .usage = .{
            .{ .regs = 160, .static = 0, .bound = 256 },
            .{ .regs = 160, .static = 0, .bound = 256 },
            .{ .regs = 160, .static = 0, .bound = 256 },
            .{ .regs = 114, .static = 0, .bound = 256 },
            .{ .regs = 114, .static = 0, .bound = 256 },
        },
        .later_pending = true,
    },
};
