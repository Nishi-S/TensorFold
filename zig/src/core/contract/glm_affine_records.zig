//! Affine Q4 compile resources measured with nvcc 13.0.88; device qualification remains separate.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{
        .launch = .{ .name = "glm_mm glm_mm_affine4_16", .image = "glm_mm", .symbol = "glm_mm_affine4_16", .threads = 128, .dynamic = 13248, .opt_in = true, .min_sm = 80 },
        .usage = .{
            .{ .regs = 119, .static = 0, .bound = 128 },
            .{ .regs = 119, .static = 0, .bound = 128 },
            .{ .regs = 119, .static = 0, .bound = 128 },
            .{ .regs = 54, .static = 0, .bound = 128 },
            .{ .regs = 54, .static = 0, .bound = 128 },
        },
        .later_pending = true,
    },
    .{
        .launch = .{ .name = "glm_mm glm_mm_affine4_32", .image = "glm_mm", .symbol = "glm_mm_affine4_32", .threads = 128, .dynamic = 19584, .opt_in = true, .min_sm = 80 },
        .usage = .{
            .{ .regs = 128, .static = 0, .bound = 128 },
            .{ .regs = 121, .static = 0, .bound = 128 },
            .{ .regs = 121, .static = 0, .bound = 128 },
            .{ .regs = 64, .static = 0, .bound = 128 },
            .{ .regs = 64, .static = 0, .bound = 128 },
        },
        .later_pending = true,
    },
    .{
        .launch = .{ .name = "glm_mm glm_mm_affine4_64", .image = "glm_mm", .symbol = "glm_mm_affine4_64", .threads = 256, .dynamic = 32256, .opt_in = true, .min_sm = 80 },
        .usage = .{
            .{ .regs = 100, .static = 0, .bound = 256 },
            .{ .regs = 101, .static = 0, .bound = 256 },
            .{ .regs = 101, .static = 0, .bound = 256 },
            .{ .regs = 64, .static = 0, .bound = 256 },
            .{ .regs = 64, .static = 0, .bound = 256 },
        },
        .later_pending = true,
    },
    .{
        .launch = .{ .name = "glm_mm glm_mm_affine4_128", .image = "glm_mm", .symbol = "glm_mm_affine4_128", .threads = 256, .dynamic = 57600, .opt_in = true, .min_sm = 80 },
        .usage = .{
            .{ .regs = 128, .static = 0, .bound = 256 },
            .{ .regs = 124, .static = 0, .bound = 256 },
            .{ .regs = 124, .static = 0, .bound = 256 },
            .{ .regs = 104, .static = 0, .bound = 256 },
            .{ .regs = 104, .static = 0, .bound = 256 },
        },
        .later_pending = true,
    },
    .{
        .launch = .{ .name = "glm_mm glm_mm_tile_affine4", .image = "glm_mm", .symbol = "glm_mm_tile_affine4", .threads = 256, .dynamic = 0, .opt_in = true, .min_sm = 80 },
        .usage = .{
            .{ .regs = 39, .static = 0, .bound = 0 },
            .{ .regs = 40, .static = 0, .bound = 0 },
            .{ .regs = 40, .static = 0, .bound = 0 },
            .{ .regs = 40, .static = 0, .bound = 0 },
            .{ .regs = 40, .static = 0, .bound = 0 },
        },
        .later_pending = true,
    },
};
