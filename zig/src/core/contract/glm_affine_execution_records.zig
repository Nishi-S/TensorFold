//! Native affine execution routes measured with nvcc 13.0.88; GPU arithmetic qualification remains pending.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{
        .launch = .{ .name = "glm_mm glm_affine_sums", .image = "glm_mm", .symbol = "glm_affine_sums", .threads = 128, .dynamic = 0, .opt_in = true, .min_sm = 80 },
        .usage = .{
            .{ .regs = 19, .static = 0, .bound = 128 },
            .{ .regs = 19, .static = 0, .bound = 128 },
            .{ .regs = 19, .static = 0, .bound = 128 },
            .{ .regs = 18, .static = 0, .bound = 128 },
            .{ .regs = 18, .static = 0, .bound = 128 },
        },
        .later_pending = true,
    },
    .{
        .launch = .{ .name = "glm_mm glm_affine_xs16", .image = "glm_mm", .symbol = "glm_affine_xs16", .threads = 128, .dynamic = 13248, .opt_in = true, .min_sm = 80 },
        .usage = .{
            .{ .regs = 93, .static = 0, .bound = 128 },
            .{ .regs = 93, .static = 0, .bound = 128 },
            .{ .regs = 93, .static = 0, .bound = 128 },
            .{ .regs = 56, .static = 0, .bound = 128 },
            .{ .regs = 56, .static = 0, .bound = 128 },
        },
        .later_pending = true,
    },
    .{
        .launch = .{ .name = "glm_mm glm_affine_xs32", .image = "glm_mm", .symbol = "glm_affine_xs32", .threads = 128, .dynamic = 19584, .opt_in = true, .min_sm = 80 },
        .usage = .{
            .{ .regs = 96, .static = 0, .bound = 128 },
            .{ .regs = 88, .static = 0, .bound = 128 },
            .{ .regs = 88, .static = 0, .bound = 128 },
            .{ .regs = 72, .static = 0, .bound = 128 },
            .{ .regs = 72, .static = 0, .bound = 128 },
        },
        .later_pending = true,
    },
    .{
        .launch = .{ .name = "glm_mm glm_affine_xs64", .image = "glm_mm", .symbol = "glm_affine_xs64", .threads = 256, .dynamic = 32256, .opt_in = true, .min_sm = 80 },
        .usage = .{
            .{ .regs = 95, .static = 0, .bound = 256 },
            .{ .regs = 95, .static = 0, .bound = 256 },
            .{ .regs = 95, .static = 0, .bound = 256 },
            .{ .regs = 64, .static = 0, .bound = 256 },
            .{ .regs = 64, .static = 0, .bound = 256 },
        },
        .later_pending = true,
    },
    .{
        .launch = .{ .name = "glm_mm glm_affine_xs128", .image = "glm_mm", .symbol = "glm_affine_xs128", .threads = 256, .dynamic = 57600, .opt_in = true, .min_sm = 80 },
        .usage = .{
            .{ .regs = 128, .static = 0, .bound = 256 },
            .{ .regs = 128, .static = 0, .bound = 256 },
            .{ .regs = 128, .static = 0, .bound = 256 },
            .{ .regs = 102, .static = 0, .bound = 256 },
            .{ .regs = 102, .static = 0, .bound = 256 },
        },
        .later_pending = true,
    },
    .{
        .launch = .{ .name = "glm_mm glm_affine_wide16", .image = "glm_mm", .symbol = "glm_affine_wide16", .threads = 256, .dynamic = 26880, .opt_in = true, .min_sm = 80 },
        .usage = .{
            .{ .regs = 72, .static = 0, .bound = 256 },
            .{ .regs = 72, .static = 0, .bound = 256 },
            .{ .regs = 72, .static = 0, .bound = 256 },
            .{ .regs = 64, .static = 0, .bound = 256 },
            .{ .regs = 64, .static = 0, .bound = 256 },
        },
        .later_pending = true,
    },
    .{
        .launch = .{ .name = "glm_mm glm_affine_wide16s6", .image = "glm_mm", .symbol = "glm_affine_wide16s6", .threads = 128, .dynamic = 40320, .opt_in = true, .min_sm = 80 },
        .usage = .{
            .{ .regs = 96, .static = 0, .bound = 128 },
            .{ .regs = 96, .static = 0, .bound = 128 },
            .{ .regs = 96, .static = 0, .bound = 128 },
            .{ .regs = 65, .static = 0, .bound = 128 },
            .{ .regs = 65, .static = 0, .bound = 128 },
        },
        .later_pending = true,
    },
    .{
        .launch = .{ .name = "glm_mm glm_affine_prefill64", .image = "glm_mm", .symbol = "glm_affine_prefill64", .threads = 128, .dynamic = 43008, .opt_in = true, .min_sm = 80 },
        .usage = .{
            .{ .regs = 128, .static = 0, .bound = 128 },
            .{ .regs = 128, .static = 0, .bound = 128 },
            .{ .regs = 128, .static = 0, .bound = 128 },
            .{ .regs = 70, .static = 0, .bound = 128 },
            .{ .regs = 70, .static = 0, .bound = 128 },
        },
        .later_pending = true,
    },
    .{
        .launch = .{ .name = "glm_mm glm_affine_prefill128", .image = "glm_mm", .symbol = "glm_affine_prefill128", .threads = 128, .dynamic = 43008, .opt_in = true, .min_sm = 80 },
        .usage = .{
            .{ .regs = 255, .static = 0, .bound = 128 },
            .{ .regs = 255, .static = 0, .bound = 128 },
            .{ .regs = 255, .static = 0, .bound = 128 },
            .{ .regs = 235, .static = 0, .bound = 128 },
            .{ .regs = 235, .static = 0, .bound = 128 },
        },
        .later_pending = true,
    },
};
