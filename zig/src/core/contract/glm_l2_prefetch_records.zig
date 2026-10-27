//! Read-only L2 cache hints: measured nvcc 13.0.88 launch usage, later compiler receipts pending.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{
        .launch = .{
            .name = "glm_l2_prefetch tf_glm_l2_prefetch",
            .image = "glm_l2_prefetch",
            .symbol = "tf_glm_l2_prefetch",
            .threads = 128,
            .min_sm = 80,
        },
        .usage = .{
            .{ .regs = 12, .static = 0, .bound = 128 },
            .{ .regs = 12, .static = 0, .bound = 128 },
            .{ .regs = 12, .static = 0, .bound = 128 },
            .{ .regs = 10, .static = 0, .bound = 128 },
            .{ .regs = 10, .static = 0, .bound = 128 },
        },
        .later_pending = true,
    },
    .{
        .launch = .{
            .name = "glm_l2_prefetch tf_glm_l2_prefetch_bulk",
            .image = "glm_l2_prefetch",
            .symbol = "tf_glm_l2_prefetch_bulk",
            .threads = 128,
            .min_sm = 80,
        },
        .usage = .{
            .{ .regs = 12, .static = 0, .bound = 128 },
            .{ .regs = 12, .static = 0, .bound = 128 },
            .{ .regs = 12, .static = 0, .bound = 128 },
            .{ .regs = 14, .static = 0, .bound = 128 },
            .{ .regs = 14, .static = 0, .bound = 128 },
        },
        .later_pending = true,
    },
    .{
        .launch = .{
            .name = "glm_l2_prefetch tf_glm_l2_prefetch_touch",
            .image = "glm_l2_prefetch",
            .symbol = "tf_glm_l2_prefetch_touch",
            .threads = 128,
            .min_sm = 80,
        },
        .usage = .{
            .{ .regs = 8, .static = 0, .bound = 128 },
            .{ .regs = 8, .static = 0, .bound = 128 },
            .{ .regs = 8, .static = 0, .bound = 128 },
            .{ .regs = 10, .static = 0, .bound = 128 },
            .{ .regs = 10, .static = 0, .bound = 128 },
        },
        .later_pending = true,
    },
};
