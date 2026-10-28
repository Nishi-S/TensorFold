//! Direct GLM route/combine resource usage measured with nvcc 13.0.88; later compilers need their own receipts.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{
        .launch = .{ .name = "glm_glue tf_glm_topk_routed", .image = "glm_glue", .symbol = "tf_glm_topk_routed", .threads = 32, .min_sm = 80 },
        .usage = .{
            .{ .regs = 64, .static = 0, .bound = 32 },
            .{ .regs = 67, .static = 0, .bound = 32 },
            .{ .regs = 67, .static = 0, .bound = 32 },
            .{ .regs = 67, .static = 0, .bound = 32 },
            .{ .regs = 67, .static = 0, .bound = 32 },
        },
        .later_pending = true,
    },
    .{
        .launch = .{ .name = "glm_glue tf_glm_combine_shared", .image = "glm_glue", .symbol = "tf_glm_combine_shared", .threads = 128, .min_sm = 80 },
        .usage = .{
            .{ .regs = 40, .static = 0, .bound = 128 },
            .{ .regs = 44, .static = 0, .bound = 128 },
            .{ .regs = 44, .static = 0, .bound = 128 },
            .{ .regs = 40, .static = 0, .bound = 128 },
            .{ .regs = 40, .static = 0, .bound = 128 },
        },
        .later_pending = true,
    },
};
