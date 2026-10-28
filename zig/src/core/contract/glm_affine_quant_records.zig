//! Native affine startup quantizer: nvcc 13.0.88 measured records; later compiler receipts remain pending.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{.{
    .launch = .{ .name = "glm_affine_quant tf_glm_affine_quant", .image = "glm_affine_quant", .symbol = "tf_glm_affine_quant", .threads = 128, .min_sm = 80 },
    .usage = .{
        .{ .regs = 29, .static = 0, .bound = 128 },
        .{ .regs = 30, .static = 0, .bound = 128 },
        .{ .regs = 30, .static = 0, .bound = 128 },
        .{ .regs = 32, .static = 0, .bound = 128 },
        .{ .regs = 32, .static = 0, .bound = 128 },
    },
    .later_pending = true,
}};
