//! Q4 pair conversion launch: nvcc 13.0.88 compiles; later compiler receipts remain pending.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{.{
    .launch = .{
        .name = "glm_mm glm_mm4_pair_16",
        .image = "glm_mm",
        .symbol = "glm_mm4_pair_16",
        .threads = 128,
        .dynamic = 12672,
        .opt_in = true,
        .min_sm = 80,
    },
    .usage = .{
        .{ .regs = 92, .static = 0, .bound = 128 },
        .{ .regs = 92, .static = 0, .bound = 128 },
        .{ .regs = 92, .static = 0, .bound = 128 },
        .{ .regs = 49, .static = 0, .bound = 128 },
        .{ .regs = 49, .static = 0, .bound = 128 },
    },
    .later_pending = true,
}};
