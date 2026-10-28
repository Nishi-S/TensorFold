//! HC fast schedules measured with nvcc 13.0.88; later compiler resource receipts remain pending.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{
        .launch = .{ .name = "glm_glue tf_glm_hc_partial_mma", .image = "glm_glue", .symbol = "tf_glm_hc_partial_mma", .threads = 128, .min_sm = 80 },
        .usage = .{
            .{ .regs = 72, .static = 20480, .bound = 128 },
            .{ .regs = 128, .static = 20480, .bound = 128 },
            .{ .regs = 128, .static = 20480, .bound = 128 },
            .{ .regs = 98, .static = 20480, .bound = 128 },
            .{ .regs = 98, .static = 20480, .bound = 128 },
        },
        .later_pending = true,
    },
    .{
        .launch = .{ .name = "glm_glue tf_glm_hc_partial_dec", .image = "glm_glue", .symbol = "tf_glm_hc_partial_dec", .threads = 288, .min_sm = 80 },
        .usage = @splat(.{ .regs = 56, .static = 0, .bound = 288 }),
        .later_pending = true,
    },
};

test {
    _ = @import("glm_hc_fast_stress.zig");
}
