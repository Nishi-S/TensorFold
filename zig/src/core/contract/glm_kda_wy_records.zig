//! WY32 native prompt step measured with nvcc 13.0.88; later compiler qualification remains pending.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{
        .launch = .{ .name = "glm_kda tf_glm_kda_wy", .image = "glm_kda", .symbol = "tf_glm_kda_wy", .threads = 256, .dynamic = 74372, .opt_in = true, .min_sm = 80 },
        .usage = .{
            .{ .regs = 166, .static = 0, .bound = 256 },
            .{ .regs = 168, .static = 0, .bound = 256 },
            .{ .regs = 168, .static = 0, .bound = 256 },
            .{ .regs = 162, .static = 0, .bound = 256 },
            .{ .regs = 162, .static = 0, .bound = 256 },
        },
        .later_pending = true,
    },
};
