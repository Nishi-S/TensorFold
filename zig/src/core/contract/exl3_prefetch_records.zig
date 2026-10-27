//! EXL3 load variants: nvcc 13.0.88 compiles; later compiler receipts remain pending.
const registry = @import("registry.zig");
fn record(comptime symbol: []const u8, regs: [5]u32, shared: u32) registry.CudaRecord {
    return .{
        .launch = .{ .name = "exl3_experts " ++ symbol, .image = "exl3_experts", .symbol = symbol, .threads = 128, .min_sm = 80 },
        .usage = .{
            .{ .regs = regs[0], .static = shared }, .{ .regs = regs[1], .static = shared },
            .{ .regs = regs[2], .static = shared }, .{ .regs = regs[3], .static = shared },
            .{ .regs = regs[4], .static = shared },
        },
        .later_pending = true,
    };
}
pub const records = [_]registry.CudaRecord{
    record("tf_exl3_grouped_1_pf1", .{ 164, 164, 164, 127, 127 }, 32832),
    record("tf_exl3_grouped_1_nc1", .{ 164, 164, 164, 128, 128 }, 32832),
    record("tf_exl3_grouped_1_nc2", .{ 164, 164, 164, 127, 127 }, 32832),
    record("tf_exl3_grouped_1_nc4", .{ 164, 164, 164, 143, 143 }, 32832),
    record("tf_exl3_down_fused", .{ 166, 168, 168, 146, 146 }, 32832),
    record("tf_exl3_gateup_fused", .{ 168, 168, 168, 149, 149 }, 32848),
};
