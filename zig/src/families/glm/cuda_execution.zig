//! Rank-agreed execution experiments are installed before any weight packing or forward capture.
const fw = @import("cuda_forward.zig");
const Execution = @import("cuda_mia_settings.zig").Execution;

pub fn configure(m: *fw.Modules, e: Execution) !void {
    m.mm.affine_xs = e.affine_xs;
    m.mm.affine_shapes = e.affine_shapes;
    m.mm.affine_prefill = e.affine_prefill;
    m.glue.hc_mma = e.hc_mma;
    m.glue.hc_decode = e.hc_decode;
    m.moe_direct = e.moe_direct;
    try m.kda.configureWy(m.mods[0], e.kda_wy);
}
