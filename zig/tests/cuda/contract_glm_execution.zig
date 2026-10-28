//! Independent contracts for the opt-in execution routes; ownership lasts through the complete gate run.
const std = @import("std");
const cuda = @import("cuda");
const g = @import("core").contract.gates;
const Gpu = @import("check.zig").Gpu;

pub const Group = struct {
    combine: ?@import("contract_glm_moe_direct.zig").Combine = null,
    topk: ?@import("contract_glm_moe_direct.zig").Topk = null,
    quant: @import("contract_glm_affine_quant.zig").Group = .{},
    hc: @import("contract_glm_hc_fast.zig").Group = .{},
    kda: @import("contract_glm_kda_wy.zig").Group = .{},

    pub fn add(s: *Group, gpu: Gpu, stream: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !void {
        try s.quant.add(gpu, stream, kernels);
        try s.hc.add(gpu, stream, kernels);
        try s.kda.add(gpu, stream, kernels);
    }

    pub fn addMoe(s: *Group, gpu: Gpu, stream: cuda.Stream, f: *const @import("glm_cuda").glue.Kernels, kernels: *std.ArrayList(g.Kernel)) !void {
        s.combine = .{ .rig = .{ .gpu = gpu, .s = stream }, .set = .{ .glue = f }, .arena = .init(gpu.gpa) };
        s.topk = .{ .rig = .{ .gpu = gpu, .s = stream }, .set = .{ .glue = f }, .arena = .init(gpu.gpa) };
        try kernels.append(gpu.gpa, s.combine.?.kernel());
        try kernels.append(gpu.gpa, s.topk.?.kernel());
    }

    pub fn deinit(s: *Group) void {
        if (s.topk) |*x| x.deinit();
        if (s.combine) |*x| x.deinit();
        s.kda.deinit();
        s.hc.deinit();
        s.quant.deinit();
    }
};
