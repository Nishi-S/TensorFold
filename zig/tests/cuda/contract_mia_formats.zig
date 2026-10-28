//! The independent fork's affine Q4 and FP8 format gates share one loaded GLM module.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const g = @import("core").contract.gates;
const Gpu = @import("check.zig").Gpu;
const affine = @import("contract_glm_affine.zig");
const fp8 = @import("contract_glm_fp8.zig");

pub const Group = struct {
    module: ?cuda.Module = null,
    mm: glm.mm.Kernels = undefined,
    affine_cases: [11]affine.Case = undefined,
    fp8_cases: [3]fp8.Case = undefined,
    ready: bool = false,

    pub fn deinit(self: *Group) void {
        if (self.ready) {
            for (&self.affine_cases) |*item| item.deinit();
            for (&self.fp8_cases) |*item| item.deinit();
        }
        if (self.module) |*m| m.unload();
    }

    pub fn add(self: *Group, gpu: Gpu, stream: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !void {
        self.module = try cuda.Module.load(gpu.d, cuda.kernels.glm_mm);
        self.mm = try glm.mm.Kernels.load(self.module.?);
        try self.mm.loadAffine(self.module.?);
        try self.mm.loadFp8(self.module.?);
        for (&self.affine_cases, 0..) |*item, i| item.* = .{
            .rig = .{ .gpu = gpu, .s = stream },
            .k = &self.mm,
            .wide = i % 2 == 1 or i == 4,
            .route = if (i < 5) .native else if (i < 7) .xs else if (i < 9) .shapes else .reconstruct,
            .prompt = i == 2 or i == 3,
            .drop_tail = i == 4,
        };
        for (&self.fp8_cases, 0..) |*item, i| item.* = .{
            .rig = .{ .gpu = gpu, .s = stream },
            .kernels = &self.mm,
            .wide = i != 0,
            .drop_tail = i == 2,
        };
        self.ready = true;
        for (&self.affine_cases) |*item| try kernels.append(gpu.gpa, item.kernel());
        for (&self.fp8_cases) |*item| try kernels.append(gpu.gpa, item.kernel());
    }
};
