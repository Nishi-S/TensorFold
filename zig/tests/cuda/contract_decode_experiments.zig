//! Opt-in GLM decode experiments registered together without growing the contract runner.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const g = @import("core").contract.gates;
const Gpu = @import("check.zig").Gpu;
const exl3 = @import("contract_exl3_prefetch.zig");
const q4 = @import("contract_glm_q4_pair.zig");
const l2 = @import("contract_glm_l2_prefetch.zig");

pub const Group = struct {
    exl3_mod: ?cuda.Module = null,
    q4_mod: ?cuda.Module = null,
    l2_mod: ?cuda.Module = null,
    q4_kernels: glm.mm.Kernels = undefined,
    exl3_cases: [11]exl3.Case = undefined,
    q4_cases: [3]q4.Case = undefined,
    l2_cases: [4]l2.Case = undefined,
    exl3_ready: bool = false,
    q4_ready: bool = false,
    l2_ready: bool = false,

    pub fn deinit(self: *Group) void {
        if (self.exl3_ready) for (&self.exl3_cases) |*item| item.deinit();
        if (self.q4_ready) for (&self.q4_cases) |*item| item.deinit();
        if (self.l2_ready) for (&self.l2_cases) |*item| item.deinit();
        if (self.exl3_mod) |*m| m.unload();
        if (self.q4_mod) |*m| m.unload();
        if (self.l2_mod) |*m| m.unload();
    }

    pub fn add(self: *Group, gpu: Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !void {
        self.exl3_mod = try cuda.Module.load(gpu.d, cuda.kernels.exl3_experts);
        const names = [_][:0]const u8{ "tf_exl3_grouped_1_pf1", "tf_exl3_grouped_1_nc1", "tf_exl3_grouped_1_nc2", "tf_exl3_grouped_1_nc4" };
        var candidates: [4]cuda.Function = undefined;
        for (names, &candidates) |name, *f| f.* = try self.exl3_mod.?.function(name);
        const baseline = try self.exl3_mod.?.function("tf_exl3_grouped_1");
        const down = try self.exl3_mod.?.function("tf_exl3_down_fused");
        const down_baseline = try self.exl3_mod.?.function("tf_exl3_down_combine");
        const gateup = try self.exl3_mod.?.function("tf_exl3_gateup_fused");
        const gateup_baseline = try self.exl3_mod.?.function("tf_exl3_gateup_epilogue");
        const labels = [_][]const u8{
            "cuda EXL3 scalar down split1", "cuda EXL3 scalar gate/up split4",
            "cuda EXL3 nc1 down split1",    "cuda EXL3 nc1 gate/up split4",
            "cuda EXL3 nc2 down split1",    "cuda EXL3 nc2 gate/up split4",
            "cuda EXL3 nc4 down split1",    "cuda EXL3 nc4 gate/up split4",
        };
        for (&self.exl3_cases, 0..) |*item, i| item.* = .{
            .rig = .{ .gpu = gpu, .s = s },
            .candidate = if (i == 10) gateup else if (i == 9) down else candidates[if (i < 8) i / 2 else 0],
            .baseline = baseline,
            .sk = if (i == 10 or (i < 8 and i % 2 == 1)) 4 else 1,
            .mats = if (i == 10 or (i < 8 and i % 2 == 1)) 2 else 1,
            .drop_tail = i == 8,
            .down_fused = i == 9,
            .down_baseline = if (i == 9) down_baseline else null,
            .gateup_fused = i == 10,
            .gateup_baseline = if (i == 10) gateup_baseline else null,
            .name = if (i < 8) labels[i] else null,
            .arena = .init(gpu.gpa),
        };
        self.exl3_ready = true;
        for (&self.exl3_cases) |*item| try kernels.append(gpu.gpa, item.kernel());
        self.q4_mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_mm);
        self.q4_kernels = try glm.mm.Kernels.load(self.q4_mod.?);
        self.q4_kernels.q4_pair = try self.q4_mod.?.function("glm_mm4_pair_16");
        try self.q4_kernels.q4_pair.?.allowDynamicShared(glm.mm.sharedBytes(4, 16));
        for (&self.q4_cases, 0..) |*item, i| item.* = .{
            .rig = .{ .gpu = gpu, .s = s },
            .k = &self.q4_kernels,
            .wide = i != 0,
            .drop_tail = i == 2,
        };
        self.q4_ready = true;
        for (&self.q4_cases) |*item| try kernels.append(gpu.gpa, item.kernel());
        self.l2_mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_l2_prefetch);
        const l2_functions = [_]cuda.Function{
            try self.l2_mod.?.function("tf_glm_l2_prefetch_bulk"),
            try self.l2_mod.?.function("tf_glm_l2_prefetch"),
            try self.l2_mod.?.function("tf_glm_l2_prefetch_touch"),
        };
        for (&self.l2_cases, 0..) |*item, i| item.* = .{
            .rig = .{ .gpu = gpu, .s = s },
            .candidate = l2_functions[if (i < 3) i else 1],
            .mode = @intCast(if (i < 3) i + 1 else 2),
            .mutate = i == 3,
        };
        self.l2_ready = true;
        for (&self.l2_cases) |*item| try kernels.append(gpu.gpa, item.kernel());
    }
};
