//! Record production KDA chain dispatch: WY requires the explicit prompt position and opt-in mode.
const std = @import("std");
const cuda = @import("cuda");
const kda = @import("cuda_kda.zig");
const Recorder = struct {
    var ids: [3]usize = undefined;
    var count: usize = 0;
    var shared: usize = 0;
    var offset: i32 = -1;
    fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, smem: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const id = @intFromPtr(f);
        ids[count] = id;
        count += 1;
        if (id == 4) {
            shared = smem;
            const p: *const i32 = @ptrCast(@alignCast(args.?[10].?));
            offset = p.*;
        }
        return cuda.abi.success;
    }
};
fn function(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}
test "KDA WY32: explicit prompt dispatch leaves decode serial" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Recorder.launch;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var k: kda.Kernels = .{ .chain = function(&d, 5), .replay = function(&d, 6), .replay_layers = function(&d, 7), .prep = function(&d, 1), .step = function(&d, 2), .out = function(&d, 3), .wy = function(&d, 4) };
    var chain: kda.Chain = .{ .p = 0, .p_stride = 512, .b_off = 384, .a = 0, .a_stride = 128, .g = 0, .g_stride = 128, .conv_state = 0, .conv_w = 0, .state_in = 0, .a_log = 0, .dt_bias = 0, .norm_w = 0, .eps = 1e-6, .lower = -5, .heads = 1 };
    const saves: kda.Saves = .{ .out = 0, .state_out = 0, .k = 0, .v = 0, .g = 0, .b = 0 };
    for ([_]bool{ false, true }) |enabled| for ([_]?usize{ null, 0, 31, 64 }) |position| {
        Recorder.count = 0;
        k.prompt_wy = enabled;
        chain.prompt_position = position;
        try kda.chain(&k, stream, chain, 33, saves, .{ .q = 0, .y = 0 });
        try std.testing.expectEqual(@as(usize, 3), Recorder.count);
        try std.testing.expectEqual(@as(usize, 1), Recorder.ids[0]);
        try std.testing.expectEqual(@as(usize, if (enabled and position != null) 4 else 2), Recorder.ids[1]);
        try std.testing.expectEqual(@as(usize, 3), Recorder.ids[2]);
        if (enabled and position != null) {
            try std.testing.expectEqual(@as(i32, @intCast(position.? % 32)), Recorder.offset);
            try std.testing.expectEqual(@as(usize, kda.wy_shared_bytes), Recorder.shared);
        }
    };
}

test "KDA WY32: native prompt and snapshot grid preserve subchunk alignment" {
    try std.testing.expectEqual(@as(usize, 0), @import("cuda_state.zig").prefill_rows % 32);
    try std.testing.expectEqual(@as(usize, 0), @import("cuda_native.zig").snap_grid % 32);
}
