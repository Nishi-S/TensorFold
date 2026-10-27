const std = @import("std");
const cuda = @import("cuda");
const mm = @import("mm");
const kernels: mm.Kernels = .{
    .by_rows = .{ .{ .id = 16 }, .{ .id = 32 }, .{ .id = 64 }, .{ .id = 128 } },
    .packed_rows = @splat(.{ .{ .id = 16 }, .{ .id = 32 }, .{ .id = 64 }, .{ .id = 128 } }),
    .prompt = .{ .id = 512 },
    .reduce = .{ .id = 1024 },
    .tile = .{},
    .tile_packed = @splat(.{}),
    .pdl = false,
};
test "verify and draft windows keep shape-selected split K at every width" {
    for ([_]u8{ 16, 4, 6, 8 }) |bits| for ([_]usize{ 1, 2, 3, 4, 8, 16, 17, 31, 32 }) |rows| {
        cuda.count = 0;
        try mm.matmul(&kernels, .{}, 4096, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = bits }, 16384, 32768, rows, false, false);
        try std.testing.expectEqual(@as(usize, 2), cuda.count);
        try std.testing.expectEqual(@as(u32, 4), cuda.configs[0].grid.z);
        try std.testing.expectEqual(@as(u32, 1024), cuda.symbols[1]);
    };
}

test "packed Q4 decoding is opt-in and excludes prompts, wide windows and other formats" {
    var k = kernels;
    k.q4_pair = .{ .id = 4096 };
    for ([_]u8{ 16, 4, 6, 8 }) |bits| for ([_]usize{ 1, 3, 16, 17, 32, 65 }) |rows| {
        for ([_]bool{ false, true }) |prompt| {
            cuda.count = 0;
            try mm.matmul(&k, .{}, 4096, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = bits }, 16384, 32768, rows, false, prompt);
            try std.testing.expectEqual(bits == 4 and rows <= 16 and !prompt, cuda.symbols[0] == 4096);
        }
    };
}

test "affine and FP8 formats have separate storage and dispatch at every row bucket" {
    const ptrs = [_]u64{ 4096, 8192, 16384 };
    const bits = [_]u8{ 4, 4, 8 };
    const formats = [_]mm.Format{ .symmetric, .affine4, .fp8 };
    const packs: mm.Packed = .{ .ptrs = &ptrs, .bits = &bits, .formats = &formats };
    var k = kernels;
    k.affine_rows = @splat(.{ .id = 2048 });
    k.fp8_rows = @splat(.{ .id = 4096 });
    for (formats, ptrs) |format, ptr| {
        const w = packs.tiled(ptr, 72, 512);
        try std.testing.expectEqual(format, w.format);
        const layout = mm.formatBytes(72, 512, w.bits, format);
        try std.testing.expectEqual(ptr + layout[0], w.s);
        try std.testing.expectEqual(@as(usize, if (format == .affine4) 72 * 8 * 2 else 0), layout[2]);
        for ([_]usize{ 1, 16, 17, 32, 64, 129 }) |rows| for ([_]bool{ false, true }) |prompt| {
            cuda.count = 0;
            try mm.matmul(&k, .{}, 32768, 520, w, 65536, 131072, rows, true, prompt);
            if (format == .affine4) try std.testing.expectEqual(@as(u32, 2048), cuda.symbols[0]);
            if (format == .fp8) try std.testing.expectEqual(@as(u32, 4096), cuda.symbols[0]);
            if (format != .symmetric) {
                try std.testing.expectEqual(@as(usize, 2 * ((rows + 63) / 64)), cuda.count);
                try std.testing.expectEqual(@as(u32, @intCast(mm.weightSplitK(w))), cuda.configs[0].grid.z);
            }
        };
    }
    const legacy: mm.Packed = .{ .ptrs = &ptrs, .bits = &bits };
    try std.testing.expectEqual(mm.Format.symmetric, legacy.tiled(8192, 72, 512).format);
}
test "prompt flag chooses one K slice even for one row, alone or many" {
    for ([_]u8{ 16, 4, 6, 8 }) |bits| for ([_]usize{ 1, 3, 16, 64, 127, 128, 129, 2048 }) |rows| {
        cuda.count = 0;
        try mm.matmul(&kernels, .{}, 4096, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = bits }, 16384, 32768, rows, false, true);
        try std.testing.expectEqual(@as(usize, 1), cuda.count);
        try std.testing.expectEqual(@as(u32, 1), cuda.configs[0].grid.z);
        try std.testing.expectEqual(@as(u32, 256), cuda.configs[0].block.x);
    };
}

test "fixed K prompt experiment uses the same shape split across bucket, offset and tail boundaries" {
    var k = kernels;
    k.prompt_fixed_k = true;
    for ([_]u8{ 16, 4, 6, 8 }) |bits| for ([_]usize{ 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2048 }) |rows| {
        cuda.count = 0;
        try mm.matmul(&k, .{}, 4096 + 7 * 2048 * 2, 2048, .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = bits }, 16384, 32768, rows, false, true);
        try std.testing.expectEqual(@as(usize, 2 * ((rows + 63) / 64)), cuda.count);
        var i: usize = 0;
        while (i < cuda.count) : (i += 2) {
            try std.testing.expectEqual(@as(u32, 4), cuda.configs[i].grid.z);
            try std.testing.expectEqual(@as(u32, 1024), cuda.symbols[i + 1]);
        }
    };
    try std.testing.expect(!kernels.prompt_fixed_k);
}
