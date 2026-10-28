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

test "packed split windows cover row seams with unchanged K slices and bounded scratch" {
    var k = kernels;
    try std.testing.expect(!k.packed_prompt_batch);
    k.affine_rows = .{ .{ .id = 2016 }, .{ .id = 2032 }, .{ .id = 2064 }, .{ .id = 2128 } };
    k.fp8_rows = .{ .{ .id = 4016 }, .{ .id = 4032 }, .{ .id = 4064 }, .{ .id = 4128 } };
    for ([_]mm.Format{ .affine4, .fp8 }) |format| {
        const w: mm.Tiled = .{ .w = 8192, .s = 12288, .n = 4096, .k = 2048, .bits = if (format == .affine4) 4 else 8, .format = format };
        for ([_]usize{ 126, 127, 128, 129, 254, 255, 256, 257, 1862 }) |rows| {
            for ([_]bool{ false, true }) |batched| for ([_]bool{ false, true }) |prompt| for ([_]bool{ false, true }) |wide| {
                k.packed_prompt_batch = batched;
                const cap: usize = if (batched) 127 else 64;
                cuda.count = 0;
                const input = 4096 + 7 * 2056 * 2;
                try mm.matmul(&k, .{}, input, 2056, w, 65536, 131072, rows, wide, prompt);
                try std.testing.expectEqual(2 * ((rows + cap - 1) / cap), cuda.count);
                var covered: usize = 0;
                var launch: usize = 0;
                while (launch < cuda.count) : (launch += 2) {
                    const a = cuda.args[launch].values;
                    const reduction = cuda.args[launch + 1].values;
                    const count: usize = @intCast(a[7]);
                    try std.testing.expect(count > 0 and count <= cap);
                    try std.testing.expectEqual(input + covered * 2056 * 2, a[0]);
                    try std.testing.expectEqual(@as(u64, 2056), a[1]);
                    try std.testing.expectEqual(@as(u64, 65536 + covered * 4096 * @as(usize, if (wide) 4 else 2)), a[4]);
                    try std.testing.expectEqual(@as(u64, 131072), a[5]);
                    try std.testing.expectEqual(@as(u64, 4), a[10]);
                    try std.testing.expectEqual(@as(u32, 4), cuda.configs[launch].grid.z);
                    try std.testing.expectEqual(@as(u32, if (count > 64) 2 else 1), cuda.configs[launch].grid.y);
                    const bucket: u32 = if (count <= 16) 16 else if (count <= 32) 32 else 64;
                    try std.testing.expectEqual(@as(u32, if (format == .affine4) 2000 else 4000) + bucket, cuda.symbols[launch]);
                    try std.testing.expectEqual(@as(u32, 1024), cuda.symbols[launch + 1]);
                    try std.testing.expectEqual(a[5], reduction[0]);
                    try std.testing.expectEqual(a[4], reduction[1]);
                    try std.testing.expectEqual(count * w.n, reduction[2]);
                    try std.testing.expectEqual(@as(u64, 4), reduction[3]);
                    try std.testing.expect(count * w.n * 4 * 4 <= mm.partBytes(rows, w));
                    covered += count;
                }
                try std.testing.expectEqual(rows, covered);
            };
        }
    }
}

test "packed single slice keeps its existing short and prompt row policy" {
    var k = kernels;
    k.affine_rows = kernels.by_rows;
    k.fp8_rows = kernels.by_rows;
    for ([_]mm.Format{ .affine4, .fp8 }) |format| {
        const w: mm.Tiled = .{ .w = 8192, .s = 12288, .n = 12288, .k = 4096, .bits = if (format == .affine4) 4 else 8, .format = format };
        for ([_]usize{ 1, 16, 17, 32, 33, 64, 65, 126, 127, 128, 129, 257, 1862 }) |rows| for ([_]bool{ false, true }) |prompt| for ([_]bool{ false, true }) |batched| {
            k.packed_prompt_batch = batched;
            cuda.count = 0;
            try mm.matmul(&k, .{}, 4096, 4096, w, 65536, 0, rows, false, prompt);
            try std.testing.expectEqual(@as(usize, 0), mm.partBytes(rows, w));
            try std.testing.expectEqual(@as(usize, if (rows >= 128) 1 else (rows + 63) / 64), cuda.count);
            try std.testing.expectEqual(@as(u32, if (rows >= 128) 128 else if (rows <= 16) 16 else if (rows <= 32) 32 else 64), cuda.symbols[0]);
            for (0..cuda.count) |i| try std.testing.expectEqual(@as(u32, 1), cuda.configs[i].grid.z);
        };
    }
}

test "affine XS reuse avoids reduction and fallback computes each row group once" {
    var k = kernels;
    k.affine_rows = kernels.by_rows;
    k.affine_xs_rows = @splat(.{ .id = 3000 });
    k.affine_sums = .{ .id = 3001 };
    k.affine_xs = true;
    const w: mm.Tiled = .{ .w = 8192, .s = 12288, .n = 4096, .k = 4096, .bits = 4, .format = .affine4 };
    cuda.count = 0;
    try mm.matmulXs(&k, .{}, 4096, 4096, w, 65536, 131072, 65, false, false, 262144, 0);
    try std.testing.expectEqual(@as(usize, 4), cuda.count);
    try std.testing.expectEqual(@as(u32, 3000), cuda.symbols[0]);
    try std.testing.expectEqual(@as(u64, 262144), cuda.args[0].values[11]);
    try std.testing.expectEqual(@as(u64, 262144 + 64 * 64 * 4), cuda.args[2].values[11]);
    cuda.count = 0;
    try mm.matmulXs(&k, .{}, 4096, 4096, w, 65536, 131072, 65, false, false, 0, 262144);
    try std.testing.expectEqual(@as(usize, 5), cuda.count);
    try std.testing.expectEqual(@as(u32, 3001), cuda.symbols[0]);
    try std.testing.expectEqual(@as(u32, 65), cuda.configs[0].grid.x);
    cuda.count = 0;
    try std.testing.expectError(error.AffineSumsScratchMissing, mm.matmulXs(&k, .{}, 4096, 4096, w, 65536, 131072, 1, false, false, 0, 0));
    try std.testing.expectEqual(@as(usize, 0), cuda.count);
}

test "affine shape policy and reconstructed prompt are separate opt ins" {
    var k = kernels;
    k.affine_rows = kernels.by_rows;
    k.affine_xs_rows = kernels.by_rows;
    k.affine_wide = .{ .id = 5000 };
    k.affine_prompt = .{ .{ .id = 6064 }, .{ .id = 6128 } };
    const w: mm.Tiled = .{ .w = 8192, .s = 12288, .n = 4096, .k = 1024, .bits = 4, .format = .affine4 };
    k.affine_shapes = true;
    cuda.count = 0;
    try mm.matmulXs(&k, .{}, 4096, 1024, w, 65536, 131072, 1, true, false, 262144, 0);
    try std.testing.expectEqual(@as(u32, 5000), cuda.symbols[0]);
    try std.testing.expectEqual(@as(u32, 32), cuda.configs[0].grid.x);
    try std.testing.expectEqual(@as(u32, 2), cuda.configs[0].grid.z);
    try std.testing.expectEqual(@as(u32, 256), cuda.configs[0].block.x);
    k.affine_prefill = true;
    cuda.count = 0;
    try mm.matmulXs(&k, .{}, 4096, 1024, w, 65536, 131072, 640, true, true, 0, 0);
    try std.testing.expectEqual(@as(usize, 1), cuda.count);
    try std.testing.expectEqual(@as(u32, 6128), cuda.symbols[0]);
    try std.testing.expectEqual(@as(u32, 1), cuda.configs[0].grid.z);
    try std.testing.expectEqual(@as(u32, 5), cuda.configs[0].grid.y);
    const expanded: mm.Tiled = .{ .w = 0, .n = 12576, .k = 4096, .bits = 4, .format = .affine4 };
    try std.testing.expectEqual(@as(usize, 4 * 127 * 12576 * 4), mm.planPartBytes(640, expanded));
    try std.testing.expect(!kernels.affine_xs and !kernels.affine_shapes and !kernels.affine_prefill);
}

test "Mia Q4 shape table covers forced splits and the distinct fallback minimum" {
    for ([_][3]usize{ .{ 12576, 4096, 4 }, .{ 4096, 4096, 2 }, .{ 2048, 4096, 4 }, .{ 8192, 1536, 4 }, .{ 8192, 512, 4 }, .{ 4096, 8192, 8 }, .{ 4096, 1024, 2 }, .{ 128, 512, 1 } }) |s|
        try std.testing.expectEqual(s[2], mm.affineSplitK(s[0], s[1]));
    var k = kernels;
    k.affine_rows = kernels.by_rows;
    k.affine_xs_rows = kernels.by_rows;
    k.affine_wide_s6 = .{ .id = 5006 };
    k.affine_shapes = true;
    cuda.count = 0;
    try mm.matmulXs(&k, .{}, 4096, 8192, .{ .w = 8192, .s = 12288, .n = 4096, .k = 8192, .bits = 4, .format = .affine4 }, 65536, 131072, 1, true, false, 262144, 0);
    try std.testing.expectEqual(@as(u32, 5006), cuda.symbols[0]);
    try std.testing.expectEqual(@as(u32, 32), cuda.configs[0].grid.x);
    try std.testing.expectEqual(@as(u32, 8), cuda.configs[0].grid.z);
    try std.testing.expectEqual(@as(u32, 128), cuda.configs[0].block.x);
    try std.testing.expectEqual(mm.affineWideSharedBytes() / 4 * 6, cuda.configs[0].shared);
}
