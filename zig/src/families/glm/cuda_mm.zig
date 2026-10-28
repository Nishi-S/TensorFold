//! GLM's bf16 linears on tensor cores (glm_mm): weights tiled once at load, any row count, row-stable K slices.
const std = @import("std");
const cuda = @import("cuda");

pub const Format = enum(u8) { symmetric, affine4, fp8 };

/// A linear's tiles: bf16 pairs (glm_mm_tile16), or `bits`-bit codes (glm_mm_tile4/6/8) with fp16 scales at `s`.
pub const Tiled = struct { w: u64, n: usize, k: usize, s: u64 = 0, bits: u8 = 16, format: Format = .symmetric };

/// The widths glm_mm packs to, by kernel set: bf16 first.
const widths = [_]u8{ 16, 4, 6, 8 };

fn widthIndex(bits: u8) !usize {
    return std.mem.indexOfScalar(u8, &widths, bits) orelse error.BadBits;
}

/// Bytes a packed linear's codes take (its scales follow, 256-aligned): what fits in place of its bf16 tiles.
pub fn packedBytes(n: usize, k: usize, bits: u8) [2]usize {
    return .{ std.mem.alignForward(usize, n * k * bits / 8, 256), n * (k / 64) * 2 };
}

/// Codes followed by format-specific scale and bias tables; all formats fit inside the source BF16 allocation.
pub fn formatBytes(n: usize, k: usize, bits: u8, format: Format) [3]usize {
    const old = packedBytes(n, k, bits);
    return switch (format) {
        .symmetric => .{ old[0], old[1], 0 },
        .affine4 => .{ old[0], old[1], old[1] },
        .fp8 => .{ old[0], n * (k / 128) * 4, 0 },
    };
}

/// The linears packed at load: each one's tiles pointer (sorted) and its bits; a linear not here is bf16.
pub const Packed = struct {
    ptrs: []const u64 = &.{},
    bits: []const u8 = &.{},
    formats: []const Format = &.{},

    /// The tiled weight `w` [n][k] as the matmul reads it: packed codes with their scales after them, or bf16.
    pub fn tiled(p: *const Packed, w: u64, n: usize, k: usize) Tiled {
        var lo: usize = 0;
        var hi: usize = p.ptrs.len;
        while (lo < hi) {
            const mid = (lo + hi) / 2;
            if (p.ptrs[mid] < w) lo = mid + 1 else hi = mid;
        }
        if (lo == p.ptrs.len or p.ptrs[lo] != w) return .{ .w = w, .n = n, .k = k };
        const b = p.bits[lo];
        return .{ .w = w, .n = n, .k = k, .s = w + packedBytes(n, k, b)[0], .bits = b, .format = if (p.formats.len == 0) .symmetric else p.formats[lo] };
    }
};

/// glm_mm's bf16 kernels by row bucket, the K-slice reduce and the tiling.
pub const Kernels = struct {
    by_rows: [4]cuda.Function, // bf16: 16, 32, 64, 128 rows a block
    packed_rows: [3][4]cuda.Function, // 4, 6 and 8 bits, as by_rows
    prompt: cuda.Function, // 128-row blocks, row blocks fastest: a prompt's blocks share each weight tile in L2
    reduce: cuda.Function,
    tile: cuda.Function,
    tile_packed: [3]cuda.Function, // glm_mm_tile4/6/8: packed rows and their scales into the tiles
    prompt_fixed_k: bool = false, // opt-in prompt experiment; never selected from the number of batched rows
    packed_prompt_batch: bool = false, // opt-in affine/FP8 split windows using two 64-row blocks
    affine_xs: bool = false,
    affine_shapes: bool = false,
    affine_prefill: bool = false,
    affine_xs_rows: ?[4]cuda.Function = null,
    affine_sums: ?cuda.Function = null,
    affine_wide: ?cuda.Function = null,
    affine_wide_s6: ?cuda.Function = null,
    affine_prompt: ?[2]cuda.Function = null,
    affine_rows: ?[4]cuda.Function = null,
    affine_tile: ?cuda.Function = null,
    fp8_rows: ?[4]cuda.Function = null,
    q4_pair: ?cuda.Function = null, // opt-in packed bf16 decoding for Q4 decode windows up to 16 rows
    pdl: bool, // dependent launches: a matmul's weights stream in while the kernel before it finishes

    pub fn load(m: cuda.Module) !Kernels {
        var k: Kernels = .{ .by_rows = undefined, .packed_rows = undefined, .tile_packed = undefined, .prompt = try m.function("glm_mm16_rows"), .reduce = try m.function("glm_mm_reduce"), .tile = try m.function("glm_mm_tile16"), .pdl = true };
        inline for (.{ "glm_mm16_16", "glm_mm16_32", "glm_mm16_64", "glm_mm16_128" }, 0..) |name, i| {
            k.by_rows[i] = try m.function(name);
            try k.by_rows[i].allowDynamicShared(smem(bucket(i)));
        }
        inline for (widths[1..], 0..) |b, wi| {
            inline for (.{ 16, 32, 64, 128 }, 0..) |bm, i| {
                k.packed_rows[wi][i] = try m.function(std.fmt.comptimePrint("glm_mm{d}_{d}", .{ b, bm }));
                try k.packed_rows[wi][i].allowDynamicShared(sharedBytes(b, bm));
            }
            k.tile_packed[wi] = try m.function(std.fmt.comptimePrint("glm_mm_tile{d}", .{b}));
        }
        try k.prompt.allowDynamicShared(smem(128));
        if (std.c.getenv("TF_GLM_PDL")) |v| k.pdl = !std.mem.eql(u8, std.mem.span(v), "0");
        if (std.c.getenv("TF_GLM_Q4_PAIR_DECODE")) |v| {
            const setting = std.mem.span(v);
            if (std.mem.eql(u8, setting, "1")) {
                k.q4_pair = try m.function("glm_mm4_pair_16");
                try k.q4_pair.?.allowDynamicShared(sharedBytes(4, 16));
            } else if (!std.mem.eql(u8, setting, "0")) return error.BadQ4PairDecode;
        }
        if (std.c.getenv("TF_GLM_DENSE_FORMAT")) |v| if (std.mem.eql(u8, std.mem.span(v), "affine4")) try k.loadAffine(m);
        if (std.c.getenv("TF_GLM_HEAD_FORMAT")) |v| if (std.mem.eql(u8, std.mem.span(v), "fp8")) try k.loadFp8(m);
        return k;
    }

    pub fn loadAffine(k: *Kernels, m: cuda.Module) !void {
        var rows: [4]cuda.Function = undefined;
        inline for (.{ 16, 32, 64, 128 }, 0..) |bm, i| {
            rows[i] = try m.function(std.fmt.comptimePrint("glm_mm_affine4_{d}", .{bm}));
            try rows[i].allowDynamicShared(affineSharedBytes(bm));
        }
        k.affine_rows = rows;
        var xs_rows: [4]cuda.Function = undefined;
        inline for (.{ 16, 32, 64, 128 }, 0..) |bm, i| {
            xs_rows[i] = try m.function(std.fmt.comptimePrint("glm_affine_xs{d}", .{bm}));
            try xs_rows[i].allowDynamicShared(affineSharedBytes(bm));
        }
        k.affine_xs_rows = xs_rows;
        k.affine_sums = try m.function("glm_affine_sums");
        k.affine_wide = try m.function("glm_affine_wide16");
        try k.affine_wide.?.allowDynamicShared(affineWideSharedBytes());
        k.affine_wide_s6 = try m.function("glm_affine_wide16s6");
        try k.affine_wide_s6.?.allowDynamicShared(affineWideSharedBytes() / 4 * 6);
        var prompt: [2]cuda.Function = undefined;
        inline for (.{ 64, 128 }, 0..) |bm, i| {
            prompt[i] = try m.function(std.fmt.comptimePrint("glm_affine_prefill{d}", .{bm}));
            try prompt[i].allowDynamicShared(affinePromptSharedBytes(bm));
        }
        k.affine_prompt = prompt;
        k.affine_tile = try m.function("glm_mm_tile_affine4");
    }

    pub fn loadFp8(k: *Kernels, m: cuda.Module) !void {
        var rows: [4]cuda.Function = undefined;
        inline for (.{ 16, 32, 64, 128 }, 0..) |bm, i| {
            rows[i] = try m.function(std.fmt.comptimePrint("glm_fp8_{d}", .{bm}));
            try rows[i].allowDynamicShared(fp8SharedBytes(bm));
        }
        k.fp8_rows = rows;
    }
};

const split_target = 192; // column tiles times K slices a launch aims at (the GB10's SMs, a few blocks each)

fn bucket(i: usize) usize {
    return @as(usize, 16) << @intCast(i);
}

fn smem(bm: usize) u32 {
    return sharedBytes(16, bm);
}

/// A block's shared memory: three stages of `bm` rows of x and a 64-column tile of weights (`bits / 2` words a lane).
pub fn sharedBytes(bits: u8, bm: usize) u32 {
    const wpc: usize = if (bits == 16) 8 else bits / 2;
    return @intCast(3 * (bm * 128 + 8 * 32 * wpc * 4 + 128));
}

pub fn affineSharedBytes(bm: usize) u32 {
    return sharedBytes(4, bm) + @as(u32, @intCast(3 * (128 + bm * 4)));
}

pub fn fp8SharedBytes(bm: usize) u32 {
    return @intCast(2 * (bm * 256 + 64 * 128 + 64 * 4));
}

pub fn weightSplitK(w: Tiled) usize {
    if (w.format != .fp8) return splitK(w.n, w.k);
    var sk = splitK(w.n, w.k);
    while (sk > 1 and (w.k / 128) % sk != 0) sk /= 2;
    return sk;
}

/// Mia v0.6.0 Q4 SHAPE_SK and its eight-group minimum fallback, independent of the native BF16 policy.
pub fn affineSplitK(n: usize, k: usize) usize {
    if (k == 4096) {
        if (n == 4096) return 2;
        if (n == 2048 or n == 12576) return 4;
    }
    if (n == 8192 and (k == 1536 or k == 512)) return 4;
    if (n == 4096 and k == 8192) return 8;
    const tiles = (n + 63) / 64;
    const groups = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < split_target and groups % (sk * 2) == 0 and groups / (sk * 2) >= 8) sk *= 2;
    return sk;
}

pub fn affinePromptSharedBytes(bm: usize) u32 {
    return if (bm == 64) 4 * (64 * 128 + 8 * 32 * 2 * 4 + 2 * 64 * 2 + 64 * 4) else 2 * (128 * 128 + 16 * 32 * 2 * 4 + 2 * 128 * 2 + 128 * 4);
}

pub fn affineWideSharedBytes() u32 {
    return 4 * (16 * 128 + 16 * 32 * 2 * 4 + 2 * 128 * 2 + 16 * 4);
}

/// Scratch reserves both native and optional Q4 split policies, including the 127-row window.
pub fn planPartBytes(m: usize, w: Tiled) usize {
    const sk = @max(weightSplitK(w), if (w.format == .affine4) affineSplitK(w.n, w.k) else 1);
    return if (sk > 1) sk * @min(m, packed_split_rows) * w.n * 4 else 0;
}

/// K slices fixed by the shape: column tiles times slices reach the target, each slice at least 4 groups of 64.
pub fn splitK(n: usize, k: usize) usize {
    const tiles = (n + 63) / 64;
    const steps = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < split_target and steps % (sk * 2) == 0 and steps / (sk * 2) >= 4) sk *= 2;
    return sk;
}

/// Two 64-row blocks stay within the native planner's 127-row split-K budget per shape.
pub const packed_split_rows = 127;

/// Scratch for 64 symmetric rows or packed_split_rows affine/FP8 rows, conservatively including disabled batching.
pub fn partBytes(m: usize, w: Tiled) usize {
    const sk = weightSplitK(w);
    return if (sk > 1) sk * @min(m, if (w.format == .symmetric) @as(usize, 64) else packed_split_rows) * w.n * 4 else 0;
}

/// x [m][k] by `w` -> out [m][n]; affine/FP8 always keep shape-fixed K slices, including prompt rows.
pub fn matmul(k: *const Kernels, s: cuda.Stream, x: u64, ldx: usize, w: Tiled, out: u64, part: u64, m: usize, wide: bool, one: bool) !void {
    return matmulXs(k, s, x, ldx, w, out, part, m, wide, one, 0, 0);
}

/// Supplied XS must use native eight-values-per-lane sums; otherwise compute once into the caller-owned scratch.
pub fn matmulXs(k: *const Kernels, s: cuda.Stream, x: u64, ldx: usize, w: Tiled, out: u64, part: u64, m: usize, wide: bool, one: bool, supplied_xs: u64, scratch_xs: u64) !void {
    if (w.k % 64 != 0 or w.n % 8 != 0 or x % 16 != 0 or ldx % 8 != 0) return error.BadShape;
    if (w.format == .affine4 and (w.bits != 4 or k.affine_rows == null)) return error.AffineFormatNotLoaded;
    if (w.format == .fp8 and (w.bits != 8 or w.k % 128 != 0 or k.fp8_rows == null)) return error.Fp8FormatNotLoaded;
    const reconstruct = w.format == .affine4 and k.affine_prefill and one;
    const reuse = w.format == .affine4 and (k.affine_xs or k.affine_shapes) and !reconstruct;
    var xs = supplied_xs;
    if (reuse and xs == 0) {
        xs = scratch_xs;
        if (xs == 0) return error.AffineSumsScratchMissing;
        var sums: cuda.Args = .{};
        sums.add(x);
        sums.add(@as(c_longlong, @intCast(ldx)));
        sums.add(xs);
        sums.add(@as(c_int, @intCast(m)));
        sums.add(@as(c_int, @intCast(w.k / 64)));
        try cuda.launch.launch(k.affine_sums orelse return error.AffineFormatNotLoaded, .{ .grid = .{ .x = @intCast(m), .y = @intCast((w.k / 64 + 15) / 16) }, .block = .{ .x = 128 } }, s, &sums);
    }
    const shape_sk = if (w.format == .affine4 and k.affine_shapes) affineSplitK(w.n, w.k) else weightSplitK(w);
    var done: usize = 0;
    const wi = try widthIndex(w.bits);
    while (done < m) { // split windows preserve K slices; unsplit calls may use 128-row blocks
        const big = reconstruct or if (w.format != .symmetric)
            m - done >= 128 and shape_sk == 1
        else if (one) !k.prompt_fixed_k else m - done >= 128;
        const cap: usize = if (k.packed_prompt_batch and w.format != .symmetric and shape_sk > 1) packed_split_rows else 64;
        const rows = if (big) m - done else @min(m - done, cap);
        const i: usize = if (big) 3 else if (rows <= 16) 0 else if (rows <= 32) 1 else 2;
        const bm = if (reconstruct) @as(usize, if (w.k >= 1024 and ((rows + 127) / 128) * ((w.n + 127) / 128) >= 96) 128 else 64) else bucket(i);
        const sk = if (big) 1 else shape_sk; // new formats only take big blocks when their shape needs one slice
        const o = out + done * w.n * @as(usize, if (wide) 4 else 2);
        var a: cuda.Args = .{};
        a.add(x + done * ldx * 2);
        a.add(@as(c_longlong, @intCast(ldx)));
        a.add(w.w);
        a.add(w.s);
        a.add(o);
        a.add(part);
        a.add(@as(c_int, @intFromBool(wide)));
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, @intCast(w.n)));
        a.add(@as(c_int, @intCast(w.k)));
        a.add(@as(c_int, @intCast(sk)));
        const wide_tile = reuse and k.affine_shapes and rows <= 16 and ((w.n == 4096 and (w.k == 4096 or w.k == 1024)) or (w.n == 2048 and w.k == 4096) or (w.n == 8192 and w.k == 1536));
        const deep_tile = reuse and k.affine_shapes and rows <= 16 and w.n == 4096 and w.k == 8192;
        if (reuse or reconstruct) a.add(if (reconstruct) @as(u64, 0) else xs + done * (w.k / 64) * 4);
        const cols: u32 = @intCast((w.n + @as(usize, if (wide_tile or deep_tile or (reconstruct and bm == 128)) 127 else 63)) / @as(usize, if (wide_tile or deep_tile or (reconstruct and bm == 128)) 128 else 64));
        const row_blocks: u32 = @intCast((rows + bm - 1) / bm);
        if (big and wi == 0 and w.format == .symmetric) {
            try cuda.launch.launch(k.prompt, .{ .grid = .{ .x = row_blocks, .y = cols }, .block = .{ .x = 256 }, .shared = smem(128), .pdl = k.pdl }, s, &a);
        } else { // packed prompts take the 128-row bucket's blocks, one slice
            const f = if (reconstruct) (k.affine_prompt orelse return error.AffineFormatNotLoaded)[@intFromBool(bm == 128)] else if (deep_tile) k.affine_wide_s6 orelse return error.AffineFormatNotLoaded else if (wide_tile) k.affine_wide orelse return error.AffineFormatNotLoaded else if (reuse) (k.affine_xs_rows orelse return error.AffineFormatNotLoaded)[i] else if (w.format == .affine4) k.affine_rows.?[i] else if (w.format == .fp8) k.fp8_rows.?[i] else if (!one and m <= 16 and w.bits == 4 and k.q4_pair != null)
                k.q4_pair.?
            else if (wi == 0) k.by_rows[i] else k.packed_rows[wi - 1][i];
            try cuda.launch.launch(f, .{ .grid = .{ .x = cols, .y = row_blocks, .z = @intCast(sk) }, .block = .{ .x = if (reconstruct or deep_tile) 128 else if (wide_tile or bm > 32) 256 else 128 }, .shared = switch (w.format) {
                .symmetric => sharedBytes(w.bits, bm),
                .affine4 => if (reconstruct) affinePromptSharedBytes(bm) else if (deep_tile) affineWideSharedBytes() / 4 * 6 else if (wide_tile) affineWideSharedBytes() else affineSharedBytes(bm),
                .fp8 => fp8SharedBytes(bm),
            }, .pdl = k.pdl }, s, &a);
        }
        if (sk > 1) {
            const total = rows * w.n;
            var r: cuda.Args = .{};
            r.add(part);
            r.add(o);
            r.add(@as(c_longlong, @intCast(total)));
            r.add(@as(c_int, @intCast(sk)));
            r.add(@as(c_int, @intFromBool(wide)));
            try cuda.launch.launch(k.reduce, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &r);
        }
        done += rows;
    }
}

/// Packed rows (cuda_qpack's) and fp16 scales [n][k / 64] at `src`, `ss` tiled into `dst`, `ds` (`bits` 4/6/8).
pub fn tilePacked(k: *const Kernels, s: cuda.Stream, bits: u8, src: u64, ss: u64, dst: u64, ds: u64, n: usize, kk: usize) !void {
    const wi = try widthIndex(bits);
    if (wi == 0) return error.BadBits;
    const chunks = n / 8 * (kk / 64) * 32;
    var a: cuda.Args = .{};
    a.add(src);
    a.add(ss);
    a.add(dst);
    a.add(ds);
    a.add(@as(c_int, @intCast(n)));
    a.add(@as(c_int, @intCast(kk)));
    try cuda.launch.launch(k.tile_packed[wi - 1], .{ .grid = .{ .x = @intCast((chunks + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
}

/// Affine Q4 codes and consecutive BF16 scale/bias tables tiled without changing their stored bits.
pub fn tileAffine(k: *const Kernels, s: cuda.Stream, src: u64, ss: u64, dst: u64, ds: u64, n: usize, kk: usize) !void {
    const f = k.affine_tile orelse return error.AffineFormatNotLoaded;
    const chunks = n / 8 * (kk / 64) * 32;
    var a: cuda.Args = .{};
    for ([_]u64{ src, ss, dst, ds }) |v| a.add(v);
    a.add(@as(c_int, @intCast(n)));
    a.add(@as(c_int, @intCast(kk)));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((chunks + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
}

/// A row-major bf16 [n][k] linear at `src` tiled into `dst` (as many bytes).
pub fn tile(k: *const Kernels, s: cuda.Stream, src: u64, dst: u64, n: usize, kk: usize) !void {
    const chunks = n / 8 * (kk / 64) * 32;
    var a: cuda.Args = .{};
    a.add(src);
    a.add(dst);
    a.add(@as(c_int, @intCast(n)));
    a.add(@as(c_int, @intCast(kk)));
    try cuda.launch.launch(k.tile, .{ .grid = .{ .x = @intCast((chunks + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
}

test "K slices as the shapes take them" {
    try std.testing.expectEqual(@as(usize, 1), splitK(12288, 4096)); // 192 column tiles already
    try std.testing.expectEqual(@as(usize, 4), splitK(4096, 2048)); // 64 tiles, 32 groups: 4 slices of 8
}
