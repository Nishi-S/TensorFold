//! Bounded single-sequence verify/MTP graphs; uploads and state bookkeeping stay outside capture and replay.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const Engine = @import("cuda_engine.zig").Engine;
const policy = @import("cuda_graph_policy.zig");

const shared = cuda.graph_cache;
const Shared = shared.Cache(policy.Key);
pub const Cache = struct {
    driver: shared.CudaEngine = undefined,
    cache: ?Shared = null,
    warmed: [2]bool = .{ false, false },

    pub fn clear(c: *Cache) void {
        if (c.cache) |*v| v.deinit();
        c.cache = null;
        c.warmed = .{ false, false };
    }
};

fn agree(ptr: *anyopaque, ok: bool) !bool {
    const e: *Engine = @ptrCast(@alignCast(ptr));
    return try e.shared(if (ok) 0 else 1) == 0;
}

const Body = struct {
    f: fw.Forward,
    rows: usize,
    mtp: bool,
    fn run(ptr: *anyopaque, _: cuda.Stream) !void {
        const b: *Body = @ptrCast(@alignCast(ptr));
        if (b.mtp) try b.f.mtpPrepared(b.rows) else try b.f.runPrepared(b.rows, null);
    }
};

fn imagesOverlap(e: *const Engine, pos: usize, rows: usize) bool {
    const images = e.images orelse return false;
    if (images.slot != e.slot) return false;
    for (images.spans.items) |span| if (span.at < pos + rows and pos < @as(usize, span.at) + span.n) return true;
    return false;
}

/// Returns false before any work for unsupported calls; capture errors propagate to the rank failure machinery.
pub fn run(e: *Engine, original: fw.Forward, ids: []const i32, mtp: bool, hidden: u64) !bool {
    if (!e.settings.execution.decode_graphs or original.prof != null or original.taps != null or e.pages != null or
        original.ranks.len != 1 or original.ranks[0].segs.len != 0 or original.ranks[0].b.prefill) return false;
    const r = original.ranks[0];
    const st = r.st;
    const rows = ids.len;
    if (rows == 0 or rows > fw.window_rows) return false;
    const pos = if (mtp) st.mtp_len else st.pos;
    if (imagesOverlap(e, pos + @intFromBool(mtp), rows)) return false;
    var parity: u1 = 0;
    if (!mtp and st.cur.len > 0) {
        parity = st.cur[0];
        for (st.cur) |p| if (p != parity) return error.KdaLayersOutOfStep;
    }
    if (mtp) {
        if (r.w.mtp == null) return error.NoMtpHead;
        try st.readyMtp(e.stream, rows);
    } else {
        try st.ready(e.stream, rows);
        st.pending = rows;
        st.prompt = false;
    }
    try cuda.copy_source.enqueue(e.d, r.b.ids, std.mem.sliceAsBytes(ids), e.stream);
    if (mtp and hidden != r.b.hin) try e.copy(r.b.hin, hidden, rows * e.c.hidden * 2);
    const max_pools = if (mtp) st.mtp_caches.pk_rows - 2 else if (st.caches.len > 0) st.caches[0].pk_rows - 2 else 1024;
    const key = policy.geometry(@intFromPtr(st), if (e.group) |g| @intFromPtr(g) else 0, rows, mtp, parity, if (mtp) st.mtp_base else st.base, pos, e.c.keyWidth(), r.b.nch, max_pools);
    var f = original;
    f.images = null; // eligibility checked the actual row interval before dropping the dynamic splice
    const cache = &e.graphs;
    var body: Body = .{ .f = f, .rows = rows, .mtp = mtp };
    const mode: usize = @intFromBool(mtp);
    if (!cache.warmed[mode]) {
        try Body.run(&body, e.stream);
        cache.warmed[mode] = true;
        return true;
    }
    if (cache.cache == null) {
        cache.driver = .{ .d = e.d };
        // The rank-agreed family flag owns this cache; do not apply a second unagreed environment switch.
        cache.cache = .{ .gpa = e.gpa, .engine = cache.driver.engine(), .settings = .{ .max = 16 }, .cap = 16, .agree = .{ .ctx = e, .all = agree }, .name = "glm verify/mtp" };
    }
    _ = try cache.cache.?.run(key, e.stream, 0, .{ .ctx = &body, .run = Body.run });
    return true;
}

test {
    _ = policy;
}

test "GLM graph lifecycle: changing values replays, parity and bucket recapture, release discards executables" {
    const Fake = struct {
        held: [16]u8 = undefined,
        captures: usize = 0,
        launches: usize = 0,
        frees: usize = 0,
        votes: usize = 0,
        fn capture(ctx: *anyopaque, _: cuda.Stream, _: shared.Body) !*anyopaque {
            const v: *@This() = @ptrCast(@alignCast(ctx));
            v.captures += 1;
            return &v.held[v.captures - 1];
        }
        fn launch(ctx: *anyopaque, _: *anyopaque, _: cuda.Stream) !void {
            const v: *@This() = @ptrCast(@alignCast(ctx));
            v.launches += 1;
        }
        fn free(ctx: *anyopaque, _: *anyopaque) void {
            const v: *@This() = @ptrCast(@alignCast(ctx));
            v.frees += 1;
        }
        fn all(ctx: *anyopaque, ok: bool) !bool {
            const v: *@This() = @ptrCast(@alignCast(ctx));
            v.votes += 1;
            return ok;
        }
        fn eager(_: *anyopaque, _: cuda.Stream) !void {
            return error.UnexpectedEager;
        }
    };
    var v: Fake = .{};
    var c: Cache = .{};
    c.cache = .{ .gpa = std.testing.allocator, .engine = .{ .ctx = &v, .capture = Fake.capture, .launch = Fake.launch, .free = Fake.free }, .settings = .{ .max = 16 }, .cap = 16, .agree = .{ .ctx = &v, .all = Fake.all } };
    defer c.clear();
    const stream: cuda.Stream = .{ .d = undefined, .handle = null };
    const body: shared.Body = .{ .ctx = &v, .run = Fake.eager };
    const a = policy.geometry(1, 2, 1, false, 0, 0, 100, 2051, 16, 2048);
    try std.testing.expectEqual(shared.Outcome.captured, try c.cache.?.run(a, stream, 0, body));
    const b = policy.geometry(1, 2, 1, false, 0, 0, 101, 2051, 16, 2048);
    try std.testing.expectEqual(shared.Outcome.replayed, try c.cache.?.run(b, stream, 0, body));
    for ([_]policy.Key{
        policy.geometry(1, 2, 1, false, 1, 0, 101, 2051, 16, 2048),
        policy.geometry(1, 2, 1, false, 1, 0, 512, 2051, 16, 2048),
    }) |key| try std.testing.expectEqual(shared.Outcome.captured, try c.cache.?.run(key, stream, 0, body));
    try std.testing.expectEqual(@as(usize, 3), v.captures);
    try std.testing.expectEqual(@as(usize, 4), v.launches);
    try std.testing.expectEqual(@as(usize, 6), v.votes);
    c.clear();
    try std.testing.expectEqual(@as(usize, 3), v.frees);
    try std.testing.expect(c.cache == null);
}
