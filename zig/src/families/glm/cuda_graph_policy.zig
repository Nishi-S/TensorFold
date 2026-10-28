//! Graph identity covers every host-selected launch parameter; input values and positions stay device-resident.
const std = @import("std");

pub const Key = struct {
    state: usize,
    group: usize,
    rows: usize,
    mtp: bool,
    parity: u1,
    base: usize,
    chunks: usize,
    pools: usize,
    score_blocks: usize,
    dense: bool,
    sparse: bool,
    zero_embed: bool,
};

pub fn geometry(state: usize, group: usize, rows: usize, mtp: bool, parity: u1, base: usize, pos: usize, dense_limit: usize, max_chunks: usize, max_pools: usize) Key {
    const visible = (pos + rows) / 4 + 1;
    const pools = @min(max_pools, @max(1024, std.math.ceilPowerOfTwoAssert(usize, visible)));
    const columns = @min(pools, @max(512, (pos + rows) / 4));
    return .{ .state = state, .group = group, .rows = rows, .mtp = mtp, .parity = parity, .base = base, .chunks = @min((pos + rows + 511) / 512, max_chunks), .pools = pools, .score_blocks = (columns + 63) / 64, .dense = pos < dense_limit, .sparse = pos + rows - 1 >= dense_limit, .zero_embed = mtp and pos == 0 };
}

test "graph keys track attention boundaries, parity, state and ring rebasing" {
    const a = geometry(1, 2, 1, false, 0, 0, 100, 2051, 16, 2048);
    try std.testing.expectEqualDeep(a, geometry(1, 2, 1, false, 0, 0, 101, 2051, 16, 2048));
    for ([_]Key{
        geometry(3, 2, 1, false, 0, 0, 100, 2051, 16, 2048),
        geometry(1, 3, 1, false, 0, 0, 100, 2051, 16, 2048),
        geometry(1, 2, 2, false, 0, 0, 100, 2051, 16, 2048),
        geometry(1, 2, 1, false, 1, 0, 100, 2051, 16, 2048),
        geometry(1, 2, 1, false, 0, 4, 100, 2051, 16, 2048),
        geometry(1, 2, 1, false, 0, 0, 512, 2051, 16, 2048),
    }) |b| try std.testing.expect(!std.meta.eql(a, b));
    try std.testing.expect(!std.meta.eql(geometry(1, 2, 4, false, 0, 0, 2047, 2051, 16, 2048), geometry(1, 2, 4, false, 0, 0, 2048, 2051, 16, 2048)));
    try std.testing.expect(!std.meta.eql(geometry(1, 2, 1, false, 0, 0, 2306, 2051, 16, 2048), geometry(1, 2, 1, false, 0, 0, 2307, 2051, 16, 2048)));
    try std.testing.expect(!std.meta.eql(geometry(1, 2, 1, true, 0, 0, 0, 2051, 16, 2048), geometry(1, 2, 1, true, 0, 0, 1, 2051, 16, 2048)));
}
