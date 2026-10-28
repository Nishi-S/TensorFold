//! Independent FP64 recurrence checks the blocked WY equations, including zero decay and partial sub-chunks.
const std = @import("std");
const d = 8;
const values = 5;
const Row = struct { q: [d]f64, k: [d]f64, g: [d]f64, v: [values]f64, beta: f64 };
const State = [values][d]f64;

fn serial(s: *State, rows: []const Row, out: [][values]f64) void {
    for (rows, out) |r, *o| for (s, 0..) |*v, j| {
        var dot: f64 = 0;
        for (v, r.g, r.k) |*x, decay, key| {
            x.* *= decay;
            dot += x.* * key;
        }
        const delta = r.beta * (r.v[j] - dot);
        o[j] = 0;
        for (v, r.k, r.q) |*x, key, query| {
            x.* += key * delta;
            o[j] += x.* * query;
        }
    };
}

fn blocked(s: *State, rows: []const Row, out: [][values]f64, position: usize) void {
    var first: usize = 0;
    while (first < rows.len) {
        const n = @min(rows.len - first, 32 - (position + first) % 32);
        const block = rows[first..][0..n];
        var prefix: [32][d]f64 = undefined;
        var l: [32][32]f64 = @splat(@splat(0));
        var m: [32][32]f64 = @splat(@splat(0));
        var delta: [32][values]f64 = undefined;
        var end: [d]f64 = @splat(1);
        for (block, 0..) |r, i| {
            for (&end, r.g, &prefix[i]) |*e, decay, *p| {
                e.* *= decay;
                p.* = e.*;
            }
            for (0..i + 1) |t| for (0..d) |c| {
                var decay: f64 = 1;
                for (t + 1..i + 1) |j| decay *= block[j].g[c];
                const ke = block[t].k[c] * decay;
                if (t < i) l[i][t] += r.beta * r.k[c] * ke;
                m[i][t] += r.q[c] * ke;
            };
        }
        for (0..values) |v| for (block, 0..) |r, i| {
            var dot: f64 = 0;
            for (0..d) |c| dot += r.k[c] * prefix[i][c] * s[v][c];
            var y = r.beta * (r.v[v] - dot);
            for (0..i) |t| y -= l[i][t] * delta[t][v];
            delta[i][v] = y;
            var o: f64 = 0;
            for (0..d) |c| o += r.q[c] * prefix[i][c] * s[v][c];
            for (0..i + 1) |t| o += m[i][t] * delta[t][v];
            out[first + i][v] = o;
        };
        for (0..values) |v| for (0..d) |c| {
            s[v][c] *= end[c];
            for (block, 0..) |r, i| {
                var tail: f64 = 1;
                for (i + 1..n) |j| tail *= block[j].g[c];
                s[v][c] += delta[i][v] * r.k[c] * tail;
            }
        };
        first += n;
    }
}

test "KDA WY32: independent FP64 serial recurrence and aligned continuation" {
    var rng = std.Random.DefaultPrng.init(0x57493332);
    const random = rng.random();
    var rows: [97]Row = undefined;
    for (&rows, 0..) |*r, i| {
        r.beta = random.float(f64);
        for (&r.q, &r.k, &r.g, 0..) |*q, *k, *g, c| {
            q.* = (random.float(f64) - 0.5) / 4;
            k.* = (random.float(f64) - 0.5) / 2;
            g.* = if ((i + c) % 17 == 0) 0 else if ((i + c) % 19 == 0) 1 else random.float(f64);
        }
        for (&r.v) |*v| v.* = random.float(f64) - 0.5;
    }
    var initial: State = undefined;
    for (&initial) |*row| for (row) |*v| {
        v.* = random.float(f64) - 0.5;
    };
    for ([_]usize{ 1, 31, 32, 33, 63, 64, 65, 97 }) |n| for ([_]usize{ 0, 7, 31 }) |position| {
        var expected = initial;
        var candidate = initial;
        var a: [97][values]f64 = undefined;
        var b: [97][values]f64 = undefined;
        serial(&expected, rows[0..n], a[0..n]);
        blocked(&candidate, rows[0..n], b[0..n], position);
        for (a[0..n], b[0..n]) |x, y| for (x, y) |u, v| try std.testing.expectApproxEqAbs(u, v, 1e-13);
        for (expected, candidate) |x, y| for (x, y) |u, v| try std.testing.expectApproxEqAbs(u, v, 1e-13);
        if (n > 32) {
            var resumed = initial;
            const cut = 32 - position % 32;
            var c: [97][values]f64 = undefined;
            blocked(&resumed, rows[0..cut], c[0..cut], position);
            blocked(&resumed, rows[cut..n], c[cut..n], position + cut);
            try std.testing.expectEqualDeep(candidate, resumed);
            try std.testing.expectEqualSlices([values]f64, b[0..n], c[0..n]);
        }
    };
}
