pub const Dim3 = struct { x: u32 = 1, y: u32 = 1, z: u32 = 1 };
pub const Config = struct { grid: Dim3 = .{}, block: Dim3 = .{}, shared: u32 = 0, pdl: bool = false };
pub const Stream = struct {};
pub const Function = struct {
    id: u32 = 0,
    pub fn allowDynamicShared(_: Function, _: u32) !void {}
};
pub const Module = struct {
    pub fn function(_: Module, _: []const u8) !Function {
        return .{};
    }
};
pub const Args = struct {
    values: [16]u64 = @splat(0),
    len: usize = 0,
    pub fn add(a: *Args, value: anytype) void {
        a.values[a.len] = @intCast(value);
        a.len += 1;
    }
};
pub var configs: [128]Config = undefined;
pub var symbols: [128]u32 = undefined;
pub var count: usize = 0;
pub var args: [128]Args = undefined;
pub const launch = struct {
    pub fn launch(f: Function, c: Config, _: Stream, a: *Args) !void {
        configs[count] = c;
        args[count] = a.*;
        symbols[count] = f.id;
        count += 1;
    }
};
