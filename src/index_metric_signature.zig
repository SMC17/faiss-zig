//! index_metric_signature: v0.0.7 — per-index metadata bundle.
//!
//! FlatIndex carries (dim, metric); HNSW adds (M, efC, efS);
//! IVFPQ adds (nlist, nprobe, M). Callers writing an index
//! catalog or persistence layer want a unified signature record
//! that captures the minimum identifying metadata for any
//! index variant so they can route to the right loader.
//!
//! Scope of this revision:
//!   IndexKind enum { flat, hnsw, ivf, ivf_pq }.
//!   IndexSignature record carrying kind + dim + metric +
//!     variant-specific u32 params.
//!   forFlat / forHnsw / forIvf / forIvfPq constructors.
//!   description(sig, buf) writes a single-line "flat d=128
//!     l2" or "ivf_pq d=128 l2 nlist=100 M=8" caller-readable
//!     string into buf, returns the written slice.
//!   matches(a, b) returns true iff every load-bearing field
//!     equals.
//!
//! Out of scope: serialization (caller composes with tableformat
//! or wflog Triple at the boundary), version negotiation between
//! signature schema revisions, multi-language naming.

const std = @import("std");
const root = @import("root.zig");

pub const Metric = root.Metric;

pub const IndexKind = enum(u8) {
    flat = 0,
    hnsw = 1,
    ivf = 2,
    ivf_pq = 3,
};

pub const IndexSignature = struct {
    kind: IndexKind,
    dim: u32,
    metric: Metric,
    param_a: u32 = 0,
    param_b: u32 = 0,
    param_c: u32 = 0,
};

pub fn forFlat(dim: u32, metric: Metric) IndexSignature {
    return .{ .kind = .flat, .dim = dim, .metric = metric };
}

pub fn forHnsw(dim: u32, metric: Metric, m: u32, ef_c: u32, ef_s: u32) IndexSignature {
    return .{
        .kind = .hnsw,
        .dim = dim,
        .metric = metric,
        .param_a = m,
        .param_b = ef_c,
        .param_c = ef_s,
    };
}

pub fn forIvf(dim: u32, metric: Metric, nlist: u32, nprobe: u32) IndexSignature {
    return .{
        .kind = .ivf,
        .dim = dim,
        .metric = metric,
        .param_a = nlist,
        .param_b = nprobe,
    };
}

pub fn forIvfPq(dim: u32, metric: Metric, nlist: u32, nprobe: u32, m: u32) IndexSignature {
    return .{
        .kind = .ivf_pq,
        .dim = dim,
        .metric = metric,
        .param_a = nlist,
        .param_b = nprobe,
        .param_c = m,
    };
}

pub fn matches(a: IndexSignature, b: IndexSignature) bool {
    return a.kind == b.kind and a.dim == b.dim and a.metric == b.metric and a.param_a == b.param_a and a.param_b == b.param_b and a.param_c == b.param_c;
}

fn metricStr(m: Metric) []const u8 {
    return switch (m) {
        .l2_squared => "l2",
        .inner_product => "ip",
        .cosine => "cos",
    };
}

pub fn description(sig: IndexSignature, buf: []u8) ![]u8 {
    return switch (sig.kind) {
        .flat => try std.fmt.bufPrint(buf, "flat d={d} {s}", .{ sig.dim, metricStr(sig.metric) }),
        .hnsw => try std.fmt.bufPrint(buf, "hnsw d={d} {s} M={d} efC={d} efS={d}", .{ sig.dim, metricStr(sig.metric), sig.param_a, sig.param_b, sig.param_c }),
        .ivf => try std.fmt.bufPrint(buf, "ivf d={d} {s} nlist={d} nprobe={d}", .{ sig.dim, metricStr(sig.metric), sig.param_a, sig.param_b }),
        .ivf_pq => try std.fmt.bufPrint(buf, "ivf_pq d={d} {s} nlist={d} nprobe={d} M={d}", .{ sig.dim, metricStr(sig.metric), sig.param_a, sig.param_b, sig.param_c }),
    };
}

// --- Tests -----------------------------------------------------------------

const testing = std.testing;

test "forFlat carries kind dim metric" {
    const s = forFlat(128, .l2_squared);
    try testing.expectEqual(IndexKind.flat, s.kind);
    try testing.expectEqual(@as(u32, 128), s.dim);
    try testing.expectEqual(Metric.l2_squared, s.metric);
}

test "forHnsw carries M efC efS" {
    const s = forHnsw(64, .cosine, 16, 200, 100);
    try testing.expectEqual(IndexKind.hnsw, s.kind);
    try testing.expectEqual(@as(u32, 16), s.param_a);
    try testing.expectEqual(@as(u32, 200), s.param_b);
    try testing.expectEqual(@as(u32, 100), s.param_c);
}

test "forIvf carries nlist nprobe" {
    const s = forIvf(32, .inner_product, 100, 10);
    try testing.expectEqual(IndexKind.ivf, s.kind);
    try testing.expectEqual(@as(u32, 100), s.param_a);
    try testing.expectEqual(@as(u32, 10), s.param_b);
}

test "forIvfPq carries nlist nprobe M" {
    const s = forIvfPq(128, .l2_squared, 100, 10, 8);
    try testing.expectEqual(@as(u32, 8), s.param_c);
}

test "matches true on identical signatures" {
    const a = forFlat(128, .l2_squared);
    const b = forFlat(128, .l2_squared);
    try testing.expect(matches(a, b));
}

test "matches false on differing kind" {
    const a = forFlat(128, .l2_squared);
    const b = forHnsw(128, .l2_squared, 16, 200, 100);
    try testing.expect(!matches(a, b));
}

test "matches false on differing dim" {
    const a = forFlat(128, .l2_squared);
    const b = forFlat(64, .l2_squared);
    try testing.expect(!matches(a, b));
}

test "description prints flat" {
    var buf: [128]u8 = undefined;
    const s = forFlat(128, .l2_squared);
    const out = try description(s, &buf);
    try testing.expectEqualStrings("flat d=128 l2", out);
}

test "description prints ivf_pq with all params" {
    var buf: [128]u8 = undefined;
    const s = forIvfPq(128, .l2_squared, 100, 10, 8);
    const out = try description(s, &buf);
    try testing.expectEqualStrings("ivf_pq d=128 l2 nlist=100 nprobe=10 M=8", out);
}
