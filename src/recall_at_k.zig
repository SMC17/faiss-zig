//! recall_at_k: v0.0.6 — recall@K substrate.
//!
//! Index variants (HNSW, IVF, IVFPQ) trade recall for memory and
//! speed. README captures the trade qualitatively; this primitive
//! provides the programmatic check so a caller can validate
//! their deployment against an L2-Flat baseline.
//!
//! recall@K = |ground_truth_top_k INTERSECT estimated_top_k| / K
//!
//! Useful: a CI gate that fails if IVFPQ recall@10 falls below
//! 0.6 on a sampled query set; a dashboard that tracks recall
//! drift over time; an internal qualifier when citing the
//! 16.94x compression headline that L120 flagged.
//!
//! Scope of this revision:
//!   recallAtK(truth, estimate) returns the fraction of the
//!     estimate that hits the truth top-K. Both slices must have
//!     equal length; order within each slice does not matter.
//!   meanRecallOverQueries(truths, estimates) averages recallAtK
//!     across paired query results.
//!
//! Out of scope: NDCG, MRR, mAP (caller composes from id sets
//! when needed); partial-recall weighting by score (caller
//! supplies a relevance score per id).

const std = @import("std");
const root = @import("root.zig");

pub const Hit = root.Hit;

pub const Error = error{
    LengthMismatch,
    EmptySlice,
};

/// Fraction of `estimate` hits that appear in `truth`. Both must
/// be equal length. Order ignored. Returns 0 when truth is empty.
pub fn recallAtK(truth: []const Hit, estimate: []const Hit) Error!f64 {
    if (truth.len != estimate.len) return Error.LengthMismatch;
    if (truth.len == 0) return 0.0;
    var hits: usize = 0;
    for (estimate) |e| {
        for (truth) |t| {
            if (e.id == t.id) {
                hits += 1;
                break;
            }
        }
    }
    return @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(truth.len));
}

/// Mean recall across paired query results. truths.len ==
/// estimates.len; each pair must satisfy recallAtK constraints.
pub fn meanRecallOverQueries(
    truths: []const []const Hit,
    estimates: []const []const Hit,
) Error!f64 {
    if (truths.len != estimates.len) return Error.LengthMismatch;
    if (truths.len == 0) return Error.EmptySlice;
    var sum: f64 = 0.0;
    var i: usize = 0;
    while (i < truths.len) : (i += 1) {
        sum += try recallAtK(truths[i], estimates[i]);
    }
    return sum / @as(f64, @floatFromInt(truths.len));
}

// --- Tests -----------------------------------------------------------------

const testing = std.testing;

test "recallAtK on identical slices is 1.0" {
    const t = [_]Hit{
        .{ .id = 1, .score = 0.0 },
        .{ .id = 2, .score = 0.0 },
        .{ .id = 3, .score = 0.0 },
    };
    try testing.expectApproxEqAbs(@as(f64, 1.0), try recallAtK(&t, &t), 1e-12);
}

test "recallAtK on disjoint slices is 0.0" {
    const t = [_]Hit{ .{ .id = 1, .score = 0.0 }, .{ .id = 2, .score = 0.0 } };
    const e = [_]Hit{ .{ .id = 5, .score = 0.0 }, .{ .id = 6, .score = 0.0 } };
    try testing.expectEqual(@as(f64, 0.0), try recallAtK(&t, &e));
}

test "recallAtK on half-overlap is 0.5" {
    const t = [_]Hit{
        .{ .id = 1, .score = 0.0 },
        .{ .id = 2, .score = 0.0 },
        .{ .id = 3, .score = 0.0 },
        .{ .id = 4, .score = 0.0 },
    };
    const e = [_]Hit{
        .{ .id = 1, .score = 0.0 },
        .{ .id = 2, .score = 0.0 },
        .{ .id = 99, .score = 0.0 },
        .{ .id = 98, .score = 0.0 },
    };
    try testing.expectApproxEqAbs(@as(f64, 0.5), try recallAtK(&t, &e), 1e-12);
}

test "recallAtK is order-invariant" {
    const t = [_]Hit{
        .{ .id = 1, .score = 0.0 },
        .{ .id = 2, .score = 0.0 },
        .{ .id = 3, .score = 0.0 },
    };
    const e_reversed = [_]Hit{
        .{ .id = 3, .score = 0.0 },
        .{ .id = 2, .score = 0.0 },
        .{ .id = 1, .score = 0.0 },
    };
    try testing.expectApproxEqAbs(@as(f64, 1.0), try recallAtK(&t, &e_reversed), 1e-12);
}

test "recallAtK rejects length mismatch" {
    const t = [_]Hit{ .{ .id = 1, .score = 0.0 } };
    const e = [_]Hit{ .{ .id = 1, .score = 0.0 }, .{ .id = 2, .score = 0.0 } };
    try testing.expectError(Error.LengthMismatch, recallAtK(&t, &e));
}

test "meanRecallOverQueries averages per-query recall" {
    const t1 = [_]Hit{ .{ .id = 1, .score = 0.0 }, .{ .id = 2, .score = 0.0 } };
    const e1 = [_]Hit{ .{ .id = 1, .score = 0.0 }, .{ .id = 2, .score = 0.0 } };
    const t2 = [_]Hit{ .{ .id = 5, .score = 0.0 }, .{ .id = 6, .score = 0.0 } };
    const e2 = [_]Hit{ .{ .id = 5, .score = 0.0 }, .{ .id = 99, .score = 0.0 } };
    const truths: [2][]const Hit = .{ &t1, &t2 };
    const estimates: [2][]const Hit = .{ &e1, &e2 };
    const m = try meanRecallOverQueries(&truths, &estimates);
    try testing.expectApproxEqAbs(@as(f64, 0.75), m, 1e-12);
}

test "meanRecallOverQueries rejects empty" {
    const empty: [0][]const Hit = .{};
    try testing.expectError(Error.EmptySlice, meanRecallOverQueries(&empty, &empty));
}

test "recallAtK ignores duplicates in estimate (each truth counted once)" {
    const t = [_]Hit{ .{ .id = 1, .score = 0.0 }, .{ .id = 2, .score = 0.0 } };
    const e = [_]Hit{ .{ .id = 1, .score = 0.0 }, .{ .id = 1, .score = 0.0 } };
    try testing.expectApproxEqAbs(@as(f64, 1.0), try recallAtK(&t, &e), 1e-12);
}
