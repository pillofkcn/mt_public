"""Version 1.0 statistics: the paired cluster bootstrap, and the one way a CI is read.

Part D of the version 1.0 brief states the whole statistical procedure, and
this module is all of it:

  1. Estimate. Delta = M(A) - M(B) in points, M being a task's corpus metric
     recomputed from summed per-sentence counts. `metric` computes it, and on
     the same counts it returns exactly what metrics.corpus_metric returns.
  2. Interval. A paired percentile bootstrap: the units of a test set
     (documents, files or sentences - plan datasets.<id>.resampling.unit) are
     drawn with replacement, by ONE matrix of unit indices that every
     configuration scored on the set shares (`cluster_bootstrap`), and in each
     replicate the metric is recomputed from the summed counts. B, the
     confidence level and the seed are the plan's.
  3. A difference of differences (H1b, H2a) is taken from the same replicates
     as its two Deltas (`diff_of_deltas_ci`).
  4. Two test sets (H2b) are resampled independently, and the drop M_in - M_out
     and Delta_step,max are taken in the same replicate, the step from the
     in-domain one (`two_set_drop_ci`).
  5. The threshold delta and its sensitivity values are the plan's; the
     caller passes them.
  6. An interval is read against delta by one function, `classify`, and a
     hypothesis part by its rule, `decide`.
  7. No p-value, no correction, no other test: there is none here.
  8. Every Estimate carries the half-width of its CI.

The interval is the percentile interval with numpy's default quantile, method
"linear" (definition 7 of Hyndman and Fan 1996): for probability p and B sorted
replicates x_0 <= ... <= x_{B-1}, h = (B - 1) p and the quantile is
x_floor(h) + (h - floor(h)) (x_floor(h)+1 - x_floor(h)). With B = 10 000 and
95 % the bounds lie between the 250th and 251st, and between the 9750th and
9751st, smallest replicates.

Seeds. The plan holds one seed. Each test set and each choice of unit gets a
stream of its own, `seed_sequence(seed, dataset, unit)`: a numpy SeedSequence
with spawn_key (crc32 of the dataset id, crc32 of the unit). crc32 because it
is the same on every machine and in every process; Python's hash() of a string
is salted per process and would draw other indices in every run. So one set
always gets the same indices - H1b's two legs and every contrast on one set
share them - and two sets get independent ones, which is what H2b needs.

Memory. A replicate is computed as W @ C: W holds how many times each unit was
drawn in each replicate, C the per-unit counts of every configuration side by
side. W is built a block of replicates at a time (CHUNK_CELLS index cells per
block), so a set of 5000 units at B = 10 000 needs under 100 MB at once (83 MB
measured with 16 configurations of 6 counts) rather than the 400 MB of the
whole index matrix. Drawing the matrix a block of rows
at a time gives the same indices as drawing it at once (PCG64 keeps its spare
32-bit half in the generator's state, not in the call); a test holds it to
that.

Added 23 Oct 2026 (version 1.0, step 6). It replaces the bootstrap functions of
evaluation/stats.py, which version 1.0 removed together with the scripts that
imported it (step 12).
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Hashable, Mapping, NamedTuple, Sequence

import numpy as np

__all__ = [
    "CLASSES",
    "ABOVE_ZERO",
    "EQUIVALENCE",
    "SUPERIORITY",
    "RULES",
    "METRIC_COLUMNS",
    "CHUNK_CELLS",
    "SNAP_DECIMALS",
    "Estimate",
    "ClusterSums",
    "stable_id",
    "seed_sequence",
    "unit_indices",
    "cluster_bootstrap",
    "metric",
    "percentile_ci",
    "delta_ci",
    "diff_of_deltas_ci",
    "two_set_drop_ci",
    "classify",
    "decide",
]

# The six places a 95 % CI can lie relative to 0 and +-delta (brief D.6), from
# best to worst. Every interval gets exactly one of them.
CLASSES: tuple[str, ...] = (
    "better_by_more_than_delta",   # ci_low > +delta
    "better_not_shown_by_delta",   # ci_low > 0, and the CI reaches +delta
    "practically_equivalent",      # -delta < ci_low and ci_high < +delta
    "inconclusive",                # the CI holds 0 and reaches -delta or +delta
    "worse_not_shown_by_delta",    # ci_high < 0, and the CI reaches -delta
    "worse_by_more_than_delta",    # ci_high < -delta
)

# The rules a hypothesis part is decided by. H1a, H1b, H2a and H2b hold when
# the lower bound of the CI lies above 0 (brief part A); the two H2c rules are
# the ones plan statistics.decisions names, under the plan's own names.
ABOVE_ZERO = "ci_lower_bound_above_zero"
EQUIVALENCE = "ci_within_plus_minus_delta"
SUPERIORITY = "ci_lower_bound_above_delta"
RULES: tuple[str, ...] = (ABOVE_ZERO, EQUIVALENCE, SUPERIORITY)

# The counts each metric reads, by the field names of metrics.PosCounts,
# metrics.DepCounts and evalb.ConCounts (and so by the column names of a
# counts CSV). The metric names are the plan's.
METRIC_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "upos_accuracy": ("correct", "total"),
    "las": ("las_correct", "total"),
    "uas": ("uas_correct", "total"),
    "clas": ("clas_correct", "clas_total", "clas_pred_total"),
    "bracket_f1": ("matched", "gold", "test"),
}

# Index cells per block of replicates in cluster_bootstrap: 2**21 cells is
# 16 MB of int64 indices, and the weights built from them three times that.
CHUNK_CELLS = 1 << 21


# --- the estimate ------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    """A point estimate with its percentile CI, in points.

    Attributes:
        point:      The estimate on the whole set.
        ci_low:     Lower bound of the CI.
        ci_high:    Upper bound of the CI.
        replicates: The replicate values the CI was read from, kept so a
                    statistic built on this one (a difference of Deltas, a
                    drop minus a step) can be taken from the same replicates.
                    Not compared and not printed.

    `half_width`, (ci_high - ci_low) / 2, is what brief D.8 asks to be
    reported with every contrast: how precisely the set measures it.
    """

    point: float
    ci_low: float
    ci_high: float
    replicates: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def half_width(self) -> float:
        """Half the width of the CI, in points."""
        return (self.ci_high - self.ci_low) / 2


class ClusterSums(NamedTuple):
    """What cluster_bootstrap returns, keyed as its input was.

    Attributes:
        replicates: key -> (B x k) int64 array, the column sums of each
                    replicate.
        points:     key -> (k,) int64 array, the column sums over every unit
                    once: the counts of the point estimate.
    """

    replicates: dict[Hashable, np.ndarray]
    points: dict[Hashable, np.ndarray]


# --- seeds ------------------------------------------------------------------


def stable_id(text: str) -> int:
    """A number for `text` that is the same in every process: crc32 of its UTF-8."""
    return zlib.crc32(text.encode("utf-8"))


def seed_sequence(seed: int, dataset: str, unit: str) -> np.random.SeedSequence:
    """The stream of one test set resampled by one unit.

    np.random.SeedSequence(seed, spawn_key=(stable_id(dataset), stable_id(unit))):
    the plan's one seed, and a key that makes the stream of every
    (dataset, unit) independent of every other's.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, not {seed!r}")
    return np.random.SeedSequence(seed, spawn_key=(stable_id(dataset), stable_id(unit)))


def _generator(seed: int | np.random.SeedSequence) -> np.random.Generator:
    if isinstance(seed, np.random.SeedSequence):
        return np.random.Generator(np.random.PCG64(seed))
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer or a SeedSequence, not {seed!r}")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed))))


def unit_indices(n_units: int, replicates: int, seed: int | np.random.SeedSequence) -> np.ndarray:
    """The whole B x n_units matrix of unit indices cluster_bootstrap draws.

    For a check, or a small set: cluster_bootstrap never holds it at once, but
    draws exactly these rows a block at a time.
    """
    _check_sizes(n_units, replicates)
    return _generator(seed).integers(0, n_units, size=(replicates, n_units))


def _check_sizes(n_units: int, replicates: int) -> None:
    if n_units < 1:
        raise ValueError(f"nothing to resample: {n_units} units")
    if replicates < 1:
        raise ValueError(f"replicates must be >= 1, not {replicates}")


# --- the bootstrap -----------------------------------------------------------


def cluster_bootstrap(
    counts_by_unit: Mapping[Hashable, np.ndarray],
    B: int,
    seed: int | np.random.SeedSequence,
    *,
    chunk_cells: int = CHUNK_CELLS,
) -> ClusterSums:
    """Resample the units of one test set, once for every configuration on it.

    Args:
        counts_by_unit: configuration -> 2-D integer array, one row per unit
            (the per-sentence counts of the unit summed), one column per count.
            Every array has the same units in the same order; the columns may
            differ (a POS and a dependency configuration of one set side by
            side, as H1b needs).
        B: Replicates.
        seed: The stream: seed_sequence(plan seed, dataset, unit), or an int.
        chunk_cells: Index cells per block of replicates (memory, not result).

    Returns:
        ClusterSums: per key, the (B x k) column sums of every replicate and
        the (k,) column sums of the set.

    One B x n_units matrix of indices is drawn with replacement
    (np.random.Generator(PCG64)) and shared by every key: brief D.2, the same
    indices for every tool of one set, which is what makes the bootstrap paired.
    Each replicate is a sum of counts, so every metric of it is computed by
    the same arithmetic as the point estimate.

    Raises:
        ValueError: If there is nothing to resample, an array is not 2-D or not
            integer or holds a negative count, or two arrays differ in units.
    """
    if not counts_by_unit:
        raise ValueError("no counts to resample")
    keys = list(counts_by_unit)
    arrays = []
    for key in keys:
        array = np.asarray(counts_by_unit[key])
        if array.ndim != 2:
            raise ValueError(f"counts of {key!r} must be 2-D (units x counts), not {array.ndim}-D")
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"counts of {key!r} must be integers, not {array.dtype}")
        if array.size and array.min() < 0:
            raise ValueError(f"counts of {key!r} hold a negative count")
        arrays.append(array.astype(np.int64))
    n_units = arrays[0].shape[0]
    for key, array in zip(keys, arrays):
        if array.shape[0] != n_units:
            raise ValueError(
                f"counts of {key!r} have {array.shape[0]} units, those of {keys[0]!r} {n_units}"
            )
    _check_sizes(n_units, B)
    if chunk_cells < 1:
        raise ValueError(f"chunk_cells must be >= 1, not {chunk_cells}")

    widths = [array.shape[1] for array in arrays]
    stacked = np.concatenate(arrays, axis=1)
    # A replicate sum is at most n_units times the largest unit count. Integer
    # sums below 2**53 are exact in float64 whatever order BLAS adds them in,
    # so the product is exact and rint only drops a ".0".
    if int(stacked.max(initial=0)) * n_units >= 2**53:
        raise ValueError("counts too large to sum exactly")
    as_float = stacked.astype(np.float64)
    rng = _generator(seed)
    rows_per_block = max(1, chunk_cells // n_units)
    sums = np.empty((B, stacked.shape[1]), dtype=np.int64)
    for start in range(0, B, rows_per_block):
        rows = min(rows_per_block, B - start)
        indices = rng.integers(0, n_units, size=(rows, n_units))
        flat = (indices + (np.arange(rows, dtype=np.int64) * n_units)[:, None]).ravel()
        weights = np.bincount(flat, minlength=rows * n_units).reshape(rows, n_units)
        sums[start:start + rows] = np.rint(weights.astype(np.float64) @ as_float).astype(np.int64)

    point = stacked.sum(axis=0)
    replicates, points = {}, {}
    offset = 0
    for key, width in zip(keys, widths):
        replicates[key] = sums[:, offset:offset + width]
        points[key] = point[offset:offset + width]
        offset += width
    return ClusterSums(replicates=replicates, points=points)


# --- metrics from summed counts ----------------------------------------------


def metric(name: str, columns: Sequence[str], sums: np.ndarray) -> np.ndarray:
    """A corpus metric, in points, from summed counts: one value per row of `sums`.

    Args:
        name: upos_accuracy, las, uas, clas or bracket_f1 (METRIC_COLUMNS).
        columns: The names of the columns of `sums`, in order.
        sums: (..., k) summed counts: the point sums (k,) give a 0-d array,
            the replicate sums (B x k) one value per replicate.

    The arithmetic is metrics.corpus_metric's, operation for operation, so the
    same counts give the same float, not merely a close one:

      * upos_accuracy, las, uas: 100 * correct / total;
      * clas: 100 * (2 * clas_correct) / (clas_total + clas_pred_total), the
        F1 of conll18_ud_eval.py over gold and predicted content words;
      * bracket_f1: 100 * 2 * P * R / (P + R) with P = matched / test (0 when
        test is 0) and R = matched / gold (0 when gold is 0); 0 when P + R is
        0. An EVALB error sentence is all zeros in matched, gold and test
        (evalb.ConCounts), so it adds nothing, as in corpus_metric.

    A denominator of 0 (gold + test for bracket_f1) is NaN: not measured.

    Raises:
        ValueError: For an unknown metric, or columns that lack its counts.
    """
    if name not in METRIC_COLUMNS:
        raise ValueError(f"unknown metric {name!r}; expected one of {list(METRIC_COLUMNS)}")
    columns = list(columns)
    missing = [column for column in METRIC_COLUMNS[name] if column not in columns]
    if missing:
        raise ValueError(f"metric {name} needs the counts {missing}; the columns are {columns}")
    sums = np.asarray(sums)
    if sums.shape[-1] != len(columns):
        raise ValueError(f"{len(columns)} column names for {sums.shape[-1]} columns")
    parts = [sums[..., columns.index(column)].astype(np.float64) for column in METRIC_COLUMNS[name]]
    with np.errstate(divide="ignore", invalid="ignore"):
        if name == "clas":
            correct, gold, predicted = parts
            return _points(2 * correct, gold + predicted)
        if name == "bracket_f1":
            return _bracket_f1(*parts)
        return _points(*parts)


def _points(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """100 * numerator / denominator, NaN where the denominator is 0."""
    value = 100 * numerator / denominator
    return np.where(denominator == 0, np.nan, value)


def _bracket_f1(matched: np.ndarray, gold: np.ndarray, test: np.ndarray) -> np.ndarray:
    """corpus_metric's bracket F1, element by element."""
    precision = np.where(test > 0, matched / test, 0.0)
    recall = np.where(gold > 0, matched / gold, 0.0)
    f1 = 100 * 2 * precision * recall / (precision + recall)
    f1 = np.where(precision + recall == 0, 0.0, f1)
    return np.where(gold + test == 0, np.nan, f1)


# --- intervals ---------------------------------------------------------------


def percentile_ci(values: np.ndarray, confidence: float) -> tuple[float, float]:
    """The percentile interval of replicate values, numpy's default (linear) quantile.

    Raises:
        ValueError: If `values` is empty or not 1-D, holds NaN or infinity (a
            replicate the metric could not measure), or `confidence` is not in
            (0, 1).
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"replicates must be a non-empty 1-D array, not shape {values.shape}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must lie in (0, 1), not {confidence}")
    if not np.isfinite(values).all():
        bad = int((~np.isfinite(values)).sum())
        raise ValueError(f"{bad} of {values.size} replicates are not finite (a metric with nothing to measure)")
    # 1 - 0.95 is 0.050000000000000044 in floating point; rounded, 95 % is
    # read at the probabilities 0.025 and 0.975 themselves.
    alpha = round(1 - confidence, 12)
    low, high = np.quantile(values, (alpha / 2, 1 - alpha / 2))
    return float(low), float(high)


# Decimals an estimate and its replicates are rounded to before the interval is
# read. The values are points (|x| <= 200), and a difference that is exactly 0
# in rational arithmetic - two Deltas of the same counts, as H1b, H2a and H2b
# take them - comes out of floating point as +-1.4e-14 about as often as 0. A
# bound lying exactly on 0 or +-delta would then be decided by that noise, not
# by the rule that a bound equal to a threshold does not lie beyond it
# (`classify`). 1e-9 is far above the noise and far below anything written
# (4 decimals).
SNAP_DECIMALS = 9


def _estimate(point: float, replicates: np.ndarray, confidence: float) -> Estimate:
    point = round(float(point), SNAP_DECIMALS)
    if not math.isfinite(point):
        raise ValueError(f"the point estimate is {point}: a metric with nothing to measure")
    replicates = np.round(np.asarray(replicates, dtype=np.float64), SNAP_DECIMALS)
    low, high = percentile_ci(replicates, confidence)
    return Estimate(point=point, ci_low=low, ci_high=high, replicates=replicates)


def delta_ci(
    rep_a: np.ndarray, rep_b: np.ndarray, point_a: float, point_b: float, confidence: float
) -> Estimate:
    """Delta = M(a) - M(b): the point difference, and the CI of the replicate differences.

    `rep_a` and `rep_b` are the metric of a and of b in the same replicates:
    from one cluster_bootstrap of one set, or (for a drop between two sets,
    which H2b and the descriptive drop need) from two independent ones, paired
    by replicate number.
    """
    rep_a = np.asarray(rep_a, dtype=np.float64)
    rep_b = np.asarray(rep_b, dtype=np.float64)
    if rep_a.shape != rep_b.shape:
        raise ValueError(f"replicates of shapes {rep_a.shape} and {rep_b.shape}")
    return _estimate(float(point_a) - float(point_b), rep_a - rep_b, confidence)


def diff_of_deltas_ci(
    first: Estimate, second: Estimate, confidence: float, *, absolute_first: bool = False
) -> Estimate:
    """first - second, both taken from the same replicate (brief D.3).

    H1b: Delta_LAS - Delta_UPOS of the step stanza_comb -> stanza_trf on the
    same documents. H2a: |Delta_source| - Delta_step,max, with
    `absolute_first`: the size of the training-data effect, whatever its
    sign, against the largest H1a step on the same set, both in each replicate.

    Raises:
        ValueError: If either estimate has no replicates, or they differ in number.
    """
    if first.replicates is None or second.replicates is None:
        raise ValueError("both estimates need their replicates")
    if first.replicates.shape != second.replicates.shape:
        raise ValueError(
            f"replicates of shapes {first.replicates.shape} and {second.replicates.shape}"
        )
    if absolute_first:
        return _estimate(
            abs(first.point) - second.point,
            np.abs(first.replicates) - second.replicates,
            confidence,
        )
    return _estimate(first.point - second.point, first.replicates - second.replicates, confidence)


def two_set_drop_ci(
    rep_in: np.ndarray,
    rep_out: np.ndarray,
    point_in: float,
    point_out: float,
    step: Estimate,
    confidence: float,
) -> tuple[Estimate, Estimate]:
    """H2b: the drop M_in - M_out, and the drop minus Delta_step,max (brief D.4).

    Args:
        rep_in, point_in: The model's metric on the source set, from the source
            set's bootstrap.
        rep_out, point_out: The same on the target set, from the target set's
            bootstrap, an independent stream (seed_sequence of another dataset).
        step: Delta_step,max as an Estimate from the SOURCE set's bootstrap,
            so its replicate b is from the same in-domain replicate as rep_in[b].
        confidence: The CI level.

    Returns:
        (drop, stat): the drop with its CI, and drop - Delta_step,max with its
        CI, both from the same replicates.
    """
    drop = delta_ci(rep_in, rep_out, point_in, point_out, confidence)
    return drop, diff_of_deltas_ci(drop, step, confidence)


# --- reading an interval -----------------------------------------------------


def _bounds(ci: Estimate | Sequence[float]) -> tuple[float, float]:
    if isinstance(ci, Estimate):
        low, high = ci.ci_low, ci.ci_high
    else:
        low, high = ci
    low, high = float(low), float(high)
    if math.isnan(low) or math.isnan(high):
        raise ValueError(f"an interval with NaN: ({low}, {high})")
    if low > high:
        raise ValueError(f"an interval whose lower bound is above its upper: ({low}, {high})")
    return low, high


def classify(ci: Estimate | Sequence[float], delta: float) -> str:
    """Where a CI lies relative to 0 and +-delta: one of CLASSES (brief D.6).

    In this order, so every interval gets exactly one class:

      better_by_more_than_delta   ci_low > +delta
      worse_by_more_than_delta    ci_high < -delta
      practically_equivalent      -delta < ci_low and ci_high < +delta
      better_not_shown_by_delta   ci_low > 0 (the CI then reaches +delta)
      worse_not_shown_by_delta    ci_high < 0 (the CI then reaches -delta)
      inconclusive                the rest: the CI holds 0 and reaches
                                  -delta or +delta

    A bound equal to a threshold does not lie beyond it: ci_low = +delta is
    better_not_shown_by_delta, ci_low = 0 with ci_high >= +delta inconclusive.
    Every Estimate of this module is rounded to SNAP_DECIMALS before its
    interval is read, so a bound that is exactly 0 is 0.0 and not the
    floating-point noise of the subtraction that gave it.

    Raises:
        ValueError: If delta is not positive or the interval is not one.
    """
    if not delta > 0:
        raise ValueError(f"delta must be > 0, not {delta}")
    low, high = _bounds(ci)
    if low > delta:
        return "better_by_more_than_delta"
    if high < -delta:
        return "worse_by_more_than_delta"
    if -delta < low and high < delta:
        return "practically_equivalent"
    if low > 0:
        return "better_not_shown_by_delta"
    if high < 0:
        return "worse_not_shown_by_delta"
    return "inconclusive"


def decide(rule: str, ci: Estimate | Sequence[float], delta: float) -> bool:
    """Whether a hypothesis part holds under its rule, read from the CI and delta alone.

      ci_lower_bound_above_zero    ci_low > 0           H1a, H1b, H2a, H2b
      ci_within_plus_minus_delta   -delta < ci_low and ci_high < +delta
                                                        H2c, equivalence
      ci_lower_bound_above_delta   ci_low > +delta      H2c, superiority

    Raises:
        ValueError: For an unknown rule, a delta that is not positive, or an
            interval that is not one.
    """
    if not delta > 0:
        raise ValueError(f"delta must be > 0, not {delta}")
    low, high = _bounds(ci)
    if rule == ABOVE_ZERO:
        return low > 0
    if rule == EQUIVALENCE:
        return -delta < low and high < delta
    if rule == SUPERIORITY:
        return low > delta
    raise ValueError(f"unknown rule {rule!r}; expected one of {list(RULES)}")
