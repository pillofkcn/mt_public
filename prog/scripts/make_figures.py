"""The three figures of version 1.0, drawn from the thesis tables of a run.

    python scripts/make_figures.py --run-dir results/run_2026-10-24
    python scripts/make_figures.py --run-dir DIR --plan PATH

Reads DIR/csv/thesis_h1a.csv, thesis_h1b.csv, thesis_h2a.csv, thesis_h2b.csv
and thesis_h2c.csv, which scripts/analyze.py writes, and the plan
(plan/v1.yaml unless --plan names another). The plan says which rows a figure
has - the H1a chains and their steps, the H1b step and its sets, the H2a
pairs, the H2b transitions, the H2c tests, the sets of every task - and delta.
Nothing else is read: no counts file, no dataset, no analyzer. Every number
drawn is a cell of those tables; nothing is recomputed, interpolated or
smoothed.

Writes DIR/figures/:

  fig1_h1.png   H1a, one panel per task (pos, dep, con): every step of every
                chain, Delta = M(higher) - M(lower) with its 95 % CI, labelled
                "lower -> higher (Lx -> Ly)" and grouped by chain; the three
                panels share their x axis. Beside them H1b: Delta_LAS and
                Delta_UPOS of the step stanza_comb -> stanza_trf and the
                difference Delta_LAS - Delta_UPOS, each with its CI, per set.
                Lines at 0 (the criterion of H1a and H1b: ci_low > 0) and at
                +delta (dashed).
  fig2_h2.png   H2, a row for H2a and one for H2b. Left, the effect with its CI
                - |Delta_source| of every H2a pair on the sets of the plan's
                h2a.datasets (EWT test since 24 Oct 2026, so its rows are
                exploratory only), M_in - M_out of every H2b configuration -
                and a black mark at Delta_step,max of the same
                task and set (for H2b, of the source set), so the effect is
                seen against the largest paradigm step. Right, on the same
                rows, the statistic the criterion reads, effect -
                Delta_step,max from the same bootstrap replicate, with its CI
                and the line at 0 (the criterion: ci_low > 0). The effect's CI
                and the step's point estimate alone would not decide it: the
                step is uncertain too, and the criterion takes both from one
                replicate (analyze.py).
  fig3_h2c.png  H2c, one panel per task: Delta = M(stanza_comb) -
                M(stanza_ewt) with its CI on each set of the test, the
                equivalence band (-delta, +delta) shaded and the superiority
                threshold +delta dashed.
  fig*.pdf      the same three figures, as vector PDF.
  figures.prov.json  the plan (path, sha256), the sha256 of every table read,
                the sha256 of every file written, what each panel drew and
                what it awaits, the matplotlib version, the command and the
                time. No path of this machine.

|Delta_source| is drawn with the interval of Delta_source; where the point
estimate is negative, with the interval of -Delta_source (the interval
mirrored). The tables hold no other interval of it. The panel says so: its
subtitle names the interval (the 95 % CI of M(b) - M(a)), and a note lists the
mirrored rows, one line each, by task, pair and set. Near 0 that interval
reaches below 0, which |Delta_source| cannot; the criterion beside it reads
|replicate| (analyze.py), not this interval.

What every figure keeps to:

  * Exploratory and confirmatory results are never merged: each is its own
    interval, in its own colour and marker (blue circle, orange square), and
    the legend names the sets of each status. Where a row has both, they sit
    one above the other, exploratory first.
  * A row that is not in the tables is not drawn. Before the confirmatory run
    the tables have no row on GENTLE or MCTB_en, so H2b is empty and the
    panels of the other hypotheses draw their exploratory rows only; each
    such panel says in one line what awaits ("čaká na konfirmačný beh:
    gentle"), and an empty panel says only that. H2a, whose sets the plan
    names (EWT test only), awaits nothing. A row missing on an exploratory
    set is named the same way ("nie je v behu: wsj_01").
  * The text is Slovak, as the thesis is: titles, axis labels, legend, notes,
    and a decimal comma. Configuration keys, chain ids and dataset ids stay
    as the plan spells them.

Two layers. The data layer (figure1_data, figure2_data, figure3_data) turns
the rows of the tables into Panels - which intervals, with which numbers, on
which rows - and is pure: no matplotlib. The drawing layer (draw) draws
Panels and nothing else, and gives every line it draws for an interval a gid
"<panel>|<row>|<dataset>|point|ci|caps|reference", so a test can check the
drawn numbers against the tables without comparing images. The caps at both
ends of an interval are drawn over its point and taller than it, so an
interval narrower than the point still shows.

Refused, with exit status 2 and nothing written:
  * a thesis table that is missing, or lacks a column read here;
  * a row the plan does not have (another chain, step, pair, configuration or
    set), a row given twice, a status, level, decision, baseline, treatment,
    source, target or verdict set other than the plan's, an empty or
    non-numeric cell where a number is drawn;
  * a plan that gives one row of a figure two sets of one status, which the
    figure could not keep apart;
  * a figure whose layout collapses (matplotlib's "constrained_layout not
    applied") or whose title, note or legend does not fit in it: the titles
    and notes are left out of the layout, which reserves their height only,
    so their width is checked once the figure is drawn.

Rewritten 23 Oct 2026 for version 1.0 (step 10). Every figure of the earlier
design - scores per analyzer, level trends, the level x domain grid, effect
comparisons, agreement, speed, genre, length and error breakdowns, confusion
matrices - is gone with the tables it read; they are in the history and under
the tag exploracia-2026-10-17.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import warnings
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402  (after the backend)
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

from syntactic_analysis.evaluation.report import utf8_console  # noqa: E402
from syntactic_analysis.plan import Plan, PlanError, load  # noqa: E402
from syntactic_analysis.stage import now, portable, task_datasets  # noqa: E402

FIGURES_SUBDIR = "figures"
PROVENANCE = "figures.prov.json"
PROVENANCE_SCHEMA = 1
DPI = 300

STATUSES: tuple[str, ...] = ("exploratory", "confirmatory")

# The columns of the thesis tables this script reads (the contract of steps 6
# and 10; scripts/analyze.py writes more).
TABLE_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "thesis_h1a": ("task", "chain", "dataset", "status", "lower", "higher", "lower_level",
                   "higher_level", "delta", "ci_low", "ci_high"),
    "thesis_h1b": ("dataset", "status", "is_verdict", "baseline", "treatment", "diff",
                   "ci_low", "ci_high"),
    "thesis_h2a": ("task", "pair", "dataset", "status", "baseline", "treatment", "delta_source",
                   "delta_source_ci_low", "delta_source_ci_high", "step_max", "stat", "ci_low",
                   "ci_high"),
    "thesis_h2b": ("task", "config", "source", "target", "status", "drop", "drop_ci_low",
                   "drop_ci_high", "step_max", "stat", "ci_low", "ci_high"),
    "thesis_h2c": ("task", "dataset", "status", "decision", "baseline", "treatment", "delta",
                   "ci_low", "ci_high"),
}
TABLES: tuple[str, ...] = tuple(TABLE_COLUMNS)

# --- the Slovak words of the figures ---------------------------------------------

TASK_NAMES: Mapping[str, str] = {"pos": "slovné druhy", "dep": "závislosti", "con": "zložky"}
METRIC_NAMES: Mapping[str, str] = {
    "upos_accuracy": "UPOS", "las": "LAS", "bracket_f1": "F1 zátvoriek",
}
STATUS_NAMES: Mapping[str, str] = {"exploratory": "exploračné", "confirmatory": "konfirmačné"}
DECISION_NAMES: Mapping[str, str] = {"equivalence": "ekvivalencia", "superiority": "superiorita"}
AWAITING = "čaká na konfirmačný beh"
NOT_IN_RUN = "nie je v behu"
POINTS = "[p. b.]"     # percentuálne body
ARROW = "→"
MINUS = "−"
DELTA_STEP = r"$\Delta_{\mathrm{krok,max}}$"
DELTA_SOURCE = r"$|\Delta_{\mathrm{zdroj}}|$"
DROP = r"$M_{\mathrm{in}} - M_{\mathrm{out}}$"


class FigureError(Exception):
    """A run the figures refuse: what is wrong, and in which table."""


# --- the data layer: what is drawn ----------------------------------------------


@dataclass(frozen=True)
class Table:
    """One thesis table as read: its header, its rows, the sha256 of its bytes."""

    name: str
    header: tuple[str, ...]
    rows: tuple[Mapping[str, str], ...]
    sha256: str

    @property
    def file(self) -> str:
        return f"{self.name}.csv"


@dataclass(frozen=True)
class Interval:
    """One estimate on one set: the point and its 95 % CI, in points.

    reference: Delta_step,max of the same task and set (figure 2, the effects).
    mirrored:  the interval of -Delta_source, drawn for |Delta_source| because
               the point estimate of Delta_source is negative (figure 2, H2a).
    """

    dataset: str
    status: str
    point: float
    ci_low: float
    ci_high: float
    reference: float | None = None
    mirrored: bool = False


@dataclass(frozen=True)
class Row:
    """One labelled row of a panel: its intervals, one per set in the run, in plan order.

    A row holds at most one interval of each status: each status has its own
    place in the row (exploratory above, confirmatory below).
    """

    key: str
    label: str
    intervals: tuple[Interval, ...]

    def __post_init__(self) -> None:
        statuses = [interval.status for interval in self.intervals]
        if len(set(statuses)) != len(statuses):
            raise FigureError(
                f"row {self.key}: two sets of one status "
                f"({', '.join(i.dataset for i in self.intervals)}); a figure row keeps one "
                f"exploratory and one confirmatory interval apart, not two of one status"
            )


@dataclass(frozen=True)
class Group:
    """Rows that belong together (a chain, a task); a bold header unless the title is empty."""

    title: str
    rows: tuple[Row, ...]


@dataclass(frozen=True)
class Panel:
    """One panel: its rows and what is drawn behind them.

    subtitle: a line or two under the title: the contrast or the criterion drawn.
    notes:    one line each under it: what awaits, what is not in the run,
              which intervals are mirrored.
    lines:    reference lines, "zero" (solid) and "delta" (+delta, dashed).
    band:     shade the equivalence band (-delta, +delta).
    """

    key: str
    title: str
    xlabel: str
    groups: tuple[Group, ...]
    subtitle: str = ""
    notes: tuple[str, ...] = ()
    lines: tuple[str, ...] = ("zero",)
    band: bool = False

    @property
    def rows(self) -> tuple[Row, ...]:
        return tuple(row for group in self.groups for row in group.rows)

    @property
    def intervals(self) -> tuple[Interval, ...]:
        return tuple(interval for row in self.rows for interval in row.intervals)

    @property
    def empty(self) -> bool:
        return not self.intervals

    @property
    def statuses(self) -> tuple[str, ...]:
        """The statuses this panel draws, exploratory first."""
        drawn = {interval.status for interval in self.intervals}
        return tuple(status for status in STATUSES if status in drawn)


@dataclass(frozen=True)
class FigureData:
    """Everything one figure draws.

    statuses: status -> the sets of the figure with that status, for the legend
              (every set the plan gives the figure, drawn or awaited).
    """

    name: str
    panels: tuple[Panel, ...]
    delta: float
    statuses: Mapping[str, tuple[str, ...]]
    tables: tuple[str, ...]

    def panel(self, key: str) -> Panel:
        for panel in self.panels:
            if panel.key == key:
                return panel
        raise KeyError(key)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_tables(csv_dir: Path, names: Iterable[str] = TABLES) -> dict[str, Table]:
    """Read the thesis tables the figures draw from.

    Raises:
        FigureError: If a table is missing or lacks a column read here.
    """
    tables = {}
    for name in names:
        path = csv_dir / f"{name}.csv"
        try:
            data = path.read_bytes()
        except OSError:
            raise FigureError(
                f"{name}.csv is not in {portable(csv_dir)}: run scripts/analyze.py on the run first"
            ) from None
        reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
        header = tuple(reader.fieldnames or ())
        missing = [column for column in TABLE_COLUMNS[name] if column not in header]
        if missing:
            raise FigureError(f"{name}.csv lacks the column(s) {', '.join(missing)}")
        tables[name] = Table(name, header, tuple(reader), _sha256(data))
    return tables


class _Index:
    """The rows of one table by key; each row must be taken by exactly one place of the plan."""

    def __init__(self, table: Table, columns: tuple[str, ...]) -> None:
        self.table = table
        self.columns = columns
        self.rows: dict[tuple[str, ...], Mapping[str, str]] = {}
        for line, row in enumerate(table.rows, start=2):
            key = tuple(row[column] or "" for column in columns)
            if key in self.rows:
                raise FigureError(f"{table.file}: two rows for {self._text(key)} (line {line})")
            self.rows[key] = row
        self.taken: set[tuple[str, ...]] = set()

    def _text(self, key: tuple[str, ...]) -> str:
        return ", ".join(f"{column}={value}" for column, value in zip(self.columns, key))

    def take(self, *key: str) -> Mapping[str, str] | None:
        row = self.rows.get(key)
        if row is not None:
            self.taken.add(key)
        return row

    def done(self) -> None:
        """Refuse a row no place of the plan took."""
        left = [key for key in self.rows if key not in self.taken]
        if left:
            raise FigureError(
                f"{self.table.file}: row(s) the plan does not have: "
                + "; ".join(self._text(key) for key in left)
            )


def _number(table: Table, row: Mapping[str, str], column: str) -> float:
    cell = row.get(column) or ""
    try:
        value = float(cell)
    except ValueError:
        value = math.nan
    if not math.isfinite(value):
        raise FigureError(f"{table.file}: {column} is {cell!r} where a number is drawn")
    return value


def _expect(table: Table, row: Mapping[str, str], column: str, expected: str) -> None:
    if row[column] != expected:
        raise FigureError(
            f"{table.file}: {column} {row[column]!r} where the plan gives {expected!r} "
            f"(row {dict(row)})"
        )


def _status(plan: Plan, *datasets: str) -> str:
    """A row on a confirmatory set is confirmatory."""
    if any(plan.datasets[dataset].status == "confirmatory" for dataset in datasets):
        return "confirmatory"
    return "exploratory"


def _interval(table: Table, row: Mapping[str, str], dataset: str, status: str, point: str,
              low: str, high: str, *, reference: str | None = None) -> Interval:
    _expect(table, row, "status", status)
    return Interval(
        dataset=dataset, status=status, point=_number(table, row, point),
        ci_low=_number(table, row, low), ci_high=_number(table, row, high),
        reference=None if reference is None else _number(table, row, reference),
    )


def _plan_order(plan: Plan, datasets: Iterable[str]) -> tuple[str, ...]:
    wanted = set(datasets)
    return tuple(dataset for dataset in plan.datasets if dataset in wanted)


def _notes(plan: Plan, missing: Iterable[str]) -> tuple[str, ...]:
    """What a panel lacks, by status: one line for what awaits, one for what is not in the run."""
    missing = _plan_order(plan, missing)
    notes = []
    awaiting = [d for d in missing if plan.datasets[d].status == "confirmatory"]
    absent = [d for d in missing if plan.datasets[d].status != "confirmatory"]
    if awaiting:
        notes.append(f"{AWAITING}: {', '.join(awaiting)}")
    if absent:
        notes.append(f"{NOT_IN_RUN}: {', '.join(absent)}")
    return tuple(notes)


def _statuses(plan: Plan, datasets: Iterable[str]) -> dict[str, tuple[str, ...]]:
    ordered = _plan_order(plan, datasets)
    return {
        status: tuple(d for d in ordered if plan.datasets[d].status == status)
        for status in STATUSES
    }


def _level(plan: Plan, task: str, key: str) -> str:
    return plan.tasks[task].members[key].level.code


def _task_title(plan: Plan, task: str) -> str:
    try:
        return f"{TASK_NAMES[task]} ({METRIC_NAMES[plan.tasks[task].metric.name]})"
    except KeyError as exc:
        raise FigureError(f"no Slovak name for {exc.args[0]!r}: add it to make_figures.py") from None


def _step_label(plan: Plan, task: str, lower: str, higher: str) -> str:
    return (f"{lower} {ARROW} {higher}\n"
            f"({_level(plan, task, lower)} {ARROW} {_level(plan, task, higher)})")


def figure1_data(plan: Plan, tables: Mapping[str, Table]) -> FigureData:
    """H1a per task and H1b beside them."""
    h1a_table, h1b_table = tables["thesis_h1a"], tables["thesis_h1b"]
    h1a = _Index(h1a_table, ("task", "chain", "dataset", "lower", "higher"))
    panels = []
    figure_sets: list[str] = []
    for task, chains in plan.h1a.items():
        datasets = task_datasets(plan, task)
        figure_sets += datasets
        missing: set[str] = set()
        groups = []
        for chain in chains:
            rows = []
            for lower, higher in pairwise(chain.configs):
                intervals = []
                for dataset in datasets:
                    row = h1a.take(task, chain.id, dataset, lower, higher)
                    if row is None:
                        missing.add(dataset)
                        continue
                    _expect(h1a_table, row, "lower_level", _level(plan, task, lower))
                    _expect(h1a_table, row, "higher_level", _level(plan, task, higher))
                    intervals.append(_interval(h1a_table, row, dataset, _status(plan, dataset),
                                               "delta", "ci_low", "ci_high"))
                rows.append(Row(f"{task}/{chain.id}/{lower}->{higher}",
                                _step_label(plan, task, lower, higher), tuple(intervals)))
            groups.append(Group(f"reťaz {chain.id}", tuple(rows)))
        panels.append(Panel(
            key=f"h1a/{task}", title=f"H1a – {_task_title(plan, task)}",
            xlabel=f"Δ = M(vyššia) {MINUS} M(nižšia) {POINTS}", groups=tuple(groups),
            notes=_notes(plan, missing), lines=("zero", "delta"),
        ))
    h1a.done()

    h1b = plan.h1b
    leg_columns = [column for column in h1b_table.header
                   if column.startswith("delta_") and not column.endswith(("_ci_low", "_ci_high"))]
    if len(leg_columns) != len(h1b.legs) or len(leg_columns) != 2:
        raise FigureError(
            f"{h1b_table.file}: {len(leg_columns)} Delta column(s) ({', '.join(leg_columns)}) "
            f"for the {len(h1b.legs)} legs of the plan's H1b"
        )
    for column in leg_columns:
        for suffix in ("_ci_low", "_ci_high"):
            if column + suffix not in h1b_table.header:
                raise FigureError(f"{h1b_table.file} lacks the column {column}{suffix}")
    names = [column.removeprefix("delta_").upper() for column in leg_columns]
    symbols = [rf"$\Delta_{{\mathrm{{{name}}}}}$" for name in names]
    index = _Index(h1b_table, ("dataset",))
    legs: list[list[Interval]] = [[] for _ in leg_columns]
    diffs: list[Interval] = []
    missing = set()
    for dataset in h1b.datasets:
        row = index.take(dataset)
        if row is None:
            missing.add(dataset)
            continue
        _expect(h1b_table, row, "baseline", h1b.baseline)
        _expect(h1b_table, row, "treatment", h1b.treatment)
        _expect(h1b_table, row, "is_verdict", "yes" if dataset == h1b.verdict else "no")
        status = _status(plan, dataset)
        for leg, column in zip(legs, leg_columns):
            leg.append(_interval(h1b_table, row, dataset, status, column,
                                 f"{column}_ci_low", f"{column}_ci_high"))
        diffs.append(_interval(h1b_table, row, dataset, status, "diff", "ci_low", "ci_high"))
    index.done()
    figure_sets += h1b.datasets
    task = h1b.legs[0].task
    rows = [Row(f"h1b/{column}", symbol, tuple(leg))
            for column, symbol, leg in zip(leg_columns, symbols, legs)]
    rows.append(Row("h1b/diff", f"{symbols[0]} {MINUS} {symbols[1]}", tuple(diffs)))
    step = (f"{h1b.baseline} {ARROW} {h1b.treatment} "
            f"({_level(plan, task, h1b.baseline)} {ARROW} {_level(plan, task, h1b.treatment)})")
    panels.append(Panel(
        key="h1b", title="H1b – krok k transformátoru", xlabel=f"Δ {POINTS}",
        groups=(Group("", tuple(rows)),), subtitle=f"{step}\nverdikt dáva {h1b.verdict}",
        notes=_notes(plan, missing), lines=("zero", "delta"),
    ))
    return FigureData("fig1_h1", tuple(panels), plan.statistics.delta,
                      _statuses(plan, figure_sets), ("thesis_h1a", "thesis_h1b"))


def figure2_data(plan: Plan, tables: Mapping[str, Table]) -> FigureData:
    """H2a and H2b: the effect against Delta_step,max, and the statistic of the criterion."""
    h2a_table, h2b_table = tables["thesis_h2a"], tables["thesis_h2b"]
    h2a = _Index(h2a_table, ("task", "pair", "dataset"))
    effects, stats, mirrored = [], [], []
    missing: set[str] = set()
    figure_sets: list[str] = []
    for task, pairs in plan.h2a.items():
        # The plan's h2a.datasets only (EWT test since 24 Oct 2026): H2a has
        # no row on another set and awaits none.
        datasets = plan.h2a_datasets_of(task)
        figure_sets += datasets
        effect_rows, stat_rows = [], []
        for pair in pairs:
            effect, stat = [], []
            for dataset in datasets:
                row = h2a.take(task, pair.id, dataset)
                if row is None:
                    missing.add(dataset)
                    continue
                _expect(h2a_table, row, "baseline", pair.baseline)
                _expect(h2a_table, row, "treatment", pair.treatment)
                status = _status(plan, dataset)
                source = _interval(h2a_table, row, dataset, status, "delta_source",
                                   "delta_source_ci_low", "delta_source_ci_high",
                                   reference="step_max")
                if source.point < 0:
                    source = Interval(dataset, status, -source.point, -source.ci_high,
                                      -source.ci_low, source.reference, mirrored=True)
                    mirrored.append(
                        f"{TASK_NAMES[task]}: {pair.baseline} {ARROW} {pair.treatment} ({dataset})"
                    )
                effect.append(source)
                stat.append(_interval(h2a_table, row, dataset, status, "stat", "ci_low", "ci_high"))
            key = f"{task}/{pair.id}"
            label = f"{pair.baseline} {ARROW} {pair.treatment}"
            effect_rows.append(Row(key, label, tuple(effect)))
            stat_rows.append(Row(key, label, tuple(stat)))
        title = _task_title(plan, task)
        effects.append(Group(title, tuple(effect_rows)))
        stats.append(Group(title, tuple(stat_rows)))
    h2a.done()
    notes = _notes(plan, missing)
    if mirrored:
        # One line per mirrored row: joined into one, three rows outgrew the column.
        notes += (f"M(b) < M(a), interval zrkadlený (IS M(a) {MINUS} M(b)):", *mirrored)
    panels = [
        Panel("h2a/effect", "H2a – tréningové dáta", f"{DELTA_SOURCE} a {DELTA_STEP} {POINTS}",
              tuple(effects),
              subtitle=(f"riadok a {ARROW} b: bod {DELTA_SOURCE} = |M(b) {MINUS} M(a)|\n"
                        f"interval: 95 % IS M(b) {MINUS} M(a)"),
              notes=notes, lines=()),
        Panel("h2a/criterion", "kritérium H2a", f"{DELTA_SOURCE} {MINUS} {DELTA_STEP}\n{POINTS}",
              tuple(stats), subtitle="platí, keď dolná hranica IS > 0"),
    ]

    h2b = _Index(h2b_table, ("task", "config"))
    effects, stats = [], []
    missing = set()
    for task, transition in plan.h2b.items():
        status = _status(plan, transition.source, transition.target)
        figure_sets.append(transition.target)
        effect_rows, stat_rows = [], []
        for config in transition.configs:
            effect, stat = [], []
            row = h2b.take(task, config)
            if row is None:
                # What a missing row awaits: its confirmatory set(s), else both sets.
                needed = (transition.source, transition.target)
                missing.update([d for d in needed if plan.datasets[d].status == "confirmatory"]
                               or needed)
            else:
                _expect(h2b_table, row, "source", transition.source)
                _expect(h2b_table, row, "target", transition.target)
                effect.append(_interval(h2b_table, row, transition.target, status, "drop",
                                        "drop_ci_low", "drop_ci_high", reference="step_max"))
                stat.append(_interval(h2b_table, row, transition.target, status, "stat",
                                      "ci_low", "ci_high"))
            key = f"{task}/{config}"
            label = f"{config} ({_level(plan, task, config)})"
            effect_rows.append(Row(key, label, tuple(effect)))
            stat_rows.append(Row(key, label, tuple(stat)))
        title = f"{_task_title(plan, task)}: {transition.source} {ARROW} {transition.target}"
        effects.append(Group(title, tuple(effect_rows)))
        stats.append(Group(title, tuple(stat_rows)))
    h2b.done()
    notes = _notes(plan, missing)
    panels += [
        Panel("h2b/effect", "H2b – posun domény", f"{DROP} a {DELTA_STEP} {POINTS}",
              tuple(effects), subtitle=f"{DELTA_STEP} na zdrojovej množine", notes=notes,
              lines=()),
        Panel("h2b/criterion", "kritérium H2b", f"({DROP}) {MINUS} {DELTA_STEP}\n{POINTS}",
              tuple(stats), subtitle="platí, keď dolná hranica IS > 0"),
    ]
    return FigureData("fig2_h2", tuple(panels), plan.statistics.delta,
                      _statuses(plan, figure_sets), ("thesis_h2a", "thesis_h2b"))


def figure3_data(plan: Plan, tables: Mapping[str, Table]) -> FigureData:
    """H2c per task: the related-data contrast on each set of its tests."""
    table = tables["thesis_h2c"]
    index = _Index(table, ("task", "dataset"))
    panels = []
    figure_sets: list[str] = []
    for task, contrast in plan.h2c.items():
        rows, missing = [], set()
        for test in contrast.tests:
            figure_sets.append(test.dataset)
            try:
                decision = DECISION_NAMES[test.decision]
            except KeyError:
                raise FigureError(f"no Slovak name for the decision {test.decision!r}") from None
            intervals = []
            row = index.take(task, test.dataset)
            if row is None:
                missing.add(test.dataset)
            else:
                _expect(table, row, "decision", test.decision)
                _expect(table, row, "baseline", contrast.baseline)
                _expect(table, row, "treatment", contrast.treatment)
                intervals.append(_interval(table, row, test.dataset, _status(plan, test.dataset),
                                           "delta", "ci_low", "ci_high"))
            rows.append(Row(f"{task}/{test.dataset}/{test.decision}",
                            f"{test.dataset}\n({decision})", tuple(intervals)))
        panels.append(Panel(
            key=f"h2c/{task}", title=f"H2c – {_task_title(plan, task)}", xlabel=f"Δ {POINTS}",
            groups=(Group("", tuple(rows)),),
            subtitle=f"Δ = M({contrast.treatment}) {MINUS} M({contrast.baseline})",
            notes=_notes(plan, missing), lines=("zero", "delta"), band=True,
        ))
    index.done()
    return FigureData("fig3_h2c", tuple(panels), plan.statistics.delta,
                      _statuses(plan, figure_sets), ("thesis_h2c",))


FIGURES: tuple[Callable[[Plan, Mapping[str, Table]], FigureData], ...] = (
    figure1_data, figure2_data, figure3_data,
)


def build_figures(plan: Plan, tables: Mapping[str, Table]) -> tuple[FigureData, ...]:
    """The data of the three figures. Raises FigureError (see the module docstring)."""
    return tuple(build(plan, tables) for build in FIGURES)


# --- the drawing layer ----------------------------------------------------------

WIDTH = 6.3                 # inches: 16 cm, the text width of the thesis page
INK = "#1f1f1d"
INK_SECONDARY = "#52514e"
INK_MUTED = "#6f6e69"
GRID = "#e4e3df"
BAND = "#e8e7e3"
STATUS_STYLE: Mapping[str, tuple[str, str]] = {   # colour, marker
    "exploratory": ("#2a78d6", "o"),
    "confirmatory": ("#eb6834", "s"),
}
ROW_ONE, ROW_TWO, HEADER = 0.24, 0.36, 0.22      # inches per row with 1 or 2 statuses, per header
SLOT = 0.19                 # rows apart: the two statuses of one row
POINT_SIZE = 5              # points: the marker of a point estimate
CAP_SIZE = 8                # points: the caps at the ends of its interval, taller than the point
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": INK_SECONDARY,
    "axes.labelcolor": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "pdf.fonttype": 42,
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": True,
}


def number(value: float, digits: int | None = None) -> str:
    """A number as the thesis prints it: decimal comma, a true minus sign."""
    if abs(value) < 1e-12:
        value = 0.0
    text = f"{value:g}" if digits is None else f"{value:.{digits}f}"
    return text.replace("-", MINUS).replace(".", ",")


_TICKS = FuncFormatter(lambda value, _: number(round(value, 10)))


def _slots(statuses: Sequence[str]) -> dict[str, float]:
    """Vertical offset of each status within a row: one centred, two one above the other."""
    if len(statuses) <= 1:
        return {status: 0.0 for status in statuses}
    step = 2 * SLOT / (len(statuses) - 1)
    return {status: -SLOT + i * step for i, status in enumerate(statuses)}


def _layout(panel: Panel) -> tuple[list[tuple[float, str, bool]], dict[str, float]]:
    """Ticks (y, text, is a header) top to bottom, and the y of every row."""
    ticks, ys, y = [], {}, 0.0
    for group in panel.groups:
        if group.title:
            ticks.append((y, group.title, True))
            y += 0.75
        for row in group.rows:
            ticks.append((y, row.label, False))
            ys[row.key] = y
            y += 1.0
    return ticks, ys


def panel_height(panel: Panel, statuses: int) -> float:
    """Inches of rows a panel needs (its title and axis come on top)."""
    if panel.empty:
        return 0.05
    headers = sum(1 for group in panel.groups if group.title)
    lines = max(len(row.label.splitlines()) for row in panel.rows)
    row = max(ROW_TWO if statuses > 1 else ROW_ONE, 0.12 * lines + 0.06)
    return headers * HEADER + len(panel.rows) * row


def decoration(lines: int, xlabel_lines: int, *, axis: bool = True) -> float:
    """Inches a panel needs around its rows: title, lines under it, x ticks and label."""
    under = 0.27 + 0.15 * xlabel_lines if axis else 0.0
    return 0.24 + LINE / 72 * lines + under + 0.08


def _xlimits(panels: Iterable[Panel], delta: float) -> tuple[float, float]:
    values: list[float] = []
    for panel in panels:
        for interval in panel.intervals:
            values += [interval.ci_low, interval.ci_high, interval.point]
            if interval.reference is not None:
                values.append(interval.reference)
        if "zero" in panel.lines:
            values.append(0.0)
        if "delta" in panel.lines or panel.band:
            values.append(delta)
        if panel.band:
            values.append(-delta)
    low, high = min(values), max(values)
    pad = 0.07 * (high - low or 1.0)
    return low - pad, high + pad


def _gid(panel: Panel, row: Row, interval: Interval, part: str) -> str:
    return f"{panel.key}|{row.key}|{interval.dataset}|{part}"


LINE = 9.5                  # points from one line under a title to the next
TICK_PAD = 4                # points between a row label and the axes


def _labels_width(ax) -> float:
    """Points from the left edge of the widest row label to the axes."""
    if not ax.axison:
        return 0.0
    renderer = ax.figure.canvas.get_renderer()
    widths = [label.get_window_extent(renderer).width for label in ax.get_yticklabels()
              if label.get_text()]
    return (max(widths) * 72 / ax.figure.dpi + TICK_PAD) if widths else 0.0


def title_lines(panel: Panel) -> int:
    """The lines a panel writes under its title: the subtitle's and one per note."""
    return len(panel.subtitle.splitlines()) + len(panel.notes)


def _title(ax, panel: Panel, indent: float = 0.0, height: int = 0) -> None:
    """The title, then the subtitle and the notes, one line each, above the axes.

    They start `indent` points left of the axes - at the row labels, so that a
    title is as wide as its column - and take `height` lines at least, so that
    panels side by side have their titles on one line. The empty title keeps
    their height in the layout; the lines themselves are out of it. In it, a
    line wider than its column widened the column's margin until the axes
    collapsed (three mirrored H2a rows on one line did); out of it, their width
    is checked once the figure is drawn (_check_fit).
    """
    lines = [(line, "normal") for line in panel.subtitle.splitlines()]
    lines += [(note, "italic") for note in panel.notes]
    height = max(height, len(lines))
    ax.set_title(" ", loc="left", fontweight="bold", pad=4 + LINE * height)
    ax.annotate(panel.title, xy=(0, 1), xycoords="axes fraction",
                xytext=(-indent, 4 + LINE * height), textcoords="offset points", ha="left",
                va="bottom", fontsize=8.5, fontweight="bold", color=INK).set_in_layout(False)
    for i, (text, style) in enumerate(lines):
        ax.annotate(text, xy=(0, 1), xycoords="axes fraction",
                    xytext=(-indent, 3 + LINE * (height - 1 - i)), textcoords="offset points",
                    ha="left", va="bottom", fontsize=7, color=INK_MUTED,
                    style=style).set_in_layout(False)


def _titles(column: Sequence[tuple[object, Panel, int]]) -> None:
    """The titles of the panels of one column (ax, panel, height), all at its widest labels.

    constrained_layout gives the axes of a column one left edge, so one indent
    puts every title of the column at the left edge of its row labels.
    """
    indent = max((_labels_width(ax) for ax, _, _ in column), default=0.0)
    for ax, panel, height in column:
        _title(ax, panel, indent, height)


def draw_empty(ax, panel: Panel) -> None:
    """A panel with nothing in the run: no axes; its title and its notes say why (_titles)."""
    ax.set_axis_off()


def draw_panel(ax, panel: Panel, delta: float, *, slots: Mapping[str, float],
               labels: bool = True, xlim: tuple[float, float] | None = None,
               nbins: int = 6) -> None:
    """Draw one panel of intervals on `ax`; its title comes from _titles.

    slots:  the vertical offset of each status within a row (_slots).
    labels: write the row labels (the right panel of a pair has none).
    xlim:   the x range, when panels share one; else the panel's own.
    nbins:  at most so many intervals between x ticks.
    """
    ticks, ys = _layout(panel)
    ax.set_yticks([y for y, _, _ in ticks])
    ax.set_yticklabels([text for _, text, _ in ticks] if labels else [])
    if labels:
        for label, (_, _, header) in zip(ax.get_yticklabels(), ticks):
            if header:
                label.set_fontweight("bold")
                label.set_color(INK)
    ax.tick_params(axis="y", length=0, pad=4)
    top, bottom = ticks[0][0], ticks[-1][0]
    ax.set_ylim(bottom + 0.6, top - 0.6)
    for y, _, header in ticks:
        if header and y > top:
            ax.axhline(y - 0.45, color=GRID, lw=0.6, zorder=0)
    if panel.band:
        ax.axvspan(-delta, delta, color=BAND, lw=0, zorder=0)
    if "zero" in panel.lines:
        ax.axvline(0, color=INK_SECONDARY, lw=0.8, zorder=1)
    if "delta" in panel.lines:
        ax.axvline(delta, color=INK_SECONDARY, lw=0.9, ls=(0, (4, 2.5)), zorder=1)
    for row in panel.rows:
        for interval in row.intervals:
            colour, marker = STATUS_STYLE[interval.status]
            y = ys[row.key] + slots[interval.status]
            if interval.reference is not None:
                ax.plot([interval.reference], [y], marker="|", markersize=11, mew=1.6,
                        color=INK, ls="none", zorder=2,
                        gid=_gid(panel, row, interval, "reference"))
            ax.plot([interval.ci_low, interval.ci_high], [y, y], color=colour, lw=1.5,
                    solid_capstyle="butt", zorder=3, gid=_gid(panel, row, interval, "ci"))
            ax.plot([interval.point], [y], ls="none", marker=marker, markersize=POINT_SIZE,
                    mfc=colour, mec="white", mew=0.7, zorder=4,
                    gid=_gid(panel, row, interval, "point"))
            # Over the point and taller than it: an interval narrower than the
            # point still shows its two ends above and below it.
            ax.plot([interval.ci_low, interval.ci_high], [y, y], ls="none", marker="|",
                    markersize=CAP_SIZE, mew=1.2, color=colour, zorder=5,
                    gid=_gid(panel, row, interval, "caps"))
    ax.set_xlim(*(xlim or _xlimits([panel], delta)))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(_TICKS)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.set_xlabel(panel.xlabel)


def _legend_items(data: FigureData, *, reference: bool = False, zero: str | None = None,
                  delta: str | None = None, band: str | None = None,
                  wrap: bool = False) -> tuple[list, list[str]]:
    handles, labels = [], []
    for status in STATUSES:
        colour, marker = STATUS_STYLE[status]
        sets = data.statuses.get(status) or ()
        handles.append(Line2D([], [], color=colour, lw=1.5, marker=marker, markersize=POINT_SIZE,
                              mfc=colour, mec="white", mew=0.7))
        where = f"{chr(10) if wrap else ' '}({', '.join(sets)})" if sets else ""
        labels.append(f"{STATUS_NAMES[status]}{where}")
    if zero:
        handles.append(Line2D([], [], color=INK_SECONDARY, lw=0.8))
        labels.append(zero)
    if delta:
        handles.append(Line2D([], [], color=INK_SECONDARY, lw=0.9, ls=(0, (4, 2.5))))
        labels.append(delta)
    if band:
        handles.append(Patch(facecolor=BAND, edgecolor="none"))
        labels.append(band)
    if reference:
        handles.append(Line2D([], [], color=INK, ls="none", marker="|", markersize=11, mew=1.6))
        labels.append(f"{DELTA_STEP} (najväčší krok H1a úlohy; pri H2b na zdrojovej množine)")
    return handles, labels


def _delta_text(delta: float) -> str:
    return f"+δ = {number(delta)} p. b."


def _new_figure(height: float) -> Figure:
    figure = Figure(figsize=(WIDTH, height), facecolor="white", layout="constrained")
    FigureCanvasAgg(figure)     # a renderer, to measure the row labels
    figure.get_layout_engine().set(h_pad=0.06, w_pad=0.06, hspace=0.04, wspace=0.03)
    return figure


def draw_figure1(data: FigureData) -> Figure:
    """H1a (three panels, one x axis) on the left, H1b and the legend on the right."""
    h1a = [panel for panel in data.panels if panel.key.startswith("h1a/")]
    h1b = data.panel("h1b")
    sets = [len(panel.statuses) for panel in h1a]
    heights = [panel_height(panel, n) for panel, n in zip(h1a, sets)]
    top = max(title_lines(h1a[0]), title_lines(h1b))   # the first row: one title line
    decorations = [
        decoration(top if i == 0 else title_lines(panel), 1 if i == len(h1a) - 1 else 0,
                   axis=not panel.empty)
        for i, panel in enumerate(h1a)
    ]
    figure = _new_figure(sum(heights) + sum(decorations) + 0.1)
    grid = figure.add_gridspec(len(h1a), 2, width_ratios=(1.55, 1), height_ratios=heights)
    xlim = _xlimits(h1a, data.delta)
    first, column = None, []
    for i, (panel, n) in enumerate(zip(h1a, sets)):
        ax = figure.add_subplot(grid[i, 0], sharex=first)
        first = first or ax
        column.append((ax, panel, top if i == 0 else 0))
        if panel.empty:
            draw_empty(ax, panel)
            continue
        draw_panel(ax, panel, data.delta, slots=_slots(panel.statuses), xlim=xlim, nbins=8)
        if i < len(h1a) - 1:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=True)
    _titles(column)
    ax = figure.add_subplot(grid[0, 1])
    if h1b.empty:
        draw_empty(ax, h1b)
    else:
        draw_panel(ax, h1b, data.delta, slots=_slots(h1b.statuses), nbins=5)
    _titles([(ax, h1b, top)])
    legend = figure.add_subplot(grid[1:, 1])
    legend.set_axis_off()
    handles, labels = _legend_items(data, zero="0: kritérium H1a, H1b", delta=_delta_text(data.delta),
                                    wrap=True)
    legend.legend(handles, labels, loc="upper left", frameon=False, borderaxespad=0,
                  handlelength=2.2, labelspacing=0.8)
    return figure


def draw_figure2(data: FigureData) -> Figure:
    """H2a and H2b, each a row: the effects on the left, the criterion on the right."""
    rows = [(data.panel("h2a/effect"), data.panel("h2a/criterion")),
            (data.panel("h2b/effect"), data.panel("h2b/criterion"))]
    heights = [panel_height(effect, len(effect.statuses)) for effect, _ in rows]
    extra = sum(
        decoration(max(title_lines(effect), title_lines(criterion)), 2, axis=not effect.empty)
        for effect, criterion in rows
    )
    figure = _new_figure(sum(heights) + extra + 0.5)
    grid = figure.add_gridspec(len(rows), 2, width_ratios=(1.45, 1), height_ratios=heights)
    lefts, rights = [], []
    for i, (effect, criterion) in enumerate(rows):
        height = max(title_lines(effect), title_lines(criterion))
        if effect.empty:
            ax = figure.add_subplot(grid[i, :])
            draw_empty(ax, effect)
            lefts.append((ax, effect, 0))
            continue
        slots = _slots(effect.statuses)
        left = figure.add_subplot(grid[i, 0])
        draw_panel(left, effect, data.delta, slots=slots, nbins=6)
        right = figure.add_subplot(grid[i, 1])
        draw_panel(right, criterion, data.delta, slots=slots, labels=False, nbins=5)
        lefts.append((left, effect, height))
        rights.append((right, criterion, height))
    _titles(lefts)
    _titles(rights)
    handles, labels = _legend_items(data, reference=True, zero="0: kritérium H2 (dolná hranica IS > 0)")
    figure.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False,
                  handlelength=2.2, columnspacing=1.6)
    return figure


def draw_figure3(data: FigureData) -> Figure:
    """H2c, one panel per task, side by side."""
    panels = data.panels
    height = max(title_lines(panel) for panel in panels)
    figure = _new_figure(max(panel_height(panel, len(panel.statuses)) for panel in panels)
                         + decoration(height, 1) + 0.65)
    grid = figure.add_gridspec(1, len(panels))
    for i, panel in enumerate(panels):
        ax = figure.add_subplot(grid[0, i])
        if panel.empty:
            draw_empty(ax, panel)
        else:
            draw_panel(ax, panel, data.delta, slots=_slots(panel.statuses), nbins=6)
        _titles([(ax, panel, height)])
    handles, labels = _legend_items(
        data, zero="Δ = 0", delta=f"{_delta_text(data.delta)}: prah superiority",
        band=f"({MINUS}δ, +δ): pásmo ekvivalencie",
    )
    figure.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False,
                  handlelength=2.2, columnspacing=1.6)
    return figure


DRAW: Mapping[str, Callable[[FigureData], Figure]] = {
    "fig1_h1": draw_figure1, "fig2_h2": draw_figure2, "fig3_h2c": draw_figure3,
}


def _check_fit(figure: Figure, name: str) -> None:
    """Refuse a drawn figure whose titles, notes or legend reach past its edges.

    The titles and notes are out of the layout (_title), which keeps their
    height and not their width; the legend is in it, and is checked as well.
    """
    renderer = figure.canvas.get_renderer()
    box = figure.bbox
    texts = [text for ax in figure.axes for text in ax.texts]
    texts += [text for legend in figure.legends for text in legend.get_texts()]
    texts += [text for ax in figure.axes if ax.get_legend() is not None
              for text in ax.get_legend().get_texts()]
    for text in texts:
        if not text.get_visible() or not text.get_text().strip():
            continue
        extent = text.get_window_extent(renderer)
        if (extent.x0 < box.x0 - 0.5 or extent.x1 > box.x1 + 0.5
                or extent.y0 < box.y0 - 0.5 or extent.y1 > box.y1 + 0.5):
            raise FigureError(
                f"{name}: the text {text.get_text()!r} does not fit in the figure "
                f"({extent.x0:.0f}..{extent.x1:.0f} x {extent.y0:.0f}..{extent.y1:.0f} px of "
                f"{box.width:.0f} x {box.height:.0f})"
            )


def draw(data: FigureData) -> Figure:
    """The matplotlib Figure of one figure's data.

    Raises:
        FigureError: If the layout collapses or a text does not fit (_check_fit).
    """
    with matplotlib.rc_context(STYLE), warnings.catch_warnings():
        # matplotlib warns and draws on when the layout collapses; such a
        # figure has its panels squeezed to slivers and its text cut off.
        warnings.filterwarnings("error", message="constrained_layout not applied")
        try:
            figure = DRAW[data.name](data)
            figure.canvas.draw()        # lay out under the style, before saving
        except UserWarning as exc:
            raise FigureError(f"{data.name}: the layout collapsed: {exc}") from None
        _check_fit(figure, data.name)
    return figure


# --- writing ----------------------------------------------------------------------


def save(figure: Figure, out_dir: Path, name: str) -> list[Path]:
    """<name>.png at DPI and <name>.pdf (vector), with no time stamp in either."""
    png, pdf = out_dir / f"{name}.png", out_dir / f"{name}.pdf"
    with matplotlib.rc_context(STYLE):
        figure.savefig(png, dpi=DPI, facecolor="white", metadata={"Software": None})
        figure.savefig(pdf, facecolor="white", metadata={"CreationDate": None, "Creator": None,
                                                         "Producer": None})
    return [png, pdf]


def _command(run_dir: Path, plan_path: Path | None) -> str:
    parts = ["python", portable(Path(__file__)), "--run-dir", portable(run_dir)]
    if plan_path is not None:
        parts += ["--plan", portable(plan_path)]
    return " ".join(parts)


def provenance(plan: Plan, run_dir: Path, tables: Mapping[str, Table],
               figures: Sequence[FigureData], written: Mapping[str, Sequence[Path]],
               command: str) -> dict:
    """figures.prov.json: what was read, what was drawn and written, by what."""
    return {
        "schema": PROVENANCE_SCHEMA,
        "run_dir": portable(run_dir),
        "plan": {"path": portable(plan.path), "sha256": plan.sha256},
        "tables": {f"csv/{table.file}": table.sha256 for table in tables.values()},
        "figures": {
            data.name: {
                "tables": [f"csv/{name}.csv" for name in data.tables],
                "files": {path.name: _sha256(path.read_bytes()) for path in written[data.name]},
                "panels": {
                    panel.key: {"intervals": len(panel.intervals), "notes": list(panel.notes)}
                    for panel in data.panels
                },
            }
            for data in figures
        },
        "delta": figures[0].delta if figures else None,
        "dpi": DPI,
        "matplotlib": matplotlib.__version__,
        "command": command,
        "created": now(),
    }


def make_figures(run_dir: Path, plan: Plan, *, plan_path: Path | None = None) -> list[Path]:
    """Read the tables of a run, draw the three figures, write them and their provenance.

    Raises:
        FigureError: If the tables are refused; nothing is written then.
    """
    tables = read_tables(run_dir / "csv")
    figures = build_figures(plan, tables)
    drawn = [(data, draw(data)) for data in figures]     # all of them before any is written
    out_dir = run_dir / FIGURES_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[Path]] = {}
    for data, figure in drawn:
        written[data.name] = save(figure, out_dir, data.name)
    record = provenance(plan, run_dir, tables, figures, written, _command(run_dir, plan_path))
    path = out_dir / PROVENANCE
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                    newline="\n")
    return [p for paths in written.values() for p in paths] + [path]


def main(argv: Sequence[str] | None = None) -> int:
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="the run: reads RUN_DIR/csv/thesis_h*.csv, writes RUN_DIR/figures/")
    parser.add_argument("--plan", type=Path, default=None,
                        help="the plan the run was analysed under; default plan/v1.yaml")
    args = parser.parse_args(argv)
    try:
        plan = load(args.plan, strict=True)
    except (PlanError, OSError) as exc:
        print(f"make_figures: no plan to draw under: {exc}", file=sys.stderr)
        return 2
    try:
        written = make_figures(args.run_dir, plan, plan_path=args.plan)
    except FigureError as exc:
        print(f"make_figures: refused, nothing written: {exc}", file=sys.stderr)
        return 2
    out_dir = args.run_dir / FIGURES_SUBDIR
    record = json.loads((out_dir / PROVENANCE).read_text(encoding="utf-8"))
    print(f"figures of {portable(args.run_dir)} (plan sha256 {plan.sha256[:12]}):")
    for name, figure in record["figures"].items():
        print(f"  {name}: {', '.join(figure['files'])}")
        for key, panel in figure["panels"].items():
            notes = f" - {'; '.join(panel['notes'])}" if panel["notes"] else ""
            print(f"    {key:<14} {panel['intervals']:>3} interval(s){notes}")
    others = sorted(p.name for p in out_dir.iterdir() if p not in set(written))
    if others:
        print(f"  NOTE: {portable(out_dir)} also holds files this script did not write: "
              f"{', '.join(others)}")
    print(f"written: {len(written)} files in {portable(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
