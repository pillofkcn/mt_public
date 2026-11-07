"""Version 1.0 analysis: the hypotheses, D2-D4 and the thesis tables, from the counts of a run.

    python scripts/analyze.py --run-dir results/run_2026-10-24
    python scripts/analyze.py --run-dir DIR --plan PATH

Reads DIR/csv/counts_<task>_<dataset>.csv - the per-sentence counts the
measurement stages write, one row per configuration and sentence - with the
provenance beside each (counts_<task>_<dataset>.prov.json), and the plan
(plan/v1.yaml unless --plan names another). Nothing else: no analyzer is
loaded and no dataset is opened. Which configurations a contrast compares, on
which sets, by which metric, and delta, the replicates, the confidence level
and the seed all come from the plan (brief rule 7).

Writes DIR/csv/thesis_*.csv, the tables the thesis text and its figures read
(step 10), their provenance DIR/csv/thesis.prov.json (the plan analysed, the
plan each counts file was measured under, and the sha256 of every counts file
read and every table written; since 24 Oct 2026), and DIR/analysis.txt, the
report, which it also prints.

Counts are analysed under the plan they were measured under (the plan sha256
of their provenance). Since 24 Oct 2026, before the lock, also under a later
plan, but only where it is PROVED that the difference does not touch
measurement: the earlier plan is read from git at the commit the provenance
records (`git show <commit>:./<recorded path>` in the project root), must have
the sha256 the provenance records, must load under this code, and must equal
the plan analysed in every part that determines the counts
(measurement_parts: training_data, datasets, configurations, each task's
metric and members with their levels, the D2 task, configurations, datasets
and sample, and the bootstrap seed, which the stages seed with). A change to
the statistics other than the seed, the hypotheses or a comment passes;
anything else is refused as before. The report's banner and thesis.prov.json say which plan measured
and which analysed.

The one statistical procedure (brief part D, evaluation/estimate.py): Delta in
points between two corpus metrics recomputed from summed counts, with the 95 %
percentile CI of a paired cluster bootstrap - the units of a set (the plan's
resampling unit: document for EWT test and GENTLE, file for WSJ 01, sentence
for MCTB_en) drawn with replacement, B times, by one matrix of indices that
every configuration scored on the set shares - and the CI read against delta.

  H1a  every step of every H1a chain, on every set of its task:
       Delta = M(higher) - M(lower); a step holds when ci_low > 0.
  H1b  baseline -> treatment of plan h1b, Delta of the first leg (LAS) minus
       Delta of the second (UPOS), both from the same replicate of the same
       documents; holds when ci_low > 0. On every set the plan lists for it;
       the plan's h1b.verdict set gives the verdict.
  H2a  every pair, on each set of the plan's h2a.datasets that lists its task
       (EWT test only since 24 Oct 2026, so H2a is exploratory only; until
       then every set of its task): |Delta_source| - Delta_step,max from the
       same replicate, Delta_source = M(treatment) - M(baseline); holds when
       ci_low > 0.
  H2b  every configuration of a transition: (M_in - M_out) - Delta_step,max,
       the two sets resampled independently and Delta_step,max taken from the
       in-domain replicate; holds when ci_low > 0.
  H2c  M(treatment) - M(baseline) on the set of each test: equivalence holds
       when -delta < ci_low and ci_high < +delta, superiority when
       ci_low > +delta (the plan's statistics.decisions).

Delta_step,max is the largest POINT estimate of an H1a step (two consecutive
members of a chain) of the same task on the same set. It is chosen once, on
the point estimates, and the replicate Deltas of that step enter the replicate
statistic. H2a takes the step on the pair's own set, H2b on the source set.

A hypothesis with several parts holds only if every part holds (brief D.7),
decided per status: the exploratory sets (EWT test, WSJ 01) and the
confirmatory ones (GENTLE, MCTB_en) give separate verdicts, in separate rows
and report sections, and a part that uses a confirmatory set is confirmatory.
thesis_verdicts.csv has a row for each of h1a ... h2c per status it has parts
of, and for the whole hypotheses H1 (h1a and h1b) and H2 (h2a, h2b and h2c)
per status. A whole hypothesis is decided, for a status, from the parts the
plan defines for that status, once all of them are computed: a component
with no part of that status - H2a has no confirmatory part, H2b no
exploratory one - does not enter it, and the row's note says so (the rule of
24 Oct 2026, stated in the plan; until then such a whole hypothesis was left
undecided). Every hypothesis table has the class of its row (classify(CI,
delta), brief D.6) beside the verdict. Every class and verdict follows from
the CI and delta of its own row; the intervals are the whole of the
inference.

Descriptive, with a CI where a difference is reported and no class or
verdict: D2 (PCFG coverage by failure reason, and the F1 of the PCFG and of the
neural parsers on exactly the sentences the PCFG parsed), D3 (spacy_sm against
the UD parsers, LAS and CLAS), D4 (nltk_ewt -> spacy_ewt, UPOS), and the drop
from the source to the target set of every configuration of an H2b task whose
only role there is `descriptive` and which the plan scores on both sets
(supar_con, WSJ 01 -> MCTB_en; the POS members udpipe_ewt and rdr_ewt, EWT
test -> GENTLE). D3 and D4 are computed on the sets of their task where the
plan scores every one of their members (Plan.d3_datasets, Plan.d4_datasets):
since 24 Oct 2026, when the plan restricted nltk_pre and spacy_sm to EWT test
(configurations.<key>.datasets), D3 on EWT test only, D4 on EWT test and
GENTLE. A configuration the plan does not score on a set has no row in its
counts file, so thesis_scores.csv has no row of it there, and none is
required of it. Sensitivity: every
hypothesis row at the delta_sensitivity values (0.5 and 2.0), and with the
plan's unit_sensitivity (the sentence) as the unit. And, where the plan names
statistics.subset_sensitivity evalb_both_parsers (24 Oct 2026; subset_rows),
descriptively, with no class and no verdict: every Delta between two
constituency parsers on one set - each H1a step of con on each set, and the
Delta_step,max term of the H2b con rows on their source set - also on the
sentences EVALB scores for both parsers, with their number, by the same unit
and index matrix (a filter on the counts, not a new draw); its rows follow
the others in thesis_sensitivity.csv as <row>/evalb_both_parsers=<sentences>.

Refused, with exit status 2 and nothing written:
  * a counts file without provenance, or whose provenance names a task,
    dataset or status other than the file and the plan, or another plan that
    is not proved to measure as the plan read here does (above);
  * a configuration a contrast compares that is missing from a counts file
    that exists, and rows of a configuration the plan does not score on the
    set (configurations.<key>.datasets);
  * POS and dependency counts of one set that differ in sentences or units:
    H1b takes both Deltas from the same replicate of the same documents;
  * a counts file that breaks the contract of step 5: its columns, its order,
    a sentence described two ways, a configuration without a row for every
    sentence (the D2 PCFG configurations: for every sentence of the D2
    sample), counts no sentence can have;
  * a full run whose number of units differs from the plan's expected_units.
A smoke run (its provenance records max_sentences) is analysed with a
warning, and its units are not checked against expected_units. A set of one
unit (the first 20 sentences of WSJ 01 are one file) gets a second sentence in
the warning: every replicate is then the whole set, and every interval of the
set its point estimate.

A part whose set is not in the run - a confirmatory set before the
confirmatory run - is not written to the thesis tables; thesis_verdicts.csv
and the report say what awaits.

Replaces scripts/hypothesis_tests.py, which step 12 of version 1.0 removed.
Added 23 Oct 2026 (version 1.0, step 6).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import numbers
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator

import numpy as np

from syntactic_analysis.evaluation.estimate import (
    ABOVE_ZERO,
    ClusterSums,
    Estimate,
    classify,
    cluster_bootstrap,
    decide,
    delta_ci,
    diff_of_deltas_ci,
    metric,
    seed_sequence,
    two_set_drop_ci,
)
from syntactic_analysis.evaluation.provenance import recorded_path
from syntactic_analysis.plan import ROLES, TASKS, Plan, PlanError, load, plan_sha256
from syntactic_analysis.project import project_root
from syntactic_analysis.stage import portable

# --- the counts contract (step 5 writes it, this reads it) -------------------

KEY_COLUMNS: tuple[str, ...] = ("config", "dataset", "doc_id", "sent_id", "index", "n_tokens")
COUNT_COLUMNS: dict[str, tuple[str, ...]] = {
    "pos": ("correct", "total"),
    "dep": ("las_correct", "uas_correct", "total", "clas_correct", "clas_total", "clas_pred_total"),
    "con": ("matched", "gold", "test", "parsed", "evalb_error"),
}
CON_COLUMNS: tuple[str, ...] = ("failure", "d2_sample")
FAILURES: tuple[str, ...] = ("unknown_word", "no_derivation", "over_length", "error")
PROVENANCE_SCHEMA = 1
# The provenance of the thesis tables, written beside them (24 Oct 2026).
TABLES_PROVENANCE = "thesis.prov.json"
# The one sentence subset of plan statistics.subset_sensitivity (24 Oct 2026):
# the sentences EVALB scores for both configurations of a Delta.
EVALB_BOTH_PARSERS = "evalb_both_parsers"
# The count that marks a sentence EVALB could not score (evalb.ConCounts).
EVALB_ERROR = "evalb_error"
_COUNTS_FILE = re.compile(rf"counts_({'|'.join(TASKS)})_(.+)\.csv")
_COUNT = re.compile(r"[0-9]+")

STATUSES: tuple[str, ...] = ("exploratory", "confirmatory")
HYPOTHESES: tuple[str, ...] = ("h1a", "h1b", "h2a", "h2b", "h2c")
# The whole hypotheses of brief part A and their parts: H1 is h1a and h1b, H2 is
# h2a, h2b and h2c. A whole hypothesis holds only if all of its parts hold (D.7).
WHOLE: dict[str, tuple[str, ...]] = {
    whole: tuple(part for part in HYPOTHESES if part[:-1] == whole)
    for whole in dict.fromkeys(part[:-1] for part in HYPOTHESES)
}

# The two ways a set is resampled: by the plan's unit of the set, and by the
# plan's statistics.unit_sensitivity.
MAIN, ALTERNATIVE = "main", "alternative"

# The column a hypothesis table gives its tested estimate.
_ESTIMATE_COLUMN = {"h1a": "delta", "h1b": "diff", "h2a": "stat", "h2b": "stat", "h2c": "delta"}
# Short names of metrics in column names (thesis_h1b: delta_las, delta_upos).
_SHORT = {"upos_accuracy": "upos"}


def counts_header(task: str) -> tuple[str, ...]:
    """The columns of counts_<task>_<dataset>.csv, in order."""
    return KEY_COLUMNS + COUNT_COLUMNS[task] + (CON_COLUMNS if task == "con" else ())


class AnalysisError(Exception):
    """A run the analysis refuses: what is wrong, and in which file."""


# --- reading a run -----------------------------------------------------------


@dataclass
class ConfigRows:
    """One configuration's rows of a counts file.

    Attributes:
        index:   Sentence indices, ascending.
        counts:  (rows x k) int64, the columns COUNT_COLUMNS[task].
        failure: The failure code of each row ("" when parsed); con only.
    """

    index: np.ndarray
    counts: np.ndarray
    failure: list[str]


@dataclass
class CountsFile:
    """A counts CSV and its provenance, checked against the contract and the plan."""

    task: str
    dataset: str
    status: str
    path: Path
    sha256: str
    max_sentences: int | None
    doc_id: list[str]
    sent_id: list[str]
    n_tokens: np.ndarray
    d2_sample: np.ndarray | None
    configs: dict[str, ConfigRows]
    notes: list[str] = field(default_factory=list)
    measured: MeasuredUnder | None = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def n_sentences(self) -> int:
        return len(self.doc_id)

    @property
    def smoke(self) -> bool:
        return self.max_sentences is not None


def _count(value: str, where: str, column: str) -> int:
    if not _COUNT.fullmatch(value):
        raise AnalysisError(f"{where}: {column} {value!r} is not a non-negative integer")
    return int(value)


# --- the plan the counts were measured under -----------------------------------


@dataclass(frozen=True)
class MeasuredUnder:
    """The plan a counts file was measured under, as its provenance records it.

    Attributes:
        path:   The plan file as the provenance records it (relative to prog/).
        sha256: Its digest (plan.plan_sha256).
        commit: The commit the stage ran at (git.commit), None if not recorded.
        same:   True when it is the plan analysed; False for an earlier plan
                that measures as the plan analysed does (measurement_parts).
    """

    path: str | None
    sha256: str
    commit: str | None
    same: bool


# A commit as git names it in a provenance record: a hexadecimal hash.
_COMMIT = re.compile(r"[0-9a-f]{7,64}")


def measurement_parts(plan: Plan) -> dict[str, Any]:
    """The parts of a plan that determine the counts a stage writes, by name.

    What the measuring stages read of a plan (stage.py, build.py and, through
    build, the wrappers), each compared in its order:

      training_data              every corpus a model was trained on;
      datasets                   every test set: tasks, status, path or NLTK
                                 sections, source, tokens and resampling;
      configurations             wrapper, model, training, init, load, inputs,
                                 optional, pending and the datasets restriction;
      tasks                      each task's metric with its options (the
                                 EVALB parameters of con) and its members in
                                 order (the order of a counts file's rows)
                                 with their method levels;
      descriptive.d2             the task, configurations and datasets of D2
                                 and its sample (the PCFG rows, d2_sample);
      statistics.bootstrap.seed  the seed the stages also seed Python, NumPy
                                 and torch with (stage.seed_everything).

    The rest of a plan is read by the analysis only: its id, title and date,
    the roles of the members (which the hypotheses imply), the hypotheses,
    D2's versus and report, D3, D4 and the statistics other than the seed.
    """
    d2 = plan.d2
    return {
        "training_data": tuple(plan.training_data.items()),
        "datasets": tuple(plan.datasets.items()),
        "configurations": tuple(plan.configurations.items()),
        "tasks": tuple(
            (code, task.metric, tuple((key, member.level) for key, member in task.members.items()))
            for code, task in plan.tasks.items()
        ),
        "descriptive.d2": (d2.task, d2.configs, d2.datasets, d2.sample),
        "statistics.bootstrap.seed": plan.statistics.bootstrap.seed,
    }


def _why_not_alike(plan: Plan, measured: MeasuredUnder, root: Path) -> str | None:
    """Why counts of an earlier plan cannot be analysed under `plan`; None when they can.

    The proof: the earlier plan is read from git where the provenance says it
    was (`git show <commit>:./<path>`, run in `root`, the project root), its
    digest is the one the provenance records - so it is the file the stage
    read -, it loads under this code as a run loads a plan (strict), and every
    part measurement_parts names is equal to the plan analysed.
    """
    commit, path = measured.commit, measured.path
    if commit is None or not _COMMIT.fullmatch(commit):
        return f"its provenance records no commit to read that plan from (git.commit {commit!r})"
    relative = PurePosixPath(path) if isinstance(path, str) and path else None
    if relative is None or relative.is_absolute() or ".." in relative.parts or path.startswith("<"):
        return f"its provenance records the plan at {path!r}, not at a path inside the project"
    try:
        shown = subprocess.run(
            ["git", "show", f"{commit}:./{relative}"], cwd=root, capture_output=True, timeout=120,
        )
    except FileNotFoundError:
        return "git is not installed, so the plan of that commit cannot be read"
    if shown.returncode != 0:
        message = shown.stderr.decode("utf-8", "replace").strip()
        return f"git show {commit[:12]}:./{relative} failed: {message}"
    digest = plan_sha256(shown.stdout)
    if digest != measured.sha256:
        return (
            f"{relative} at commit {commit[:12]} has sha256 {digest}, not the one the provenance "
            f"records, so it is not the plan the stage read"
        )
    with tempfile.TemporaryDirectory() as folder:
        earlier_path = Path(folder) / relative.name
        earlier_path.write_bytes(shown.stdout)
        try:
            earlier = load(earlier_path, strict=True)
        except PlanError as exc:
            return f"{relative} at commit {commit[:12]} does not load under this code: {exc}"
    ours, theirs = measurement_parts(plan), measurement_parts(earlier)
    differ = [name for name in ours if ours[name] != theirs[name]]
    if differ:
        return (
            f"{relative} at commit {commit[:12]} differs from the plan read here in "
            f"{', '.join(differ)}, which determine the counts"
        )
    return None


def _measured_under(prov: dict, prov_name: str, plan: Plan, root: Path,
                    proven: dict[tuple, str | None]) -> MeasuredUnder:
    """The plan a counts file was measured under: this plan, or an earlier one shown alike.

    `proven` caches the outcome of each proof, so the counts files of one run
    are proved once.

    Raises:
        AnalysisError: If it is another plan that is not shown to measure as
            this one does (_why_not_alike).
    """
    recorded = prov["plan"]
    git = prov.get("git")
    commit = git.get("commit") if isinstance(git, dict) else None
    measured = MeasuredUnder(
        path=recorded.get("path") if isinstance(recorded.get("path"), str) else None,
        sha256=recorded["sha256"],
        commit=commit if isinstance(commit, str) else None,
        same=recorded["sha256"] == plan.sha256,
    )
    if measured.same:
        return measured
    key = (measured.path, measured.sha256, measured.commit)
    if key not in proven:
        proven[key] = _why_not_alike(plan, measured, root)
    if proven[key] is not None:
        raise AnalysisError(
            f"{prov_name}: the counts were written under the plan with sha256 {measured.sha256}; "
            f"the plan read here ({recorded_path(plan.path)}) has {plan.sha256}. Counts of one "
            f"plan are not analysed under another, unless that plan is shown to measure as this "
            f"one does, and it is not: {proven[key]}"
        )
    return measured


def _provenance(path: Path, task: str, dataset: str, plan: Plan, root: Path,
                proven: dict[tuple, str | None]) -> tuple[dict, MeasuredUnder]:
    """The provenance of a counts file, and the plan it names, refused unless it names
    this file and either this plan or an earlier one shown to measure alike."""
    prov_path = path.with_name(f"{path.stem}.prov.json")
    if not prov_path.is_file():
        raise AnalysisError(
            f"{path.name}: no provenance {prov_path.name}; counts without it cannot be "
            f"tied to a plan"
        )
    try:
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AnalysisError(f"{prov_path.name}: unreadable: {exc}") from None

    def get(*keys: str) -> Any:
        value: Any = prov
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                raise AnalysisError(f"{prov_path.name}: no {'.'.join(keys)}")
            value = value[key]
        return value

    if get("schema") != PROVENANCE_SCHEMA:
        raise AnalysisError(
            f"{prov_path.name}: schema {get('schema')!r}; this script reads schema {PROVENANCE_SCHEMA}"
        )
    sha = get("plan", "sha256")
    if not isinstance(sha, str):
        raise AnalysisError(f"{prov_path.name}: plan.sha256 {sha!r} is not a digest")
    measured = _measured_under(prov, prov_path.name, plan, root, proven)
    if get("task") != task:
        raise AnalysisError(f"{prov_path.name}: task {get('task')!r}, the file name says {task!r}")
    if get("dataset", "id") != dataset:
        raise AnalysisError(
            f"{prov_path.name}: dataset {get('dataset', 'id')!r}, the file name says {dataset!r}"
        )
    status = get("dataset", "status")
    if status != plan.datasets[dataset].status:
        raise AnalysisError(
            f"{prov_path.name}: status {status!r}; the plan says {plan.datasets[dataset].status!r}"
        )
    max_sentences = get("run", "max_sentences")
    if max_sentences is not None and (
        isinstance(max_sentences, bool) or not isinstance(max_sentences, int) or max_sentences < 1
    ):
        raise AnalysisError(f"{prov_path.name}: run.max_sentences {max_sentences!r}")
    return prov, measured


def read_counts(path: Path, task: str, dataset: str, plan: Plan, *, root: Path | None = None,
                proven: dict[tuple, str | None] | None = None) -> CountsFile:
    """Read one counts CSV with its provenance, and check both.

    `root` is where git reads an earlier plan from (the project root unless
    given); `proven` caches the proofs of a run's files (_measured_under).

    Raises:
        AnalysisError: For any breach of the contract; see the module docstring.
    """
    prov, measured = _provenance(
        path, task, dataset, plan, project_root() if root is None else root,
        {} if proven is None else proven,
    )
    name = path.name
    data = path.read_bytes()
    expected = counts_header(task)
    reader = csv.reader(io.StringIO(data.decode("utf-8"), newline=""))
    header = next(reader, None)
    if header is None or tuple(header) != expected:
        raise AnalysisError(f"{name}: columns {header}; the contract has {list(expected)}")

    positions = {key: i for i, key in enumerate(plan.tasks[task].members)}
    scored = plan.members_on(task, dataset)
    k = len(COUNT_COLUMNS[task])
    per_config: dict[str, tuple[list[int], list[list[int]], list[str]]] = {}
    described: dict[int, tuple[tuple, str]] = {}
    last = (-1, -1)
    for line, row in enumerate(reader, start=2):
        where = f"{name} line {line}"
        if len(row) != len(expected):
            raise AnalysisError(f"{where}: {len(row)} fields; the header has {len(expected)}")
        config, row_dataset, doc_id, sent_id = row[:4]
        if row_dataset != dataset:
            raise AnalysisError(f"{where}: dataset {row_dataset!r} in a file of {dataset!r}")
        if config not in positions:
            raise AnalysisError(
                f"{where}: {config!r} is not a configuration of task {task} in the plan"
            )
        if config not in scored:
            raise AnalysisError(
                f"{where}: {config!r} is not scored on {dataset} by the plan "
                f"(configurations.{config}.datasets: "
                f"{list(plan.configurations[config].datasets or ())})"
            )
        if not doc_id or not sent_id:
            raise AnalysisError(f"{where}: an empty doc_id or sent_id")
        index = _count(row[4], where, "index")
        n_tokens = _count(row[5], where, "n_tokens")
        counts = [_count(value, where, column) for value, column in zip(row[6:6 + k], COUNT_COLUMNS[task])]
        failure, sample = "", None
        if task == "con":
            failure = row[6 + k]
            sample = _count(row[7 + k], where, "d2_sample")
            if sample > 1:
                raise AnalysisError(f"{where}: d2_sample {sample}; it is 0 or 1")
        position = (positions[config], index)
        if position <= last:
            raise AnalysisError(
                f"{where}: {config} sentence {index} is repeated or out of order; rows are "
                f"ordered by configuration in the plan's order of task {task}, then by index"
            )
        last = position
        sentence = (doc_id, sent_id, n_tokens, sample)
        first = described.setdefault(index, (sentence, config))
        if first[0] != sentence:
            raise AnalysisError(
                f"{where}: sentence {index} is (doc_id, sent_id, n_tokens, d2_sample) "
                f"{sentence} here and {first[0]} in the rows of {first[1]}"
            )
        rows = per_config.setdefault(config, ([], [], []))
        rows[0].append(index)
        rows[1].append(counts)
        rows[2].append(failure)
    if not described:
        raise AnalysisError(f"{name}: no rows")
    n = len(described)
    if max(described) != n - 1:
        raise AnalysisError(f"{name}: the sentence indices are not 0..{n - 1}")
    table = [described[i][0] for i in range(n)]
    file = CountsFile(
        task=task,
        dataset=dataset,
        status=plan.datasets[dataset].status,
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        max_sentences=prov["run"]["max_sentences"],
        doc_id=[sentence[0] for sentence in table],
        sent_id=[sentence[1] for sentence in table],
        n_tokens=np.array([sentence[2] for sentence in table], dtype=np.int64),
        d2_sample=(
            np.array([sentence[3] for sentence in table], dtype=np.int64) if task == "con" else None
        ),
        configs={
            key: ConfigRows(
                index=np.array(index, dtype=np.int64),
                counts=np.array(counts, dtype=np.int64).reshape(len(index), k),
                failure=failures,
            )
            for key, (index, counts, failures) in per_config.items()
        },
        measured=measured,
    )
    recorded = prov["dataset"].get("sentences") if isinstance(prov["dataset"], dict) else None
    if isinstance(recorded, int) and not isinstance(recorded, bool):
        expected_n = recorded if file.max_sentences is None else min(file.max_sentences, recorded)
        if n != expected_n:
            raise AnalysisError(
                f"{name}: {n} sentences; its provenance records {recorded}"
                + ("" if file.max_sentences is None else f" and max_sentences {file.max_sentences}")
            )
    _check_coverage(file, plan)
    for key, rows in file.configs.items():
        _check_counts(file, key, rows)
    # Beyond the contract, read if there: what the stage itself recorded as
    # not done. A missing configuration a contrast needs is refused anyway.
    failed = prov.get("failed")
    if isinstance(failed, dict) and failed:
        file.notes.append(
            f"{name}: the stage recorded failed configurations, which have no rows: "
            + "; ".join(f"{key} ({message})" for key, message in failed.items())
        )
    if isinstance(prov["run"], dict) and prov["run"].get("completed") is False:
        file.notes.append(f"{name}: its provenance says the stage did not finish (run.completed false)")
    return file


def _d2_configs(plan: Plan, task: str) -> tuple[str, ...]:
    """The configurations that run on the D2 sample only."""
    return plan.d2.configs if task == plan.d2.task else ()


def _check_coverage(file: CountsFile, plan: Plan) -> None:
    """Every configuration has a row for every sentence; a D2 one for every sampled one."""
    sampled = (
        np.flatnonzero(file.d2_sample) if file.d2_sample is not None else np.array([], dtype=np.int64)
    )
    for key, rows in file.configs.items():
        if key in _d2_configs(plan, file.task):
            if not np.array_equal(rows.index, sampled):
                raise AnalysisError(
                    f"{file.name}: {key} runs on the D2 sample only, and has rows for "
                    f"{rows.index.size} sentences; the file marks {sampled.size} as sampled "
                    f"(d2_sample), and the two sets differ"
                )
        elif rows.index.size != file.n_sentences:
            raise AnalysisError(
                f"{file.name}: {key} has rows for {rows.index.size} of the "
                f"{file.n_sentences} sentences"
            )


def _check_counts(file: CountsFile, key: str, rows: ConfigRows) -> None:
    """Refuse counts that no sentence can have."""
    column = {name: rows.counts[:, i] for i, name in enumerate(COUNT_COLUMNS[file.task])}
    tokens = file.n_tokens[rows.index]
    if file.task == "pos":
        bad = (column["correct"] > column["total"]) | (column["total"] != tokens)
        rule = "correct <= total = n_tokens"
    elif file.task == "dep":
        las, uas, total = column["las_correct"], column["uas_correct"], column["total"]
        clas, gold, predicted = column["clas_correct"], column["clas_total"], column["clas_pred_total"]
        bad = ~(
            (total == tokens) & (clas <= las) & (las <= uas) & (uas <= total)
            & (clas <= np.minimum(gold, predicted)) & (np.maximum(gold, predicted) <= total)
        )
        rule = (
            "clas_correct <= las_correct <= uas_correct <= total = n_tokens, "
            "clas_correct <= clas_total and clas_pred_total, both <= total"
        )
    else:
        matched, gold, test = column["matched"], column["gold"], column["test"]
        parsed, error = column["parsed"], column["evalb_error"]
        failure = np.array(rows.failure, dtype=object)
        unparsed_reason = np.isin(failure, FAILURES)
        bad = (
            (matched > np.minimum(gold, test)) | (parsed > 1) | (error > 1)
            | ((parsed == 1) != (failure == ""))
            | ((parsed == 0) & ~unparsed_reason)
            | ((parsed == 0) & ((test > 0) | (error > 0)))
            | ((error == 1) & ((matched > 0) | (gold > 0) | (test > 0)))
        )
        rule = (
            "matched <= gold and test; parsed and evalb_error 0 or 1; failure empty exactly "
            f"when parsed, else one of {list(FAILURES)}; no tree: test 0; an EVALB error "
            "sentence: matched, gold and test 0"
        )
    if bad.any():
        first = int(rows.index[np.argmax(bad)])
        raise AnalysisError(
            f"{file.name}: {key} has counts no sentence can have at sentence {first} "
            f"({int(bad.sum())} rows in all); the rule is: {rule}"
        )


def _unit_codes(file: CountsFile, unit: str) -> tuple[np.ndarray, int]:
    """Each sentence's unit number, and the number of units.

    A sentence is its own unit; a document or file is the sentences that share
    a doc_id, numbered in the order the set first shows them.
    """
    if unit == "sentence":
        return np.arange(file.n_sentences, dtype=np.int64), file.n_sentences
    seen: dict[str, int] = {}
    codes = np.fromiter(
        (seen.setdefault(doc, len(seen)) for doc in file.doc_id), dtype=np.int64,
        count=file.n_sentences,
    )
    return codes, len(seen)


def datasets_of(plan: Plan, task: str) -> list[str]:
    """The datasets that list `task`, in plan order."""
    return [key for key, dataset in plan.datasets.items() if task in dataset.tasks]


def required_configs(plan: Plan, task: str, dataset: str, file: CountsFile) -> dict[str, list[str]]:
    """Configuration -> the sections that compare it on this task and set."""
    needed: dict[str, list[str]] = {}

    def need(keys, why: str) -> None:
        for key in keys:
            needed.setdefault(key, []).append(why)

    for chain in plan.h1a.get(task, ()):
        need(chain.configs, f"h1a {chain.id}")
    if dataset in plan.h1b.datasets and task in {leg.task for leg in plan.h1b.legs}:
        need((plan.h1b.baseline, plan.h1b.treatment), "h1b")
    if dataset in plan.h2a_datasets_of(task):
        for pair in plan.h2a.get(task, ()):
            need((pair.baseline, pair.treatment), f"h2a {pair.id}")
    transition = plan.h2b.get(task)
    if transition is not None and dataset in (transition.source, transition.target):
        need(transition.configs, "h2b")
    contrast = plan.h2c.get(task)
    if contrast is not None and dataset in {test.dataset for test in contrast.tests}:
        need((contrast.baseline, contrast.treatment), "h2c")
    if task == plan.d2.task and dataset in plan.d2.datasets:
        need(plan.d2.versus, "d2")
        # A run of the first N sentences may hold no sampled sentence, and
        # then the PCFG configurations have no row at all.
        if file.d2_sample is not None and file.d2_sample.any():
            need(plan.d2.configs, "d2")
    # D3 and D4 on the sets where the plan scores all their members only
    # (since 24 Oct 2026: D3 on EWT test only).
    if task == plan.d3.task and dataset in plan.d3_datasets():
        need((plan.d3.config, *plan.d3.versus), "d3")
    if task == plan.d4.task and dataset in plan.d4_datasets():
        need((plan.d4.baseline, plan.d4.treatment), "d4")
    return needed


def read_run(run_dir: Path, plan: Plan, *, root: Path | None = None
             ) -> tuple[dict[tuple[str, str], CountsFile], list[str]]:
    """Every counts file of a run, checked one by one and against each other.

    `root` is where git reads the plan of a counts file measured under an
    earlier plan from: the project root unless given.

    Returns:
        ({(task, dataset): file} in task, then plan dataset order; warnings).
    """
    csv_dir = run_dir / "csv"
    if not csv_dir.is_dir():
        raise AnalysisError(f"no directory {csv_dir}")
    root = project_root() if root is None else root
    proven: dict[tuple, str | None] = {}
    found: dict[tuple[str, str], CountsFile] = {}
    for path in sorted(csv_dir.glob("counts_*.csv")):
        match = _COUNTS_FILE.fullmatch(path.name)
        if match is None:
            raise AnalysisError(f"{path.name}: not counts_<task>_<dataset>.csv with a task of {list(TASKS)}")
        task, dataset = match.groups()
        if dataset not in plan.datasets:
            raise AnalysisError(f"{path.name}: dataset {dataset!r} is not in the plan")
        if task not in plan.datasets[dataset].tasks:
            raise AnalysisError(f"{path.name}: the plan does not list task {task} for {dataset}")
        found[(task, dataset)] = read_counts(path, task, dataset, plan, root=root, proven=proven)
    if not found:
        raise AnalysisError(f"no counts_*.csv in {csv_dir}")
    files = {
        (task, dataset): found[(task, dataset)]
        for task in TASKS for dataset in plan.datasets if (task, dataset) in found
    }

    warnings: list[str] = [note for file in files.values() for note in file.notes]
    for (task, dataset), file in files.items():
        missing = {
            key: why for key, why in required_configs(plan, task, dataset, file).items()
            if key not in file.configs
        }
        if missing:
            listed = "; ".join(f"{key} ({', '.join(why)})" for key, why in missing.items())
            raise AnalysisError(f"{file.name}: no rows of configurations a contrast compares: {listed}")
    for dataset in plan.datasets:
        same_set = [file for (_, key), file in files.items() if key == dataset]
        for other in same_set[1:]:
            first = same_set[0]
            if (first.doc_id, first.sent_id) != (other.doc_id, other.sent_id) or not np.array_equal(
                first.n_tokens, other.n_tokens
            ):
                raise AnalysisError(
                    f"{first.name} and {other.name} describe different sentences or units "
                    f"({first.n_sentences} and {other.n_sentences} sentences); H1b takes both "
                    f"Deltas from the same replicate of the same units, so both files must "
                    f"come from one run over the same sentences"
                )
        for file in same_set:
            unit = plan.datasets[dataset].resampling.unit
            _, n_units = _unit_codes(file, unit)
            expected = plan.datasets[dataset].resampling.expected_units
            if file.smoke:
                note = (
                    f"{file.name}: a smoke run (max_sentences {file.max_sentences}): "
                    f"{file.n_sentences} sentences in {n_units} {unit} unit(s). Its intervals "
                    f"describe these sentences, not the set, and expected_units is not checked."
                )
                if n_units == 1:
                    note += (
                        " With one unit every replicate is the whole set, so each interval of "
                        "this set is its point estimate (half-width 0) and its classes and "
                        "verdicts say nothing."
                    )
                warnings.append(note)
            elif expected is not None and n_units != expected:
                raise AnalysisError(
                    f"{file.name}: {n_units} {unit} units; the plan expects {expected}"
                )
    return files, warnings


# --- the run as resampled sums -----------------------------------------------


@dataclass
class Step:
    """Delta_step,max: the H1a step with the largest point estimate, and its Estimate."""

    chain: str
    lower: str
    higher: str
    estimate: Estimate


class Run:
    """The counts of a run, resampled per set and unit on demand, and cached."""

    def __init__(self, plan: Plan, files: dict[tuple[str, str], CountsFile]) -> None:
        self.plan = plan
        self.files = files
        bootstrap = plan.statistics.bootstrap
        self.replicates = bootstrap.replicates
        self.confidence = bootstrap.confidence
        self.seed = bootstrap.seed
        self.delta = plan.statistics.delta
        self._sums: dict[tuple[str, str], tuple[ClusterSums, int]] = {}
        self._measures: dict[tuple, tuple[float, np.ndarray]] = {}
        self._steps: dict[tuple, Step] = {}

    def has(self, needs: tuple[tuple[str, str], ...]) -> bool:
        return all(need in self.files for need in needs)

    def status(self, *datasets: str) -> str:
        """confirmatory if any of the sets is, else exploratory."""
        statuses = {self.plan.datasets[dataset].status for dataset in datasets}
        return "confirmatory" if "confirmatory" in statuses else "exploratory"

    def unit(self, dataset: str, kind: str) -> str:
        """The resampling unit of a set: the plan's, or its unit_sensitivity."""
        if kind == MAIN:
            return self.plan.datasets[dataset].resampling.unit
        return self.plan.statistics.unit_sensitivity

    def unit_label(self, needs: tuple[tuple[str, str], ...], kind: str) -> str:
        units = list(dict.fromkeys(self.unit(dataset, kind) for dataset in dict.fromkeys(d for _, d in needs)))
        return "->".join(units)

    def file_of(self, dataset: str) -> CountsFile:
        return next(file for (_, key), file in self.files.items() if key == dataset)

    def n_units(self, dataset: str, kind: str = MAIN) -> int:
        return _unit_codes(self.file_of(dataset), self.unit(dataset, kind))[1]

    def sums(self, dataset: str, kind: str) -> ClusterSums:
        """One bootstrap of a set, shared by every configuration of every task on it."""
        unit = self.unit(dataset, kind)
        cached = self._sums.get((dataset, unit))
        if cached is not None:
            return cached[0]
        codes, n_units = _unit_codes(self.file_of(dataset), unit)
        counts = {}
        for (task, key), file in self.files.items():
            if key != dataset:
                continue
            for config, rows in file.configs.items():
                if config in _d2_configs(self.plan, task):
                    continue
                per_unit = np.zeros((n_units, rows.counts.shape[1]), dtype=np.int64)
                np.add.at(per_unit, codes[rows.index], rows.counts)
                counts[(task, config)] = per_unit
        sums = cluster_bootstrap(counts, self.replicates, seed_sequence(self.seed, dataset, unit))
        self._sums[(dataset, unit)] = (sums, n_units)
        return sums

    def measure(self, task: str, dataset: str, config: str, name: str, kind: str) -> tuple[float, np.ndarray]:
        """(point, replicates) of one metric of one configuration on one set."""
        key = (task, dataset, config, name, self.unit(dataset, kind))
        if key not in self._measures:
            sums = self.sums(dataset, kind)
            columns = COUNT_COLUMNS[task]
            point = float(metric(name, columns, sums.points[(task, config)]))
            self._measures[key] = (point, metric(name, columns, sums.replicates[(task, config)]))
        return self._measures[key]

    def point(self, task: str, dataset: str, config: str, name: str) -> float:
        return self.measure(task, dataset, config, name, MAIN)[0]

    def contrast(self, task: str, dataset: str, a: str, b: str, name: str, kind: str) -> Estimate:
        """Delta = M(a) - M(b) on one set, with its CI.

        Raises:
            AnalysisError: If either metric has nothing to measure (a zero
                denominator) on the set or in a replicate.
        """
        point_a, rep_a = self.measure(task, dataset, a, name, kind)
        point_b, rep_b = self.measure(task, dataset, b, name, kind)
        try:
            return delta_ci(rep_a, rep_b, point_a, point_b, self.confidence)
        except ValueError as exc:
            raise AnalysisError(f"{name} of {a} - {b} on {dataset} ({task}): {exc}") from None

    def drop(self, task: str, source: str, target: str, config: str, kind: str) -> tuple[float, np.ndarray, float, np.ndarray]:
        name = self.plan.tasks[task].metric.name
        return (
            *self.measure(task, source, config, name, kind),
            *self.measure(task, target, config, name, kind),
        )

    def step_max(self, task: str, dataset: str, kind: str) -> Step:
        """Delta_step,max of a task on a set: chosen on the point estimates, in plan order on a tie."""
        cache = (task, dataset, self.unit(dataset, kind))
        if cache in self._steps:
            return self._steps[cache]
        chains = self.plan.h1a.get(task, ())
        if not chains:
            raise AnalysisError(f"task {task} has no H1a chain, so Delta_step,max is not defined")
        name = self.plan.tasks[task].metric.name
        best = None
        for chain in chains:
            for lower, higher in pairwise(chain.configs):
                point = self.point(task, dataset, higher, name) - self.point(task, dataset, lower, name)
                if best is None or point > best[0]:
                    best = (point, chain.id, lower, higher)
        _, chain, lower, higher = best
        step = Step(chain, lower, higher, self.contrast(task, dataset, higher, lower, name, kind))
        self._steps[cache] = step
        return step

    def scored_by_both(self, task: str, dataset: str, a: str, b: str) -> np.ndarray:
        """Which sentences of a set EVALB scored for both a and b: no EVALB error for either."""
        file = self.files[(task, dataset)]
        column = COUNT_COLUMNS[task].index(EVALB_ERROR)
        scored = np.ones(file.n_sentences, dtype=bool)
        for config in (a, b):
            rows = file.configs[config]
            scored[rows.index[rows.counts[:, column] == 1]] = False
        return scored

    def subset_contrast(self, task: str, dataset: str, a: str, b: str, name: str,
                        scored: np.ndarray) -> Estimate:
        """Delta = M(a) - M(b) on the sentences `scored` only, with its CI.

        The same unit and the same index matrix as the main estimate: the
        counts of every other sentence are left out of the per-unit sums, every
        unit of the set stays, and the stream is the set's seed_sequence for
        its plan unit, so cluster_bootstrap draws the same B x units indices
        as for every other contrast on the set. A filter on the counts, not a
        new draw.

        Raises:
            ValueError: If a metric has nothing to measure on those sentences
                or in a replicate.
        """
        file = self.files[(task, dataset)]
        unit = self.unit(dataset, MAIN)
        codes, n_units = _unit_codes(file, unit)
        counts = {}
        for config in (a, b):
            rows = file.configs[config]
            on = scored[rows.index]
            per_unit = np.zeros((n_units, rows.counts.shape[1]), dtype=np.int64)
            np.add.at(per_unit, codes[rows.index[on]], rows.counts[on])
            counts[config] = per_unit
        sums = cluster_bootstrap(counts, self.replicates, seed_sequence(self.seed, dataset, unit))
        columns = COUNT_COLUMNS[task]
        points = {config: float(metric(name, columns, sums.points[config])) for config in counts}
        replicates = {config: metric(name, columns, sums.replicates[config]) for config in counts}
        return delta_ci(replicates[a], replicates[b], points[a], points[b], self.confidence)


# --- the hypotheses ----------------------------------------------------------


@dataclass
class Part:
    """One part of a hypothesis: its row, and its estimate once computed.

    `estimate` stays None when a file it needs is not in the run; `fields`
    then stays empty.
    """

    hypothesis: str
    row_id: str
    dataset: str
    status: str
    needs: tuple[tuple[str, str], ...]
    rule: str
    estimate: Estimate | None = None
    fields: dict[str, Any] = field(default_factory=dict)


def _ci_fields(prefix: str, estimate: Estimate) -> dict[str, float]:
    return {
        prefix: estimate.point,
        f"{prefix}_ci_low": estimate.ci_low,
        f"{prefix}_ci_high": estimate.ci_high,
    }


def h1a_parts(run: Run, kind: str) -> Iterator[Part]:
    plan = run.plan
    for task in TASKS:
        name = plan.tasks[task].metric.name
        members = plan.tasks[task].members
        for dataset in datasets_of(plan, task):
            for chain in plan.h1a.get(task, ()):
                for lower, higher in pairwise(chain.configs):
                    part = Part(
                        "h1a", f"{task}/{chain.id}/{dataset}/{lower}->{higher}", dataset,
                        run.status(dataset), ((task, dataset),), ABOVE_ZERO,
                    )
                    if run.has(part.needs):
                        part.estimate = run.contrast(task, dataset, higher, lower, name, kind)
                        part.fields = {
                            "task": task, "chain": chain.id, "dataset": dataset,
                            "status": part.status, "lower": lower, "higher": higher,
                            "lower_level": members[lower].level.code,
                            "higher_level": members[higher].level.code,
                            "m_lower": run.point(task, dataset, lower, name),
                            "m_higher": run.point(task, dataset, higher, name),
                        }
                    yield part


def _h1b_legs(plan: Plan):
    if len(plan.h1b.legs) != 2:
        raise AnalysisError(f"H1b needs two legs; the plan has {len(plan.h1b.legs)}")
    return plan.h1b.legs


def _leg_stem(leg) -> str:
    return f"delta_{_SHORT.get(leg.metric, leg.metric)}"


def h1b_parts(run: Run, kind: str) -> Iterator[Part]:
    plan, h1b = run.plan, run.plan.h1b
    legs = _h1b_legs(plan)
    for dataset in h1b.datasets:
        needs = tuple((leg.task, dataset) for leg in legs)
        part = Part(
            "h1b",
            f"{dataset}/{h1b.baseline}->{h1b.treatment}/{legs[0].metric}-{legs[1].metric}",
            dataset, run.status(dataset), needs, ABOVE_ZERO,
        )
        if run.has(needs):
            deltas = [
                run.contrast(leg.task, dataset, h1b.treatment, h1b.baseline, leg.metric, kind)
                for leg in legs
            ]
            part.estimate = diff_of_deltas_ci(deltas[0], deltas[1], run.confidence)
            part.fields = {
                "dataset": dataset, "status": part.status, "is_verdict": dataset == h1b.verdict,
                "baseline": h1b.baseline, "treatment": h1b.treatment,
            }
            for leg, delta in zip(legs, deltas):
                part.fields.update(_ci_fields(_leg_stem(leg), delta))
        yield part


def _step_fields(step: Step) -> dict[str, Any]:
    return {
        "step_chain": step.chain, "step_lower": step.lower, "step_higher": step.higher,
        "step_max": step.estimate.point,
    }


def h2a_parts(run: Run, kind: str) -> Iterator[Part]:
    plan = run.plan
    for task in TASKS:
        name = plan.tasks[task].metric.name
        # The plan's h2a.datasets only: no pair is formed on another set, not
        # even one whose counts are in the run (GENTLE, since 24 Oct 2026).
        for dataset in plan.h2a_datasets_of(task):
            for pair in plan.h2a.get(task, ()):
                part = Part(
                    "h2a", f"{task}/{pair.id}/{dataset}/{pair.baseline}->{pair.treatment}",
                    dataset, run.status(dataset), ((task, dataset),), ABOVE_ZERO,
                )
                if run.has(part.needs):
                    source = run.contrast(task, dataset, pair.treatment, pair.baseline, name, kind)
                    step = run.step_max(task, dataset, kind)
                    part.estimate = diff_of_deltas_ci(
                        source, step.estimate, run.confidence, absolute_first=True
                    )
                    part.fields = {
                        "task": task, "pair": pair.id, "dataset": dataset, "status": part.status,
                        "baseline": pair.baseline, "treatment": pair.treatment,
                        "m_baseline": run.point(task, dataset, pair.baseline, name),
                        "m_treatment": run.point(task, dataset, pair.treatment, name),
                        **_ci_fields("delta_source", source),
                        **_step_fields(step),
                    }
                yield part


def h2b_parts(run: Run, kind: str) -> Iterator[Part]:
    plan = run.plan
    for task in TASKS:
        transition = plan.h2b.get(task)
        if transition is None:
            continue
        source, target = transition.source, transition.target
        for config in transition.configs:
            needs = ((task, source), (task, target))
            part = Part(
                "h2b", f"{task}/{config}/{source}->{target}", f"{source}->{target}",
                run.status(source, target), needs, ABOVE_ZERO,
            )
            if run.has(needs):
                point_in, rep_in, point_out, rep_out = run.drop(task, source, target, config, kind)
                step = run.step_max(task, source, kind)
                try:
                    drop, part.estimate = two_set_drop_ci(
                        rep_in, rep_out, point_in, point_out, step.estimate, run.confidence
                    )
                except ValueError as exc:
                    raise AnalysisError(f"h2b {part.row_id}: {exc}") from None
                part.fields = {
                    "task": task, "config": config, "source": source, "target": target,
                    "status": part.status, "m_in": point_in, "m_out": point_out,
                    **_ci_fields("drop", drop),
                    **_step_fields(step),
                }
            yield part


def h2c_parts(run: Run, kind: str) -> Iterator[Part]:
    plan = run.plan
    for task in TASKS:
        contrast = plan.h2c.get(task)
        if contrast is None:
            continue
        name = plan.tasks[task].metric.name
        for test in contrast.tests:
            part = Part(
                "h2c",
                f"{task}/{test.dataset}/{contrast.baseline}->{contrast.treatment}/{test.decision}",
                test.dataset, run.status(test.dataset), ((task, test.dataset),),
                plan.statistics.decisions[test.decision],
            )
            if run.has(part.needs):
                part.estimate = run.contrast(
                    task, test.dataset, contrast.treatment, contrast.baseline, name, kind
                )
                part.fields = {
                    "task": task, "dataset": test.dataset, "status": part.status,
                    "decision": test.decision, "baseline": contrast.baseline,
                    "treatment": contrast.treatment,
                }
            yield part


PARTS: dict[str, Callable[[Run, str], Iterator[Part]]] = {
    "h1a": h1a_parts, "h1b": h1b_parts, "h2a": h2a_parts, "h2b": h2b_parts, "h2c": h2c_parts,
}


# --- the thesis tables ---------------------------------------------------------


def hypothesis_headers(plan: Plan) -> dict[str, tuple[str, ...]]:
    """The columns of thesis_h1a.csv ... thesis_h2c.csv."""
    legs = _h1b_legs(plan)
    leg_columns = tuple(
        column for leg in legs
        for column in (_leg_stem(leg), f"{_leg_stem(leg)}_ci_low", f"{_leg_stem(leg)}_ci_high")
    )
    step = ("step_chain", "step_lower", "step_higher", "step_max")
    # The tested estimate's CI, half-width and class, then the verdict (brief F step 6).
    interval = ("ci_low", "ci_high", "half_width", "class")
    return {
        "h1a": ("task", "chain", "dataset", "status", "lower", "higher", "lower_level",
                "higher_level", "m_lower", "m_higher", "delta", *interval, "holds"),
        "h1b": ("dataset", "status", "is_verdict", "baseline", "treatment", *leg_columns,
                "diff", *interval, "holds"),
        "h2a": ("task", "pair", "dataset", "status", "baseline", "treatment", "m_baseline",
                "m_treatment", "delta_source", "delta_source_ci_low", "delta_source_ci_high",
                *step, "stat", *interval, "holds"),
        "h2b": ("task", "config", "source", "target", "status", "m_in", "m_out", "drop",
                "drop_ci_low", "drop_ci_high", *step, "stat", *interval, "holds"),
        "h2c": ("task", "dataset", "status", "decision", "baseline", "treatment", "delta",
                *interval, "holds"),
    }


SCORES_HEADER = ("task", "dataset", "status", "config", "level", "roles", "metric", "value",
                 "sentences", "units", "uas", "clas", "parsed", "evalb_errors")
D2_COVERAGE_HEADER = ("dataset", "status", "config", "sampled", "parsed", *FAILURES, "coverage")
D2_F1_HEADER = ("dataset", "status", "pcfg_config", "config", "sentences", "f1")
D3_HEADER = ("dataset", "status", "metric", "baseline", "other", "m_baseline", "m_other",
             "delta", "ci_low", "ci_high", "half_width")
D4_HEADER = ("dataset", "status", "baseline", "treatment", "m_baseline", "m_treatment", "delta",
             "ci_low", "ci_high", "half_width")
DESCRIPTIVE_HEADER = ("task", "config", "source", "target", "status", "m_in", "m_out", "drop",
                      "ci_low", "ci_high", "half_width")
SENSITIVITY_HEADER = ("hypothesis", "row", "dataset", "status", "unit", "delta_threshold",
                      "estimate", "ci_low", "ci_high", "half_width", "class", "holds")
VERDICTS_HEADER = ("hypothesis", "status", "parts", "parts_holding", "holds", "note")


def hypothesis_row(part: Part, header: tuple[str, ...], delta: float) -> dict[str, Any]:
    """A thesis_h*.csv row: the part's fields, its estimate, class and verdict."""
    estimate = part.estimate
    row = dict(part.fields)
    row[_ESTIMATE_COLUMN[part.hypothesis]] = estimate.point
    row.update(ci_low=estimate.ci_low, ci_high=estimate.ci_high, half_width=estimate.half_width)
    if "class" in header:
        row["class"] = classify(estimate, delta)
    row["holds"] = decide(part.rule, estimate, delta)
    return row


def _roles(roles: frozenset[str]) -> str:
    return "+".join(role for role in ROLES if role in roles)


def score_rows(run: Run) -> list[dict[str, Any]]:
    """Point values of every configuration on every set: thesis_scores.csv."""
    rows = []
    for (task, dataset), file in run.files.items():
        name = run.plan.tasks[task].metric.name
        columns = COUNT_COLUMNS[task]
        codes, _ = _unit_codes(file, run.unit(dataset, MAIN))
        for config, counts in file.configs.items():
            member = run.plan.tasks[task].members[config]
            sums = counts.counts.sum(axis=0)
            row = {
                "task": task, "dataset": dataset, "status": file.status, "config": config,
                "level": member.level.code, "roles": _roles(member.roles), "metric": name,
                "value": float(metric(name, columns, sums)),
                "sentences": int(counts.index.size),
                "units": int(np.unique(codes[counts.index]).size),
                "uas": None, "clas": None, "parsed": None, "evalb_errors": None,
            }
            if task == "dep":
                row["uas"] = float(metric("uas", columns, sums))
                row["clas"] = float(metric("clas", columns, sums))
            if task == "con":
                row["parsed"] = int(sums[columns.index("parsed")])
                row["evalb_errors"] = int(sums[columns.index("evalb_error")])
            rows.append(row)
    return rows


def d2_rows(run: Run) -> tuple[list[dict], list[dict]]:
    """D2: PCFG coverage by failure reason, and F1 on the sentences the PCFG parsed."""
    plan, d2 = run.plan, run.plan.d2
    columns = COUNT_COLUMNS[d2.task]
    parsed_column = columns.index("parsed")
    coverage, f1 = [], []
    for dataset in d2.datasets:
        file = run.files.get((d2.task, dataset))
        if file is None:
            continue
        sampled = int(file.d2_sample.sum())
        for config in d2.configs:
            rows = file.configs.get(config)
            failures = list(rows.failure) if rows is not None else []
            parsed = int(rows.counts[:, parsed_column].sum()) if rows is not None else 0
            coverage.append({
                "dataset": dataset, "status": file.status, "config": config, "sampled": sampled,
                "parsed": parsed, **{reason: failures.count(reason) for reason in FAILURES},
                "coverage": 100 * parsed / sampled if sampled else None,
            })
            if rows is None:
                continue
            processed = rows.index[rows.counts[:, parsed_column] == 1]
            for other in (config, *d2.versus):
                other_rows = file.configs[other]
                on = np.isin(other_rows.index, processed)
                f1.append({
                    "dataset": dataset, "status": file.status, "pcfg_config": config,
                    "config": other, "sentences": int(processed.size),
                    "f1": float(metric("bracket_f1", columns, other_rows.counts[on].sum(axis=0))),
                })
    return coverage, f1


def d3_rows(run: Run) -> list[dict]:
    """D3 on the sets where the plan scores spacy_sm and every UD parser (EWT test)."""
    d3 = run.plan.d3
    rows = []
    for dataset in run.plan.d3_datasets():
        if (d3.task, dataset) not in run.files:
            continue
        for name in d3.metrics:
            for other in d3.versus:
                estimate = run.contrast(d3.task, dataset, other, d3.config, name, MAIN)
                rows.append({
                    "dataset": dataset, "status": run.status(dataset), "metric": name,
                    "baseline": d3.config, "other": other,
                    "m_baseline": run.point(d3.task, dataset, d3.config, name),
                    "m_other": run.point(d3.task, dataset, other, name),
                    "delta": estimate.point, "ci_low": estimate.ci_low,
                    "ci_high": estimate.ci_high, "half_width": estimate.half_width,
                })
    return rows


def d4_rows(run: Run) -> list[dict]:
    """D4 on the sets where the plan scores both of its configurations (EWT test, GENTLE)."""
    d4 = run.plan.d4
    name = run.plan.tasks[d4.task].metric.name
    rows = []
    for dataset in run.plan.d4_datasets():
        if (d4.task, dataset) not in run.files:
            continue
        estimate = run.contrast(d4.task, dataset, d4.treatment, d4.baseline, name, MAIN)
        rows.append({
            "dataset": dataset, "status": run.status(dataset), "baseline": d4.baseline,
            "treatment": d4.treatment,
            "m_baseline": run.point(d4.task, dataset, d4.baseline, name),
            "m_treatment": run.point(d4.task, dataset, d4.treatment, name),
            "delta": estimate.point, "ci_low": estimate.ci_low, "ci_high": estimate.ci_high,
            "half_width": estimate.half_width,
        })
    return rows


def descriptive_drops(plan: Plan) -> list[tuple[str, str]]:
    """(task, configuration) of every member whose only role in an H2b task is descriptive
    and which the plan scores on the transition's source and target."""
    return [
        (task, key)
        for task in TASKS if task in plan.h2b
        for key, member in plan.tasks[task].members.items() if member.roles == {"descriptive"}
        and {plan.h2b[task].source, plan.h2b[task].target} <= set(plan.scored_on(key))
    ]


def descriptive_rows(run: Run) -> tuple[list[dict], list[str]]:
    """The source -> target drop of the descriptive members, and what could not be computed."""
    rows, skipped = [], []
    for task, config in descriptive_drops(run.plan):
        transition = run.plan.h2b[task]
        source, target = transition.source, transition.target
        if not run.has(((task, source), (task, target))):
            continue
        absent = [
            run.files[(task, dataset)].name for dataset in (source, target)
            if config not in run.files[(task, dataset)].configs
        ]
        if absent:
            skipped.append(f"{config}: no rows in {', '.join(absent)}")
            continue
        point_in, rep_in, point_out, rep_out = run.drop(task, source, target, config, MAIN)
        try:
            drop = delta_ci(rep_in, rep_out, point_in, point_out, run.confidence)
        except ValueError as exc:
            raise AnalysisError(f"the drop of {config} {source} -> {target}: {exc}") from None
        rows.append({
            "task": task, "config": config, "source": source, "target": target,
            "status": run.status(source, target), "m_in": point_in, "m_out": point_out,
            "drop": drop.point, "ci_low": drop.ci_low, "ci_high": drop.ci_high,
            "half_width": drop.half_width,
        })
    return rows, skipped


def sensitivity_rows(run: Run, main: dict[str, list[Part]], alternative: dict[str, list[Part]]) -> list[dict]:
    """Every hypothesis row at every threshold, and with the alternative unit at delta."""
    thresholds = sorted({run.delta, *run.plan.statistics.delta_sensitivity})
    rows = []
    for hypothesis in HYPOTHESES:
        for part, other in zip(main[hypothesis], alternative[hypothesis]):
            if part.estimate is None:
                continue
            unit = run.unit_label(part.needs, MAIN)
            versions = [(unit, threshold, part.estimate) for threshold in thresholds]
            other_unit = run.unit_label(other.needs, ALTERNATIVE)
            if other_unit != unit:
                versions.append((other_unit, run.delta, other.estimate))
            for unit_name, threshold, estimate in versions:
                rows.append({
                    "hypothesis": hypothesis, "row": part.row_id, "dataset": part.dataset,
                    "status": part.status, "unit": unit_name, "delta_threshold": repr(float(threshold)),
                    "estimate": estimate.point, "ci_low": estimate.ci_low,
                    "ci_high": estimate.ci_high, "half_width": estimate.half_width,
                    "class": classify(estimate, threshold),
                    "holds": decide(part.rule, estimate, threshold),
                })
    return rows


@dataclass
class SubsetRow:
    """One Delta of the plan's statistics.subset_sensitivity, beside its main estimate.

    `row_id` is the row's id in its hypothesis table (h1a), or the
    Delta_step,max term of the H2b rows of a transition (h2b); `sentences` of
    `of` sentences of the set are in the subset.
    """

    hypothesis: str
    row_id: str
    dataset: str
    status: str
    unit: str
    sentences: int
    of: int
    estimate: Estimate
    main: Estimate

    def csv_row(self, subset: str) -> dict[str, Any]:
        """Its row in thesis_sensitivity.csv: no threshold, class or verdict (descriptive)."""
        return {
            "hypothesis": self.hypothesis, "row": f"{self.row_id}/{subset}={self.sentences}",
            "dataset": self.dataset, "status": self.status, "unit": self.unit,
            "delta_threshold": None, "estimate": self.estimate.point,
            "ci_low": self.estimate.ci_low, "ci_high": self.estimate.ci_high,
            "half_width": self.estimate.half_width, "class": None, "holds": None,
        }


def subset_rows(run: Run, main: dict[str, list[Part]]) -> tuple[list[SubsetRow], list[str]]:
    """The plan's statistics.subset_sensitivity, and what could not be computed.

    evalb_both_parsers (24 Oct 2026): every Delta between two configurations of
    a task scored by EVALB (its counts have evalb_error) on one set - each H1a
    step of the task on each set in the run, and the Delta_step,max term of the
    task's H2b rows on the source set, once the H2b rows are computed - on the
    sentences EVALB scores for both configurations. An EVALB-error sentence
    adds nothing to its configuration's F1 already, so the main Delta compares
    two F1 values over different sentences; here the union of the two
    configurations' EVALB-error sentences is left out of both. Descriptive: no
    class and no verdict.
    """
    plan = run.plan
    subset = plan.statistics.subset_sensitivity
    if subset is None:
        return [], []
    if subset != EVALB_BOTH_PARSERS:
        raise AnalysisError(f"statistics.subset_sensitivity {subset!r} is not computed here")
    evalb = [task for task in TASKS if EVALB_ERROR in COUNT_COLUMNS[task]]
    rows: list[SubsetRow] = []
    skipped: list[str] = []

    def add(part, row_id, task, dataset, higher, lower, main_estimate):
        name = plan.tasks[task].metric.name
        scored = run.scored_by_both(task, dataset, higher, lower)
        if not scored.any():
            skipped.append(f"{row_id}: no sentence EVALB scores for both {lower} and {higher}")
            return
        try:
            estimate = run.subset_contrast(task, dataset, higher, lower, name, scored)
        except ValueError as exc:
            skipped.append(f"{row_id}: {exc}")
            return
        # The set column and the status are the part's: an h2b row's term is
        # computed on the source set, by its unit, and belongs to the transition.
        rows.append(SubsetRow(
            part.hypothesis, row_id, part.dataset, part.status, run.unit(dataset, MAIN),
            int(scored.sum()), int(scored.size), estimate, main_estimate,
        ))

    for part in main["h1a"]:
        if part.estimate is None or part.fields["task"] not in evalb:
            continue
        fields = part.fields
        add(part, part.row_id, fields["task"], fields["dataset"], fields["higher"],
            fields["lower"], part.estimate)
    for task in evalb:
        transition = plan.h2b.get(task)
        computed = [part for part in main["h2b"]
                    if part.estimate is not None and part.fields["task"] == task]
        if transition is None or not computed:
            continue
        step = run.step_max(task, transition.source, MAIN)
        add(computed[0],
            f"{task}/{transition.source}->{transition.target}/step_max/{step.lower}->{step.higher}",
            task, transition.source, step.higher, step.lower, step.estimate)
    return rows, skipped


def _file_name(task: str, dataset: str) -> str:
    return f"counts_{task}_{dataset}.csv"


def _awaiting(run: Run, parts: list[Part]) -> str:
    """What a group of parts waits for: the confirmatory run, or files this run lacks."""
    missing = list(dict.fromkeys(
        need for part in parts if part.estimate is None for need in part.needs
        if need not in run.files
    ))
    if not missing:
        return ""
    confirmatory = [_file_name(*need) for need in missing if run.status(need[1]) == "confirmatory"]
    exploratory = [_file_name(*need) for need in missing if run.status(need[1]) == "exploratory"]
    notes = []
    if confirmatory:
        notes.append(f"awaiting the confirmatory run ({', '.join(confirmatory)})")
    if exploratory:
        notes.append(f"not in this run: {', '.join(exploratory)}")
    return "; ".join(notes)


def _part_name(hypothesis: str) -> str:
    """h2b -> H2b."""
    return hypothesis[:-1].upper() + hypothesis[-1]


def verdict_rows(run: Run, main: dict[str, list[Part]]) -> list[dict]:
    """One row per hypothesis part and status, then per whole hypothesis and status.

    thesis_verdicts.csv. A whole hypothesis (WHOLE) is decided, for a status,
    from the parts the plan defines for that status: it counts the parts of
    every component that has parts of the status, holds only if every one of
    them holds, and is decided once all are computed. A component with no
    part of the status (H2a confirmatory, H2b exploratory) does not enter it,
    and the note says so. The rule of 24 Oct 2026, stated in the plan's
    comment on the hypotheses; until then a whole hypothesis with such a
    component was left undecided.
    """
    plan = run.plan
    rows = []
    for hypothesis in HYPOTHESES:
        for status in STATUSES:
            group = [part for part in main[hypothesis] if part.status == status]
            if not group:
                continue
            done = [part for part in group if part.estimate is not None]
            holding = [part for part in done if decide(part.rule, part.estimate, run.delta)]
            notes = [_awaiting(run, group)] if len(done) < len(group) else []
            if hypothesis == "h1b" and status != run.status(plan.h1b.verdict):
                notes.append(f"not the verdict: the plan gives it to {plan.h1b.verdict}")
            rows.append({
                "hypothesis": hypothesis, "status": status, "parts": len(group),
                "parts_holding": len(holding) if done else None,
                "holds": len(holding) == len(group) if len(done) == len(group) else None,
                "note": "; ".join(notes),
            })
    for whole, components in WHOLE.items():
        for status in STATUSES:
            by_component = {
                hypothesis: [part for part in main[hypothesis] if part.status == status]
                for hypothesis in components
            }
            group = [part for parts in by_component.values() for part in parts]
            if not group:
                continue
            done = [part for part in group if part.estimate is not None]
            holding = [part for part in done if decide(part.rule, part.estimate, run.delta)]
            absent = [_part_name(h) for h, parts in by_component.items() if not parts]
            entering = [_part_name(h) for h, parts in by_component.items() if parts]
            notes = []
            if absent:
                one = len(absent) == 1
                notes.append(
                    f"{', '.join(absent)} {'has' if one else 'have'} no {status} part and "
                    f"{'does' if one else 'do'} not enter it: {whole.upper()} on the {status} "
                    f"sets is {' + '.join(entering)}"
                )
            if len(done) < len(group):
                notes.append(_awaiting(run, group))
            if by_component.get("h1b") and status != run.status(plan.h1b.verdict):
                notes.append(f"its H1b part is not the verdict of H1b: the plan gives that to "
                             f"{plan.h1b.verdict}")
            rows.append({
                "hypothesis": whole, "status": status, "parts": len(group),
                "parts_holding": len(holding) if done else None,
                "holds": len(holding) == len(group) if len(done) == len(group) else None,
                "note": "; ".join(notes),
            })
    return rows


# --- formatting --------------------------------------------------------------


def csv_cell(value: Any) -> str:
    """A value as the thesis tables write it: 4 decimals, yes/no, empty for not available."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        value = float(value)
        if not math.isfinite(value):
            return ""
        text = f"{value:.4f}"
        return "0.0000" if text == "-0.0000" else text
    return str(value)


def write_csv(path: Path, header: tuple[str, ...], rows: list[dict]) -> None:
    """UTF-8, LF, one header row; a table with no rows is still written."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            if set(row) != set(header):
                raise RuntimeError(
                    f"{path.name}: a row with columns {sorted(row)} for the header {list(header)}"
                )
            writer.writerow([csv_cell(row[column]) for column in header])


def _num(value: Any, digits: int = 2) -> str:
    """A value as the report prints it: 2 decimals, n/a for not available."""
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    value = float(value)
    if not math.isfinite(value):
        return "n/a"
    text = f"{value:.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def _interval(low: float, high: float) -> str:
    return f"[{_num(low)}, {_num(high)}]"


class Report:
    """The text of analysis.txt, in the layout of the project's stage reports."""

    WIDTH = 90

    def __init__(self) -> None:
        self.lines: list[str] = []

    def banner(self, *lines: str) -> None:
        self.lines += ["=" * self.WIDTH, *(f"  {line}" for line in lines), "=" * self.WIDTH]

    def heading(self, title: str) -> None:
        self.lines += ["", "-" * self.WIDTH, f"  {title}", "-" * self.WIDTH]

    def text(self, *lines: str) -> None:
        self.lines += [f"  {line}" if line else "" for line in lines]

    def table(self, header: tuple[str, ...], rows: list[tuple]) -> None:
        cells = [[str(cell) for cell in header]] + [[str(cell) for cell in row] for row in rows]
        widths = [max(len(row[i]) for row in cells) for i in range(len(header))]

        def line(row: list[str]) -> str:
            return ("  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths))).rstrip()

        self.lines += [line(cells[0]), "  " + "  ".join("-" * width for width in widths)]
        self.lines += [line(row) for row in cells[1:]]


@dataclass
class Analysis:
    """Everything analyze() computed: the thesis tables, the report and their provenance.

    `provenance` is the record write() completes with the digest of every
    table and writes as csv/thesis.prov.json.
    """

    tables: dict[str, tuple[tuple[str, ...], list[dict]]]
    report: list[str]
    warnings: list[str]
    provenance: dict[str, Any] = field(default_factory=dict)


# --- the report ----------------------------------------------------------------


def _parts(n: int) -> str:
    return f"{n} part" if n == 1 else f"{n} parts"


def _verdict_line(hypothesis: str, status: str, verdicts: list[dict]) -> str:
    row = next((r for r in verdicts if r["hypothesis"] == hypothesis and r["status"] == status), None)
    if row is None:
        return ""
    label = f"verdict {hypothesis.upper()} [{status}]: "
    if row["holds"] is None:
        verdict = "not decided"
    else:
        verdict = "holds" if row["holds"] else "does not hold"
    counted = (
        f" - {row['parts_holding']} of {_parts(row['parts'])} hold"
        if row["parts_holding"] is not None else f" - {_parts(row['parts'])}"
    )
    note = f"; {row['note']}" if row["note"] else ""
    return label + verdict + counted + note


def _report_hypotheses(report: Report, run: Run, status: str, main: dict[str, list[Part]],
                       tables: dict, verdicts: list[dict]) -> None:
    delta = run.delta
    rows_of = {h: [r for r in tables[f"thesis_{h}"][1] if r["status"] == status] for h in HYPOTHESES}
    titles = {
        "h1a": "H1a - each step of an H1a chain: Delta = M(higher) - M(lower); holds when ci_low > 0",
        "h1b": "H1b - the step to the transformer: Delta_LAS - Delta_UPOS from the same replicate; "
               "holds when ci_low > 0",
        "h2a": "H2a - training data: |Delta_source| - Delta_step,max from the same replicate; "
               "holds when ci_low > 0",
        "h2b": "H2b - test data: (M_in - M_out) - Delta_step,max, the step from the in-domain "
               "replicate; holds when ci_low > 0",
        "h2c": "H2c - related data: Delta = M(treatment) - M(baseline); equivalence holds when "
               "-delta < ci_low and ci_high < +delta, superiority when ci_low > +delta",
    }
    legs = _h1b_legs(run.plan)

    for hypothesis in HYPOTHESES:
        group = [part for part in main[hypothesis] if part.status == status]
        if not group:
            continue
        report.heading(f"{titles[hypothesis]} [{status}]")
        rows = rows_of[hypothesis]
        if rows:
            if hypothesis == "h1a":
                report.table(
                    ("task", "chain", "dataset", "step", "M lower", "M higher", "Delta", "95 % CI",
                     "half-width", "class", "holds"),
                    [(r["task"], r["chain"], r["dataset"],
                      f"{r['lower']} ({r['lower_level']}) -> {r['higher']} ({r['higher_level']})",
                      _num(r["m_lower"]), _num(r["m_higher"]), _num(r["delta"]),
                      _interval(r["ci_low"], r["ci_high"]), _num(r["half_width"]), r["class"],
                      _num(r["holds"])) for r in rows],
                )
            elif hypothesis == "h1b":
                stems = [_leg_stem(leg) for leg in legs]
                report.table(
                    ("dataset", "verdict set", "step", *(f"Delta {leg.metric} [95 % CI]" for leg in legs),
                     "diff", "95 % CI", "half-width", "class", "holds"),
                    [(r["dataset"], _num(r["is_verdict"]), f"{r['baseline']} -> {r['treatment']}",
                      *(f"{_num(r[s])} {_interval(r[s + '_ci_low'], r[s + '_ci_high'])}" for s in stems),
                      _num(r["diff"]), _interval(r["ci_low"], r["ci_high"]), _num(r["half_width"]),
                      r["class"], _num(r["holds"])) for r in rows],
                )
            elif hypothesis == "h2a":
                report.table(
                    ("task", "pair", "dataset", "baseline -> treatment", "Delta_source [95 % CI]",
                     "Delta_step,max (step)", "|D_source| - step", "95 % CI", "half-width", "class",
                     "holds"),
                    [(r["task"], r["pair"], r["dataset"], f"{r['baseline']} -> {r['treatment']}",
                      f"{_num(r['delta_source'])} {_interval(r['delta_source_ci_low'], r['delta_source_ci_high'])}",
                      f"{_num(r['step_max'])} ({r['step_lower']} -> {r['step_higher']})",
                      _num(r["stat"]), _interval(r["ci_low"], r["ci_high"]), _num(r["half_width"]),
                      r["class"], _num(r["holds"])) for r in rows],
                )
            elif hypothesis == "h2b":
                report.table(
                    ("task", "config", "source -> target", "M in", "M out", "drop [95 % CI]",
                     "Delta_step,max (step)", "drop - step", "95 % CI", "half-width", "class",
                     "holds"),
                    [(r["task"], r["config"], f"{r['source']} -> {r['target']}", _num(r["m_in"]),
                      _num(r["m_out"]), f"{_num(r['drop'])} {_interval(r['drop_ci_low'], r['drop_ci_high'])}",
                      f"{_num(r['step_max'])} ({r['step_lower']} -> {r['step_higher']})",
                      _num(r["stat"]), _interval(r["ci_low"], r["ci_high"]), _num(r["half_width"]),
                      r["class"], _num(r["holds"])) for r in rows],
                )
            else:
                report.table(
                    ("task", "dataset", "decision", "baseline -> treatment", "Delta", "95 % CI",
                     "half-width", "class", "holds"),
                    [(r["task"], r["dataset"], r["decision"], f"{r['baseline']} -> {r['treatment']}",
                      _num(r["delta"]), _interval(r["ci_low"], r["ci_high"]), _num(r["half_width"]),
                      r["class"], _num(r["holds"])) for r in rows],
                )
            report.text("")
        report.text(_verdict_line(hypothesis, status, verdicts))
    lines = [_verdict_line(whole, status, verdicts) for whole in WHOLE]
    if any(lines):
        report.heading(
            f"H1 and H2 - a hypothesis holds only if every part of it holds "
            f"({'; '.join(f'{w.upper()}: ' + ', '.join(map(_part_name, c)) for w, c in WHOLE.items())}) "
            f"[{status}]"
        )
        report.text(*(line for line in lines if line))
    report.text("", f"delta = {_num(delta, 1)} point; class: where the 95 % CI lies relative to 0 and +-delta.")


def _report_descriptive(report: Report, run: Run, status: str, tables: dict, skipped: list[str]) -> None:
    def mine(name: str) -> list[dict]:
        return [row for row in tables[name][1] if row["status"] == status]

    coverage, f1 = mine("thesis_d2_coverage"), mine("thesis_d2_f1")
    if coverage:
        report.heading(f"D2 - PCFG coverage by failure reason, and F1 on the sentences the PCFG parsed [{status}]")
        report.table(
            ("dataset", "config", "sampled", "parsed", *FAILURES, "coverage"),
            [(r["dataset"], r["config"], r["sampled"], r["parsed"], *(r[f] for f in FAILURES),
              _num(r["coverage"])) for r in coverage],
        )
        report.text("")
        report.table(
            ("dataset", "PCFG", "config", "sentences", "F1"),
            [(r["dataset"], r["pcfg_config"], r["config"], r["sentences"], _num(r["f1"])) for r in f1],
        )
        report.text("", "Descriptive: no CI, no class. Coverage = parsed / sampled, in points.")
    d3 = mine("thesis_d3")
    if d3:
        report.heading(f"D3 - annotation scheme: other - {run.plan.d3.config}, descriptive [{status}]")
        report.table(
            ("dataset", "metric", "other", f"M {run.plan.d3.config}", "M other", "Delta", "95 % CI", "half-width"),
            [(r["dataset"], r["metric"], r["other"], _num(r["m_baseline"]), _num(r["m_other"]),
              _num(r["delta"]), _interval(r["ci_low"], r["ci_high"]), _num(r["half_width"])) for r in d3],
        )
    d4 = mine("thesis_d4")
    if d4:
        report.heading(f"D4 - {run.plan.d4.baseline} -> {run.plan.d4.treatment}, descriptive [{status}]")
        report.table(
            ("dataset", "M baseline", "M treatment", "Delta", "95 % CI", "half-width"),
            [(r["dataset"], _num(r["m_baseline"]), _num(r["m_treatment"]), _num(r["delta"]),
              _interval(r["ci_low"], r["ci_high"]), _num(r["half_width"])) for r in d4],
        )
    drops = mine("thesis_descriptive")
    if drops or (status == "confirmatory" and skipped):
        report.heading(f"Descriptive drops: the members whose only role is descriptive [{status}]")
        if drops:
            report.table(
                ("task", "config", "source -> target", "M in", "M out", "drop", "95 % CI", "half-width"),
                [(r["task"], r["config"], f"{r['source']} -> {r['target']}", _num(r["m_in"]),
                  _num(r["m_out"]), _num(r["drop"]), _interval(r["ci_low"], r["ci_high"]),
                  _num(r["half_width"])) for r in drops],
            )
        for line in skipped:
            report.text(f"not computed: {line}")


def _report_scores(report: Report, status: str, tables: dict) -> None:
    rows = [row for row in tables["thesis_scores"][1] if row["status"] == status]
    if not rows:
        return
    report.heading(f"Scores: every configuration, point values [{status}]")
    report.table(
        ("task", "dataset", "config", "level", "roles", "metric", "value", "UAS", "CLAS",
         "sentences", "units", "parsed", "EVALB errors"),
        [(r["task"], r["dataset"], r["config"], r["level"], r["roles"], r["metric"], _num(r["value"]),
          _num(r["uas"]) if r["uas"] is not None else "", _num(r["clas"]) if r["clas"] is not None else "",
          r["sentences"], r["units"], "" if r["parsed"] is None else r["parsed"],
          "" if r["evalb_errors"] is None else r["evalb_errors"]) for r in rows],
    )


def _report_subset(report: Report, subset: str, rows: list[SubsetRow], skipped: list[str]) -> None:
    """The short table of statistics.subset_sensitivity, at the end of SENSITIVITY."""
    report.text("", f"On the sentences EVALB scores for both parsers (statistics.subset_sensitivity "
                    f"{subset}), descriptive:")
    if rows:
        report.table(
            ("hypothesis", "row", "status", "unit", "sentences", "Delta", "95 % CI", "half-width",
             "Delta, all sentences"),
            [(r.hypothesis, r.row_id, r.status, r.unit, f"{r.sentences} of {r.of}",
              _num(r.estimate.point), _interval(r.estimate.ci_low, r.estimate.ci_high),
              _num(r.estimate.half_width), _num(r.main.point)) for r in rows],
        )
    for line in skipped:
        report.text(f"not computed: {line}")
    if not rows and not skipped:
        report.text("nothing to compute in this run: no Delta between two parsers scored by EVALB")
    report.text(
        "",
        "No class and no verdict: this changes none. An EVALB-error sentence adds nothing to its",
        "parser's F1 (0 matched, gold and test), so the main Delta ('Delta, all sentences', the",
        "row of the hypothesis table) compares two F1 values over the sentences each parser alone",
        "was scored on; here the sentences either parser could not be scored on are left out for",
        "both. The same unit and index matrix as the main estimate: a filter on the counts, not a",
        "new draw. An h2b row is the Delta_step,max term of the H2b rows, on their source set.",
        f"csv/thesis_sensitivity.csv has each as the row <row>/{subset}=<sentences>,",
        "with no delta, class or holds.",
    )


def measured_groups(run: Run) -> list[tuple[MeasuredUnder, list[CountsFile]]]:
    """The plans the counts of a run were measured under, each with its files, in file order."""
    groups: dict[MeasuredUnder, list[CountsFile]] = {}
    for file in run.files.values():
        groups.setdefault(file.measured, []).append(file)
    return list(groups.items())


def _measured_lines(run: Run) -> list[str]:
    """The banner lines that say which plan measured the counts."""
    groups = measured_groups(run)
    lines = []
    for measured, files in groups:
        which = "every counts file" if len(groups) == 1 else ", ".join(file.name for file in files)
        if measured.same:
            lines.append(f"measured   : under the plan above ({which})")
            continue
        lines += [
            f"measured   : under {measured.path}  sha256 {measured.sha256}",
            f"             ({which}): an earlier plan, read from git at commit {measured.commit[:12]},",
            "             equal to the plan above in every part that determines the counts",
        ]
    return lines


def tables_provenance(run: Run, run_dir: Path) -> dict[str, Any]:
    """csv/thesis.prov.json without the tables' digests, which write() adds."""
    plan = run.plan
    parts = ", ".join(measurement_parts(plan))
    return {
        "schema": PROVENANCE_SCHEMA,
        "run_dir": portable(run_dir),
        "plan": {"path": portable(plan.path), "sha256": plan.sha256},
        "measured": [
            {
                "plan": {"path": measured.path, "sha256": measured.sha256},
                "commit": measured.commit,
                "counts": [f"csv/{file.name}" for file in files],
                "relation": (
                    "the plan analysed" if measured.same else
                    "an earlier plan, read from git at the commit with the sha256 recorded, "
                    f"equal to the plan analysed in {parts}"
                ),
            }
            for measured, files in measured_groups(run)
        ],
        "counts": {f"csv/{file.name}": file.sha256 for file in run.files.values()},
    }


def build_report(run: Run, run_dir: Path, warnings: list[str], main: dict[str, list[Part]],
                 tables: dict, skipped: list[str], sensitivity: list[dict],
                 subset: list[SubsetRow], subset_skipped: list[str]) -> list[str]:
    plan, statistics = run.plan, run.plan.statistics
    bootstrap = statistics.bootstrap
    report = Report()
    report.banner(
        "STAGE analyze: VERSION 1.0 ANALYSIS - READS THE COUNTS, RUNS NO ANALYZER",
        f"run dir    : {recorded_path(run_dir)}",
        f"plan       : {recorded_path(plan.path)}  sha256 {plan.sha256}  (analysed under it)",
        *_measured_lines(run),
        f"bootstrap  : {bootstrap.method}, B = {bootstrap.replicates}, "
        f"{_num(100 * bootstrap.confidence, 0)} % CI (numpy's linear quantile), seed {bootstrap.seed}",
        f"delta      : {_num(statistics.delta, 1)} point; sensitivity "
        f"{', '.join(_num(d, 1) for d in statistics.delta_sensitivity)} and the unit "
        f"{statistics.unit_sensitivity}",
    )
    report.text(
        "Each set is resampled by its plan unit, with replacement, by one matrix of indices that",
        "every configuration on it shares; its stream is SeedSequence(seed, spawn_key=(crc32 of",
        "the dataset id, crc32 of the unit)), so one set always gets the same indices and two",
        "sets independent ones. Numbers are points, rounded to 2 decimals here and to 4 in",
        "csv/thesis_*.csv. Every class and verdict is computed at full precision from the CI and",
        "delta of its own row, and the intervals are the whole of the inference.",
    )

    report.heading("INPUTS")
    report.table(
        ("file", "task", "dataset", "status", "sentences", "units", "configurations", "run", "sha256"),
        [(file.name, task, dataset, file.status, file.n_sentences,
          f"{run.n_units(dataset)} {run.unit(dataset, MAIN)}", len(file.configs),
          "full" if not file.smoke else f"smoke, max_sentences {file.max_sentences}",
          file.sha256[:12]) for (task, dataset), file in run.files.items()],
    )
    absent = [
        (task, dataset) for task in TASKS for dataset in datasets_of(plan, task)
        if (task, dataset) not in run.files
    ]
    if absent:
        report.text("")
        report.table(
            ("not in this run", "status", ""),
            [(_file_name(task, dataset), run.status(dataset),
              "awaiting the confirmatory run" if run.status(dataset) == "confirmatory" else "missing")
             for task, dataset in absent],
        )
    if warnings:
        report.heading("WARNINGS")
        for warning in warnings:
            report.text(f"* {warning}")

    verdicts = tables["thesis_verdicts"][1]
    for status in STATUSES:
        sets = [key for key, dataset in plan.datasets.items() if dataset.status == status]
        present = [dataset for (_, dataset) in run.files if dataset in sets]
        report.lines += ["", "#" * Report.WIDTH, f"  {status.upper()} RESULTS: {', '.join(sets)}",
                         "#" * Report.WIDTH]
        if not present:
            report.text(
                "",
                f"Awaiting the {status} run: no counts of {', '.join(sets)} in this run.",
                "The parts that will be computed on them are listed under VERDICTS.",
            )
            continue
        _report_scores(report, status, tables)
        _report_hypotheses(report, run, status, main, tables, verdicts)
        _report_descriptive(report, run, status, tables, skipped)

    report.heading("SENSITIVITY - every hypothesis row at each delta, and with the sentence as unit")
    report.table(
        ("hypothesis", "row", "status", "unit", "delta", "estimate", "95 % CI", "class", "holds"),
        [(r["hypothesis"], r["row"], r["status"], r["unit"], r["delta_threshold"], _num(r["estimate"]),
          _interval(r["ci_low"], r["ci_high"]), r["class"], _num(r["holds"]))
         for r in sensitivity],
    )
    report.text(
        "",
        "A row whose plan unit is already the sentence (MCTB_en) has no separate sentence row.",
    )
    if statistics.subset_sensitivity is not None:
        _report_subset(report, statistics.subset_sensitivity, subset, subset_skipped)

    report.heading("VERDICTS - a hypothesis holds only if every part of it holds, per status")
    report.table(
        ("hypothesis", "status", "parts", "holding", "holds", "note"),
        [(r["hypothesis"], r["status"], r["parts"], _num(r["parts_holding"]) if r["parts_holding"] is not None else "",
          "n/a" if r["holds"] is None else _num(r["holds"]), r["note"]) for r in verdicts],
    )
    report.text(
        "",
        "Whole hypotheses: "
        + "; ".join(f"{whole} = {' + '.join(parts)}" for whole, parts in WHOLE.items())
        + " (per status, every part the plan defines for that status; a component with",
        "no part of the status does not enter it, and the note says so).",
    )
    waiting = [r for r in verdicts if "awaiting the confirmatory run" in r["note"]]
    report.heading("AWAITING THE CONFIRMATORY RUN")
    if waiting:
        for r in waiting:
            report.text(f"{r['hypothesis'].upper()} [{r['status']}], {_parts(r['parts'])}: {r['note']}")
    else:
        report.text("nothing: every part of every hypothesis is in this run")
    return report.lines + [""]


# --- the analysis --------------------------------------------------------------


def analyze(run_dir: Path, plan: Plan, *, root: Path | None = None) -> Analysis:
    """Read a run, compute every table and the report. Writes nothing.

    `root` is where git reads the plan of counts measured under an earlier
    plan from (read_run): the project root unless given.

    Raises:
        AnalysisError: If the run is refused (see the module docstring).
    """
    files, warnings = read_run(run_dir, plan, root=root)
    run = Run(plan, files)
    main = {hypothesis: list(PARTS[hypothesis](run, MAIN)) for hypothesis in HYPOTHESES}
    alternative = {hypothesis: list(PARTS[hypothesis](run, ALTERNATIVE)) for hypothesis in HYPOTHESES}
    headers = hypothesis_headers(plan)
    tables: dict[str, tuple[tuple[str, ...], list[dict]]] = {
        "thesis_scores": (SCORES_HEADER, score_rows(run)),
    }
    for hypothesis in HYPOTHESES:
        header = headers[hypothesis]
        tables[f"thesis_{hypothesis}"] = (header, [
            hypothesis_row(part, header, run.delta)
            for part in main[hypothesis] if part.estimate is not None
        ])
    coverage, f1 = d2_rows(run)
    drops, skipped = descriptive_rows(run)
    sensitivity = sensitivity_rows(run, main, alternative)
    subset, subset_skipped = subset_rows(run, main)
    subset_name = plan.statistics.subset_sensitivity
    tables.update({
        "thesis_d2_coverage": (D2_COVERAGE_HEADER, coverage),
        "thesis_d2_f1": (D2_F1_HEADER, f1),
        "thesis_d3": (D3_HEADER, d3_rows(run)),
        "thesis_d4": (D4_HEADER, d4_rows(run)),
        "thesis_descriptive": (DESCRIPTIVE_HEADER, drops),
        # The rows of statistics.subset_sensitivity after every other row.
        "thesis_sensitivity": (
            SENSITIVITY_HEADER, sensitivity + [row.csv_row(subset_name) for row in subset],
        ),
        "thesis_verdicts": (VERDICTS_HEADER, verdict_rows(run, main)),
    })
    report = build_report(run, run_dir, warnings, main, tables, skipped, sensitivity, subset,
                          subset_skipped)
    return Analysis(tables=tables, report=report, warnings=warnings,
                    provenance=tables_provenance(run, run_dir))


def write(run_dir: Path, analysis: Analysis) -> list[Path]:
    """Write the thesis tables into <run>/csv, their provenance beside them
    (csv/thesis.prov.json, with the digest of every table) and the report to
    <run>/analysis.txt."""
    written = []
    digests = {}
    for name, (header, rows) in analysis.tables.items():
        path = run_dir / "csv" / f"{name}.csv"
        write_csv(path, header, rows)
        written.append(path)
        digests[f"csv/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = run_dir / "csv" / TABLES_PROVENANCE
    record = {**analysis.provenance, "tables": digests}
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                    newline="\n")
    written.append(path)
    path = run_dir / "analysis.txt"
    path.write_text("\n".join(analysis.report), encoding="utf-8", newline="\n")
    written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="the run: reads RUN_DIR/csv/counts_*.csv, writes RUN_DIR/csv/thesis_*.csv "
                             "and RUN_DIR/analysis.txt")
    parser.add_argument("--plan", type=Path, default=None,
                        help="the plan the counts were written under; default plan/v1.yaml")
    args = parser.parse_args(argv)
    try:
        plan = load(args.plan, strict=True)
    except (PlanError, OSError) as exc:
        print(f"analyze: no plan to analyse under: {exc}", file=sys.stderr)
        return 2
    try:
        analysis = analyze(args.run_dir, plan)
    except AnalysisError as exc:
        print(f"analyze: refused, nothing written: {exc}", file=sys.stderr)
        return 2
    write(args.run_dir, analysis)
    print("\n".join(analysis.report))
    print(
        f"written: {recorded_path(args.run_dir / 'analysis.txt')} and {len(analysis.tables)} "
        f"tables in {recorded_path(args.run_dir / 'csv')} (thesis_*.csv) with their provenance "
        f"{TABLES_PROVENANCE}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
