"""The experiment as data: plan/v1.yaml, read and checked before anything runs.

Version 1.0 states its design once, in plan/v1.yaml: the configurations (wrapper,
constructor and load_model arguments, training data), the method level and the
roles of each in each task, the H1a chains, the H1b step, the H2a pairs and the
datasets they are computed on, the H2b transitions, the H2c contrast, the three
descriptive sections D2-D4, the datasets with their status and resampling
unit, and the statistical procedure.
Added 23 Oct 2026, and revised the same day twice: when the brief split H1 into
H1a and H1b and added D4, and when the plan dropped D1 (UDPipe 1 with the
published UD 2.5 model against UDPipe 1 retrained on EWT; the labels D2-D4 were
kept) and limited H2b to the models of the H1a chains. Until then the design
was not written down anywhere as such:
evaluation/design.py derived the contrasts from the axes each analyzer
declares, and each stage script chose which keys it registered. A design
stated once can be read by its author, checked by a program and cited by the
thesis, and code that reads it has nothing of its own to drift from.

A plan that is wrong in a way a run would reveal only hours later is refused
here, in about a second. load() reads the file in phases and collects every
problem of a phase before it raises, so one check lists them all:

  YAML       a strict loader. A duplicate key, or a key YAML 1.1 reads as
             something other than a string (`on:` is True), is an error rather
             than an entry silently dropped or renamed.
  structure  every mapping at every level has a fixed set of required and
             optional keys. An unknown key is refused, never ignored: an
             ignored key is a setting its author believes is in force. Types,
             identifiers and method levels are checked here too.
  meaning    references resolve; wrappers import and their signatures accept
             the arguments the plan passes; the hypotheses are well formed
             (rising levels, one training corpus per chain and per step, one
             varied resource per pair, datasets of H2a that list the tasks of
             its pairs, added 24 Oct 2026); the roles each task declares
             equal the roles the hypotheses imply; and every confirmatory
             dataset has a source pinned to a tag or a commit (added 23 Oct
             2026, with scripts/fetch_data.py, which downloads it); no two
             datasets' paths are the same or one inside the other; every constituency
             dataset, and no other, declares how it writes brackets and
             double quotes (`tokens`, added 24 Oct 2026); and a configuration
             restricted to some datasets (`datasets`, added 24 Oct 2026) is
             restricted to datasets of its tasks, keeps one of each task,
             and is not left out of a dataset a section needs it on
             (_Meaning._restrictions); statistics.subset_sensitivity, if
             given, names a known subset and the plan has a Delta it
             applies to (added 24 Oct 2026).
  strict     load(strict=True) only: what a run uses. Every pending
             configuration that is not optional is an error, and the optional
             ones are dropped from the plan it returns.

A configuration is scored on every dataset of its tasks unless the plan
restricts it (configurations.<key>.datasets; Plan.scored_on, Plan.members_on):
the owner's decision of 24 Oct 2026 scores nltk_pre and spacy_sm on EWT test
only. D3 and D4 are computed on the datasets of their task where every one of
their members is scored (Plan.d3_datasets, Plan.d4_datasets), so D3 is
computed on EWT test only.

A configuration the code cannot run yet is declared `pending`, and its
missing_params lists the constructor arguments its wrapper does not accept. The
list is checked both ways - a name the wrapper has since learned to accept is a
stale entry, and an unaccepted name that is not listed is an error - so it
always says exactly what is missing.

Cost: the meaning phase imports the wrapper modules the plan names (about
1.5 s, almost all of it NLTK itself, which nltk_transition imports at module
level for its transition parser), but no model library: no wrapper module
imports one when it is imported. Nor does it import syntactic_analysis.evaluation,
which would cost another 1.4 s.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import pkgutil
import re
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import yaml

from syntactic_analysis.base import BaseAnalyzer, MethodLevel
from syntactic_analysis.project import project_root
from syntactic_analysis.ptb_conventions import BRACKET_CONVENTIONS, QUOTE_CONVENTIONS

__all__ = [
    "TASKS",
    "ROLES",
    "PlanError",
    "default_path",
    "load",
    "plan_sha256",
    "Plan",
    "TrainingData",
    "Dataset",
    "Source",
    "Tokens",
    "Resampling",
    "Pending",
    "Configuration",
    "Member",
    "Metric",
    "Task",
    "Chain",
    "Leg",
    "H1b",
    "Pair",
    "Transition",
    "Check",
    "H2c",
    "D2",
    "Sample",
    "D3",
    "D4",
    "Statistics",
    "Bootstrap",
]

# The task codes, in reading order. Until 23 Oct 2026 evaluation/axes.py
# declared them too, for the stages before version 1.0, and a test kept the
# two equal.
TASKS: tuple[str, ...] = ("pos", "dep", "con")

# The roles a configuration can have in a task. Every one but `descriptive` is
# implied by a hypothesis or descriptive section of the plan, and the roles a
# task declares must equal the implied ones; `descriptive` marks a member that
# is scored and reported but enters no comparison. There is no d1: D1 was
# dropped on 23 Oct 2026 and its label is not reused.
ROLES: tuple[str, ...] = (
    "h1a", "h1b", "h2a", "h2b", "h2c", "d2", "d3", "d4", "descriptive",
)

# The one schema this module reads. A plan written for another one is refused
# rather than read by rules that were not written for it.
_SCHEMA = 1

# Keys, ids and parameter names: lower case, digits and underscores, starting
# with a letter. That covers every Python parameter name the wrappers use, and
# keeps a key usable as a CSV value and a file name without quoting.
_ID = re.compile(r"[a-z][a-z0-9_]*")
_SECTION = re.compile(r"[0-9]{2}")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")

_LEVELS: Mapping[str, MethodLevel] = MappingProxyType(
    {level.code: level for level in MethodLevel}
)
_STATUSES = ("exploratory", "confirmatory")
_ANNOTATIONS = ("ud", "ptb", "ontonotes")

# Where a dataset's source may be. scripts/fetch_data.py builds its download
# URLs as raw.githubusercontent.com/<owner>/<repo>/<ref>/<file>, so a source
# anywhere else could be stated but not fetched.
_GITHUB = re.compile(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
# Refs that name a branch rather than a fixed state. A source pinned to one
# downloads whatever the branch holds that day, which is the defect the EWT
# snapshot in data/ has (data/DATASET.md); a tag or a commit does not move.
_MOVING_REFS = ("HEAD", "main", "master")
_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# Resampling units and the unit ids each one can be read from. A sentence is
# its own unit, so it takes no unit_id.
_UNIT_IDS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "document": ("newdoc_id", "newdoc_id_else_sent_id_prefix"),
    "file": ("wsj_fileid",),
    "sentence": (),
})

# The constructor keywords whose value is another configuration, built first.
_INPUTS = ("pos_source",)

# One primary metric per task, fixed by the design: a plan cannot change how a
# task is scored, only state it.
_METRICS: Mapping[str, Mapping[str, str]] = MappingProxyType({
    "pos": MappingProxyType({"name": "upos_accuracy", "aggregation": "micro"}),
    "dep": MappingProxyType({
        "name": "las", "aggregation": "micro",
        "convention": "conll2018", "deprel_subtypes": "strip",
    }),
    "con": MappingProxyType({
        "name": "bracket_f1", "aggregation": "micro",
        "evalb": "collins", "nml": "collapse_both_sides",
    }),
})

_DECISIONS = ("equivalence", "superiority")
# How each decision reads a confidence interval. These are the two rules the
# design defines; a plan states them so a reader sees them beside delta, and a
# plan that states another is refused rather than computed with these.
_DECISION_RULES: Mapping[str, str] = MappingProxyType({
    "equivalence": "ci_within_plus_minus_delta",
    "superiority": "ci_lower_bound_above_delta",
})
_SCALES = ("points",)
_BOOTSTRAP_METHODS = ("paired_percentile",)
# The sentence subsets a Delta is also computed on, descriptively
# (statistics.subset_sensitivity, added 24 Oct 2026). evalb_both_parsers: every
# Delta between two configurations of a task scored by EVALB on one set - each
# H1a step of the task, and the Delta_step,max term of its H2b rows on the
# source set - on the sentences EVALB scores for both of them, the union of
# their EVALB-error sentences left out (scripts/analyze.py).
_SUBSET_SENSITIVITIES = ("evalb_both_parsers",)
_MIN_REPLICATES = 1000
# random.Random and numpy's default_rng take any non-negative integer, but
# numpy's legacy RandomState (np.random.seed) takes only 0..2**32-1; a seed in
# that range works with every generator a stage might use.
_SEED_MAX = 2**32 - 1

_D2_DRAWN_FROM = ("all_sentences",)
_D2_REPORTS = ("coverage_by_failure_reason", "f1_on_processed_sentences")
_D3_METRICS = ("las", "clas")


# --- the error ---------------------------------------------------------------


class PlanError(ValueError):
    """A plan that cannot be used, with every problem one phase found.

    Attributes:
        source: The plan file the problems are in.
        errors: One message per problem, in the order they were found.

    str() is a header and the problems as a numbered list, so a refused plan
    reads as a to-do list rather than as the first of an unknown number of
    errors.
    """

    def __init__(self, source: str | Path, errors: Iterable[str]) -> None:
        self.source = str(source)
        self.errors: tuple[str, ...] = tuple(errors)
        super().__init__(self.source, self.errors)

    def __str__(self) -> str:
        count = len(self.errors)
        lines = [f"{self.source}: {count} problem{'' if count == 1 else 's'}"]
        lines.extend(f"  {number}. {error}" for number, error in enumerate(self.errors, 1))
        return "\n".join(lines)


# --- the plan as frozen data -------------------------------------------------


@dataclass(frozen=True)
class TrainingData:
    """One corpus a model was trained on.

    `family` groups releases of one corpus; an H1a chain accepts another release
    of its own family only where it lists the member under approximations.
    """

    id: str
    name: str
    family: str
    annotation: str
    path: str | None = None


@dataclass(frozen=True)
class Resampling:
    """The unit the bootstrap resamples a dataset by, and where it is read from."""

    unit: str
    unit_id: str | None = None
    expected_units: int | None = None


@dataclass(frozen=True)
class Source:
    """Where a dataset is downloaded from: a repository, a fixed ref, its files.

    `files` are paths inside the repository at `ref`. scripts/fetch_data.py
    fetches them into the dataset's local copy (Dataset.files) and records
    their digests beside it (Dataset.digests).
    """

    repository: str
    ref: str
    files: tuple[str, ...]


@dataclass(frozen=True)
class Tokens:
    """How a constituency test set writes brackets and double quotes.

    `brackets` is one of ptb_conventions.BRACKET_CONVENTIONS and `quotes` one
    of ptb_conventions.QUOTE_CONVENTIONS. The constituency stage gives every
    configuration the sentence in the PTB convention (parser_tokens) and puts
    the test set's own words back before the tree is scored (restore_words).
    """

    brackets: str
    quotes: str


@dataclass(frozen=True)
class Dataset:
    """A test set: its tasks, its status and how it is resampled.

    Exactly one of `path` (relative to the project root) and `nltk_sections`
    (WSJ sections of NLTK's Penn Treebank sample) says where it comes from.
    `source`, which a confirmatory dataset must have, says where the local
    copy at `path` is downloaded from. `tokens`, which a constituency dataset
    must have and no other may, says how it writes brackets and quotes.
    """

    id: str
    name: str
    tasks: tuple[str, ...]
    status: str
    license: str
    resampling: Resampling
    path: str | None = None
    nltk_sections: tuple[str, ...] | None = None
    source: Source | None = None
    tokens: Tokens | None = None

    @property
    def files(self) -> tuple[str, ...]:
        """The local copy, as paths relative to the project root.

        One source file is stored at `path` itself; several are stored inside
        `path` as a directory, each under its own file name, so MCTB's
        data/MCTB_en/law.cleaned.txt becomes data/mctb_en/law.cleaned.txt. A
        path without a source is one file; a dataset read through NLTK has no
        file here.
        """
        if self.path is None:
            return ()
        if self.source is None or len(self.source.files) == 1:
            return (self.path,)
        return tuple(
            f"{self.path}/{PurePosixPath(file).name}" for file in self.source.files
        )

    @property
    def digests(self) -> str | None:
        """The file that records the SHA-256 of the local copy: `<path>.sha256`.

        In sha256sum format, with paths relative to its own directory, so
        `sha256sum -c` run there checks it: data/en_ewt-ud-test.conllu.sha256
        and data/en_gentle-ud-test.conllu.sha256 hold one line,
        data/mctb_en.sha256 one per domain file.
        """
        return None if self.path is None else f"{self.path}.sha256"


@dataclass(frozen=True)
class Pending:
    """Why a configuration cannot run yet, and which step makes it runnable."""

    reason: str
    missing_params: tuple[str, ...]
    step: str | None = None


@dataclass(frozen=True)
class Configuration:
    """One runnable (or pending) analyzer: wrapper, arguments, training data.

    `init` holds constructor keywords, `load` load_model keywords and `inputs`
    the constructor keywords whose value is another configuration's key.
    `datasets` is the plan's restriction of the datasets it is scored on
    (configurations.<key>.datasets, since 24 Oct 2026), None when it has none:
    then it is scored on every dataset of its tasks (Plan.scored_on).
    """

    key: str
    wrapper: str
    model: str
    training: str
    init: Mapping[str, Any]
    load: Mapping[str, Any]
    inputs: Mapping[str, str]
    optional: bool = False
    pending: Pending | None = None
    datasets: tuple[str, ...] | None = None

    @property
    def ready(self) -> bool:
        """True when nothing stands between this configuration and a run."""
        return self.pending is None

    @property
    def module(self) -> str:
        """The wrapper module, e.g. 'stanza' for 'stanza.StanzaAnalyzer'."""
        return self.wrapper.partition(".")[0]

    @property
    def class_name(self) -> str:
        """The wrapper class, e.g. 'StanzaAnalyzer' for 'stanza.StanzaAnalyzer'."""
        return self.wrapper.partition(".")[2]


@dataclass(frozen=True)
class Member:
    """A configuration's place in one task: its method level and its roles.

    The level is per task because one tool can sit at two: UDPipe 1 is L1 for
    POS and L2 for dependencies.
    """

    level: MethodLevel
    roles: frozenset[str]


@dataclass(frozen=True)
class Metric:
    """A task's primary metric; `options` holds the task-specific settings."""

    name: str
    aggregation: str
    options: Mapping[str, str]


@dataclass(frozen=True)
class Task:
    """One task: its metric and its members, in the order the plan lists them."""

    code: str
    metric: Metric
    members: Mapping[str, Member]


@dataclass(frozen=True)
class Chain:
    """An H1a chain: configurations of rising level on one training corpus.

    `approximations` maps a member trained on another release of the chain's
    corpus to that release, so the difference is stated rather than hidden.
    """

    id: str
    task: str
    training: str
    configs: tuple[str, ...]
    approximations: Mapping[str, str]


@dataclass(frozen=True)
class Leg:
    """One task of the H1b step, and the metric its Delta is measured in."""

    task: str
    metric: str


@dataclass(frozen=True)
class H1b:
    """H1b: one level step on one corpus, scored in two tasks on the same documents.

    The tested quantity is the Delta of the first leg minus the Delta of the
    second, both from the same bootstrap replicate; it holds when the lower
    bound of its CI lies above 0. It is computed on every dataset listed, and
    `verdict` names the confirmatory one whose result decides it.
    """

    baseline: str
    treatment: str
    legs: tuple[Leg, ...]
    datasets: tuple[str, ...]
    verdict: str


@dataclass(frozen=True)
class Pair:
    """An H2a pair: one algorithm at one level, trained on two corpora."""

    id: str
    task: str
    baseline: str
    treatment: str


@dataclass(frozen=True)
class Transition:
    """An H2b transition: the same configurations scored in and out of genre.

    `configs` are exactly the configurations of the task's H1a chains, since
    the criterion of each is read against Delta_step,max, the largest step of
    those chains.
    """

    task: str
    source: str
    target: str
    configs: tuple[str, ...]


@dataclass(frozen=True)
class Check:
    """One H2c test: a dataset and the decision its interval is read by."""

    dataset: str
    decision: str


@dataclass(frozen=True)
class H2c:
    """The H2c contrast of one task: related data against the base package."""

    task: str
    baseline: str
    treatment: str
    tests: tuple[Check, ...]


@dataclass(frozen=True)
class Sample:
    """The sentences D2 parses: how many, from where, with which seed and cap."""

    sentences: int
    seed: int
    max_tokens: int
    drawn_from: str


@dataclass(frozen=True)
class D2:
    """Descriptive D2: PCFG coverage, and F1 against neural parsers."""

    task: str
    configs: tuple[str, ...]
    versus: tuple[str, ...]
    datasets: tuple[str, ...]
    sample: Sample
    report: tuple[str, ...]


@dataclass(frozen=True)
class D3:
    """Descriptive D3: a non-UD-trained parser against the UD-trained ones."""

    task: str
    config: str
    versus: tuple[str, ...]
    metrics: tuple[str, ...]


@dataclass(frozen=True)
class D4:
    """Descriptive D4: a level step on one corpus that no H1a chain holds."""

    task: str
    baseline: str
    treatment: str


@dataclass(frozen=True)
class Bootstrap:
    """The paired bootstrap: method, number of replicates, level, seed."""

    method: str
    replicates: int
    confidence: float
    seed: int


@dataclass(frozen=True)
class Statistics:
    """The one statistical procedure: delta, its sensitivity values, the bootstrap.

    `subset_sensitivity` names a sentence subset every Delta of its kind is
    also computed on, descriptively, with no class and no verdict
    (_SUBSET_SENSITIVITIES; since 24 Oct 2026 evalb_both_parsers), or is None.
    """

    scale: str
    delta: float
    delta_sensitivity: tuple[float, ...]
    bootstrap: Bootstrap
    unit_sensitivity: str
    decisions: Mapping[str, str]
    subset_sensitivity: str | None = None


@dataclass(frozen=True)
class Plan:
    """A loaded and checked plan. Every mapping is read-only, every list a tuple.

    `path` is the file it was read from and `sha256` its digest, plan_sha256()
    of its bytes, so a run can record exactly which plan it followed. The
    hypothesis sections other than H1b are keyed by task code; a task without
    an entry has none. `h2a_datasets` are the datasets H2a is computed on
    (hypotheses.h2a.datasets; since 24 Oct 2026 EWT test only). Which datasets
    a configuration is scored on, scored_on() says, and which members of a
    task are scored on a dataset, members_on().
    """

    id: str
    title: str
    date: str
    path: Path
    sha256: str
    training_data: Mapping[str, TrainingData]
    datasets: Mapping[str, Dataset]
    configurations: Mapping[str, Configuration]
    tasks: Mapping[str, Task]
    h1a: Mapping[str, tuple[Chain, ...]]
    h1b: H1b
    h2a: Mapping[str, tuple[Pair, ...]]
    h2a_datasets: tuple[str, ...]
    h2b: Mapping[str, Transition]
    h2c: Mapping[str, H2c]
    d2: D2
    d3: D3
    d4: D4
    statistics: Statistics

    def pending(self) -> tuple[str, ...]:
        """The keys of the configurations that cannot run yet, in plan order."""
        return tuple(key for key, config in self.configurations.items() if not config.ready)

    def members(self, task: str) -> tuple[str, ...]:
        """The configuration keys of one task, in the order the plan lists them.

        Raises:
            ValueError: If `task` is not one of TASKS.
        """
        if task not in self.tasks:
            raise ValueError(f"unknown task {task!r}; expected one of {list(TASKS)}")
        return tuple(self.tasks[task].members)

    def scored_on(self, key: str) -> tuple[str, ...]:
        """The datasets the plan scores configuration `key` on, in plan order.

        Every dataset that lists a task `key` is a member of, or, where the
        plan restricts it (configurations.<key>.datasets, since 24 Oct 2026),
        those of them it names: EWT test only for nltk_pre and spacy_sm.

        Raises:
            KeyError: If `key` is not a configuration of the plan.
        """
        restriction = self.configurations[key].datasets
        tasks = {code for code, task in self.tasks.items() if key in task.members}
        return tuple(
            dataset for dataset, spec in self.datasets.items()
            if tasks & set(spec.tasks) and (restriction is None or dataset in restriction)
        )

    def members_on(self, task: str, dataset: str) -> tuple[str, ...]:
        """The members of `task` the plan scores on `dataset`, in the order it lists them.

        What a stage of the task runs on the dataset (stage.select_keys) and
        what a counts file of it may hold (scripts/analyze.py). A member left
        out is not built, run or scored there.

        Raises:
            ValueError: If `task` is not one of TASKS, or `dataset` is not a
                dataset of the plan that lists the task.
        """
        members = self.members(task)
        spec = self.datasets.get(dataset)
        if spec is None or task not in spec.tasks:
            raise ValueError(f"dataset {dataset!r} is not one the plan lists for task {task}")
        return tuple(key for key in members if dataset in self.scored_on(key))

    def datasets_scoring(self, task: str, keys: Iterable[str]) -> tuple[str, ...]:
        """The datasets of `task` on which the plan scores every one of `keys`, in plan order."""
        keys = tuple(keys)
        return tuple(
            dataset for dataset, spec in self.datasets.items()
            if task in spec.tasks and all(dataset in self.scored_on(key) for key in keys)
        )

    def d3_datasets(self) -> tuple[str, ...]:
        """The datasets D3 is computed on: where its configuration and every one
        it is compared with are scored. EWT test only since 24 Oct 2026, when
        spacy_sm was restricted to it; until then every dataset of the task."""
        return self.datasets_scoring(self.d3.task, (self.d3.config, *self.d3.versus))

    def d4_datasets(self) -> tuple[str, ...]:
        """The datasets D4 is computed on: where both of its configurations are
        scored (EWT test and GENTLE)."""
        return self.datasets_scoring(self.d4.task, (self.d4.baseline, self.d4.treatment))

    def h2a_datasets_of(self, task: str) -> tuple[str, ...]:
        """The datasets the H2a pairs of `task` are computed on, in plan order.

        Those of h2a_datasets that list the task: since 24 Oct 2026 EWT test
        for POS and dependencies, and none for constituents, which have no
        pair. Until then H2a was computed on every dataset of its task.
        """
        return tuple(
            key for key, dataset in self.datasets.items()
            if key in self.h2a_datasets and task in dataset.tasks
        )

    def require_ready(self, keys: Iterable[str]) -> None:
        """Refuse to go on unless every configuration in `keys` can run.

        For a script that runs some configurations of a plan loaded without
        strict=True: it names what it is about to run, and a pending or
        unknown one stops it before any model is loaded.

        Raises:
            PlanError: Naming every key that is pending or not in the plan.
        """
        if isinstance(keys, str):
            keys = (keys,)
        errors = []
        for key in keys:
            config = self.configurations.get(key)
            if config is None:
                errors.append(
                    f"unknown configuration {key!r}; expected one of {list(self.configurations)}"
                )
            elif config.pending is not None:
                errors.append(_pending_message(key, config.pending))
        if errors:
            raise PlanError(self.path, errors)


def _pending_message(key: str, pending: Pending) -> str:
    """One line saying that a configuration is pending, why, and until which step."""
    step = f" (step {pending.step})" if pending.step else ""
    return f"configuration {key!r} is pending{step}: {pending.reason}"


# --- loading -----------------------------------------------------------------


def default_path() -> Path:
    """The plan the project follows: <root>/plan/v1.yaml."""
    return project_root() / "plan" / "v1.yaml"


def plan_sha256(data: bytes) -> str:
    """The digest a plan is known by: SHA-256 of its bytes with CRLF read as LF.

    The same plan is checked out with LF on one machine and with CRLF on
    another (core.autocrlf), and git, which seal.verify_lock asks whether the
    plan changed since the locked commit, ignores the difference. A digest of
    the raw bytes would not, and would break the lock of a plan nobody edited.
    Plan.sha256 and seal.verify_lock both take it from here. Since 23 Oct 2026;
    .gitattributes also keeps plan/*.yaml at LF in every checkout.
    """
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def load(path: str | Path | None = None, *, strict: bool = False) -> Plan:
    """Read a plan, check it, and return it as frozen data.

    Args:
        path:   The plan file; default_path() when None. A relative path is
                taken relative to the working directory, like any path a
                caller passes; only the default is anchored to the project.
        strict: Refuse every pending configuration that is not optional, and
                drop the optional pending ones from the plan returned. A run
                loads with strict=True; a check of the plan without it.

    Raises:
        FileNotFoundError: If the file does not exist, unchanged.
        PlanError: Listing every problem of the first phase that found any.
    """
    path = default_path() if path is None else Path(path)
    data = path.read_bytes()
    raw = _parse(data, path)

    errors: list[str] = []
    _plan_schema().check(raw, "the top level", errors)
    if errors:
        raise PlanError(path, errors)

    plan = _build(raw, path.resolve(), plan_sha256(data))
    errors = _Meaning(plan).check()
    if errors:
        raise PlanError(path, errors)

    if strict:
        plan = _runnable(plan, path)
    return plan


# --- before the phases: YAML -------------------------------------------------


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses a duplicate key and a key that is not a string.

    PyYAML keeps the last of two equal keys without a word, so a value edited
    in one place and left stale in another is read from whichever comes last.
    And YAML 1.1 reads `on`, `off`, `yes`, `no` and bare numbers as booleans
    and numbers, so `on:` becomes the key True and no lookup by name finds it.
    Both are recorded with their line rather than raised one at a time, so a
    file with several is reported whole.
    """

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self.problems: list[str] = []

    def construct_mapping(self, node: yaml.Node, deep: bool = False) -> dict:
        if isinstance(node, yaml.MappingNode):
            seen: set[str] = set()
            for key_node, _ in node.value:
                if key_node.tag == "tag:yaml.org,2002:merge":
                    continue
                key = self.construct_object(key_node, deep=True)
                line = key_node.start_mark.line + 1
                if not isinstance(key, str):
                    self.problems.append(f"non-string key {key!r} at line {line}; quote it")
                elif key in seen:
                    self.problems.append(f"duplicate key {key!r} at line {line}")
                else:
                    seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _parse(data: bytes, path: Path) -> Any:
    """The YAML document in `data`, or PlanError naming what made it unreadable."""
    try:
        # utf-8-sig: an editor on Windows may start the file with a BOM.
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise PlanError(path, [f"not UTF-8 text: {exc}"]) from None
    loader = _StrictLoader(text)
    try:
        raw = loader.get_single_data()
    except yaml.YAMLError as exc:
        raise PlanError(path, [*loader.problems, f"not valid YAML: {exc}"]) from None
    finally:
        loader.dispose()
    if loader.problems:
        raise PlanError(path, loader.problems)
    return raw


# --- phase 1: structure ------------------------------------------------------


def _kind(value: Any) -> str:
    """How a YAML value reads in a message: 'a string', 'a boolean', 'null'..."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, int):
        return "an integer"
    if isinstance(value, float):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, date):
        return "a date"
    if isinstance(value, list):
        return "a list"
    if isinstance(value, dict):
        return "a mapping"
    return type(value).__name__


def _wrong_type(where: str, expected: str, value: Any) -> str:
    """The message for a value of the wrong type, with a hint where YAML coerced it."""
    message = f"{where} must be {expected}, not {_kind(value)}"
    if "string" in expected and isinstance(value, (bool, int, float, date)):
        message += "; quote it"
    return message


class _Node:
    """One expected shape in the plan. check() appends what is wrong with a value."""

    def check(self, value: Any, where: str, errors: list[str]) -> None:
        raise NotImplementedError


class _Text(_Node):
    """A non-empty string: names, titles, reasons, paths, wrapper strings."""

    def check(self, value, where, errors):
        if not isinstance(value, str):
            errors.append(_wrong_type(where, "a string", value))
        elif not value.strip():
            errors.append(f"{where} is empty")


class _Id(_Node):
    """A key or a reference to one."""

    def check(self, value, where, errors):
        if not isinstance(value, str):
            errors.append(_wrong_type(where, "a string", value))
        elif not _ID.fullmatch(value):
            errors.append(f"{where} {value!r} is not an id; expected {_ID.pattern}")


class _Int(_Node):
    """An integer. A boolean is refused: YAML reads `true` as one and Python agrees."""

    def check(self, value, where, errors):
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(_wrong_type(where, "an integer", value))


class _Number(_Node):
    """An integer or a float, never a boolean."""

    def check(self, value, where, errors):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(_wrong_type(where, "a number", value))


class _Bool(_Node):
    """true or false, never a quoted 'yes': `optional` switches what a strict load drops."""

    def check(self, value, where, errors):
        if not isinstance(value, bool):
            errors.append(_wrong_type(where, "true or false", value))


class _Value(_Node):
    """Any YAML value: a constructor or load_model argument, checked by its wrapper."""

    def check(self, value, where, errors):
        pass


class _Section(_Node):
    """A two-digit WSJ section, quoted so YAML keeps its leading zero."""

    def check(self, value, where, errors):
        if not isinstance(value, str):
            errors.append(_wrong_type(where, "a two-digit string", value))
        elif not _SECTION.fullmatch(value):
            errors.append(f"{where} {value!r} is not a two-digit WSJ section")


class _Date(_Node):
    """An ISO date, as a string: unquoted, YAML would make it a date object.

    YYYY-MM-DD only. date.fromisoformat alone would also take the basic form
    20261023 and the week form 2026-W43-5 (Python 3.11 and later), so the
    pattern is checked first and fromisoformat then refuses a day that does
    not exist, such as 2026-02-30.
    """

    def check(self, value, where, errors):
        if not isinstance(value, str):
            errors.append(_wrong_type(where, "a string", value))
            return
        if not _DATE.fullmatch(value):
            errors.append(f"{where} {value!r} is not a date; expected YYYY-MM-DD")
            return
        try:
            date.fromisoformat(value)
        except ValueError:
            errors.append(f"{where} {value!r} is not a date; expected YYYY-MM-DD")


class _Schema(_Node):
    """The schema version; a plan written for another one is refused, not read by these rules."""

    def check(self, value, where, errors):
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(_wrong_type(where, "an integer", value))
        elif value != _SCHEMA:
            errors.append(f"unknown schema {value!r} in {where}; expected one of [{_SCHEMA}]")


class _Level(_Node):
    """A method level code, L0..L4, as MethodLevel.code spells it."""

    def check(self, value, where, errors):
        if not isinstance(value, str) or value not in _LEVELS:
            errors.append(f"unknown level {value!r} in {where}; expected one of {list(_LEVELS)}")


class _Role(_Node):
    """One of ROLES; an unknown role would be a comparison no section computes."""

    def check(self, value, where, errors):
        if not isinstance(value, str) or value not in ROLES:
            errors.append(f"unknown role {value!r} in {where}; expected one of {list(ROLES)}")


class _List(_Node):
    """A list of one shape; `unique` refuses a repeated item, `nonempty` an empty list."""

    def __init__(self, item: _Node, *, unique: bool = False, nonempty: bool = False) -> None:
        self.item, self.unique, self.nonempty = item, unique, nonempty

    def check(self, value, where, errors):
        if not isinstance(value, list):
            errors.append(_wrong_type(where, "a list", value))
            return
        if self.nonempty and not value:
            errors.append(f"{where} is empty")
        for index, item in enumerate(value):
            self.item.check(item, f"{where}[{index}]", errors)
        if self.unique:
            repeated = []
            for index, item in enumerate(value):
                if item in value[:index] and item not in repeated:
                    repeated.append(item)
            for item in repeated:
                errors.append(f"duplicate {item!r} in {where}")


class _Map(_Node):
    """A mapping with a fixed set of keys: every required one, any optional one.

    The key sets are explicit at every level, because a key that is ignored is
    the one failure no later check can see: the plan says one thing and the
    run does another.
    """

    def __init__(
        self, required: Mapping[str, _Node], optional: Mapping[str, _Node] | None = None
    ) -> None:
        self.required = dict(required)
        self.optional = dict(optional or {})

    def check(self, value, where, errors):
        if not isinstance(value, dict):
            errors.append(_wrong_type(where, "a mapping", value))
            return
        allowed = [*self.required, *self.optional]
        for key in value:
            if key not in allowed:
                errors.append(f"unknown key {key!r} in {where}; expected one of {allowed}")
        for key in self.required:
            if key not in value:
                errors.append(f"missing key {key!r} in {where}")
        for key, node in (*self.required.items(), *self.optional.items()):
            if key in value:
                node.check(value[key], _join(where, key), errors)


class _Table(_Node):
    """A mapping from ids to values of one shape: configurations, members, init."""

    def __init__(self, item: _Node) -> None:
        self.item = item

    def check(self, value, where, errors):
        if not isinstance(value, dict):
            errors.append(_wrong_type(where, "a mapping", value))
            return
        for key, item in value.items():
            if not _ID.fullmatch(key):
                errors.append(f"key {key!r} in {where} is not an id; expected {_ID.pattern}")
            self.item.check(item, _join(where, key), errors)


def _join(where: str, key: str) -> str:
    """The path of a key inside `where`, e.g. 'configurations.stanza_ewt'."""
    return key if where == "the top level" else f"{where}.{key}"


def _plan_schema() -> _Map:
    """Every key the plan may hold, at every level, and the shape of its value."""
    ident, text = _Id(), _Text()
    ids = _List(ident, unique=True)
    some_ids = _List(ident, unique=True, nonempty=True)

    def by_task(node: _Node) -> _Map:
        # A hypothesis section is keyed by task code; a task may be absent.
        return _Map({}, {code: node for code in TASKS})

    def task(code: str) -> _Map:
        member = _Map({"level": _Level(), "roles": _List(_Role(), unique=True, nonempty=True)})
        return _Map({
            "metric": _Map({name: ident for name in _METRICS[code]}),
            "configs": _Table(member),
        })

    chain = _Map(
        {"id": ident, "training": ident, "configs": ids},
        {"approximations": _Table(ident)},
    )
    step = _Map({
        "baseline": ident,
        "treatment": ident,
        "legs": _List(_Map({"task": ident, "metric": ident}), nonempty=True),
        "datasets": some_ids,
        "verdict": ident,
    })
    pair = _Map({"id": ident, "baseline": ident, "treatment": ident})
    transition = _Map({"source": ident, "target": ident, "configs": some_ids})
    contrast = _Map({
        "baseline": ident,
        "treatment": ident,
        "tests": _List(_Map({"dataset": ident, "decision": ident}), nonempty=True),
    })
    return _Map({
        "schema": _Schema(),
        "plan": _Map({"id": ident, "title": text, "date": _Date()}),
        "training_data": _Table(_Map(
            {"name": text, "family": ident, "annotation": ident},
            {"path": text},
        )),
        "datasets": _Table(_Map(
            {
                "name": text,
                "tasks": ids,
                "status": ident,
                "license": text,
                "resampling": _Map(
                    {"unit": ident},
                    {"unit_id": ident, "expected_units": _Int()},
                ),
            },
            {
                "path": text,
                "nltk_sections": _List(_Section(), unique=True, nonempty=True),
                "source": _Map({
                    "repository": text,
                    "ref": text,
                    "files": _List(text, unique=True, nonempty=True),
                }),
                "tokens": _Map({"brackets": ident, "quotes": ident}),
            },
        )),
        "configurations": _Table(_Map(
            {
                "wrapper": text,
                "model": text,
                "training": ident,
                "init": _Table(_Value()),
                "load": _Table(_Value()),
            },
            {
                "optional": _Bool(),
                "inputs": _Table(ident),
                # The datasets it is scored on, when not every dataset of its
                # tasks (24 Oct 2026); checked by _Meaning._restrictions.
                "datasets": some_ids,
                "pending": _Map({"reason": text, "missing_params": ids}, {"step": text}),
            },
        )),
        "tasks": _Map({code: task(code) for code in TASKS}),
        "hypotheses": _Map({
            "h1a": by_task(_List(chain, nonempty=True)),
            "h1b": step,
            # The datasets H2a is computed on, beside its pairs by task.
            "h2a": _Map(
                {"datasets": some_ids},
                {code: _List(pair, nonempty=True) for code in TASKS},
            ),
            "h2b": by_task(transition),
            "h2c": by_task(contrast),
        }),
        "descriptive": _Map({
            "d2": _Map({
                "task": ident,
                "configs": some_ids,
                "versus": some_ids,
                "datasets": some_ids,
                "sample": _Map({
                    "sentences": _Int(),
                    "seed": _Int(),
                    "max_tokens": _Int(),
                    "drawn_from": ident,
                }),
                "report": some_ids,
            }),
            "d3": _Map({"task": ident, "config": ident, "versus": some_ids, "metrics": some_ids}),
            "d4": _Map({"task": ident, "baseline": ident, "treatment": ident}),
        }),
        "statistics": _Map(
            {
                "scale": ident,
                "delta": _Number(),
                "delta_sensitivity": _List(_Number()),
                "bootstrap": _Map({
                    "method": ident,
                    "replicates": _Int(),
                    "confidence": _Number(),
                    "seed": _Int(),
                }),
                "unit_sensitivity": ident,
                "decisions": _Map({decision: ident for decision in _DECISIONS}),
            },
            # Optional, so a plan written before it (24 Oct 2026) still loads.
            {"subset_sensitivity": ident},
        ),
    })


def _freeze(value: Any) -> Any:
    """A read-only copy: mappings become MappingProxyType, lists tuples."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _build(raw: Mapping[str, Any], path: Path, sha256: str) -> Plan:
    """The frozen Plan of a document that passed the structure phase."""
    hypotheses, descriptive, statistics = raw["hypotheses"], raw["descriptive"], raw["statistics"]
    d2, d3, d4 = (descriptive[key] for key in ("d2", "d3", "d4"))
    h1b, bootstrap = hypotheses["h1b"], statistics["bootstrap"]
    return Plan(
        id=raw["plan"]["id"],
        title=raw["plan"]["title"],
        date=raw["plan"]["date"],
        path=path,
        sha256=sha256,
        training_data=MappingProxyType({
            key: TrainingData(
                id=key, name=item["name"], family=item["family"],
                annotation=item["annotation"], path=item.get("path"),
            )
            for key, item in raw["training_data"].items()
        }),
        datasets=MappingProxyType({
            key: Dataset(
                id=key,
                name=item["name"],
                tasks=tuple(item["tasks"]),
                status=item["status"],
                license=item["license"],
                resampling=Resampling(
                    unit=item["resampling"]["unit"],
                    unit_id=item["resampling"].get("unit_id"),
                    expected_units=item["resampling"].get("expected_units"),
                ),
                path=item.get("path"),
                nltk_sections=(
                    tuple(item["nltk_sections"]) if "nltk_sections" in item else None
                ),
                source=(
                    Source(
                        repository=item["source"]["repository"],
                        ref=item["source"]["ref"],
                        files=tuple(item["source"]["files"]),
                    )
                    if "source" in item else None
                ),
                tokens=(
                    Tokens(brackets=item["tokens"]["brackets"], quotes=item["tokens"]["quotes"])
                    if "tokens" in item else None
                ),
            )
            for key, item in raw["datasets"].items()
        }),
        configurations=MappingProxyType({
            key: Configuration(
                key=key,
                wrapper=item["wrapper"],
                model=item["model"],
                training=item["training"],
                init=_freeze(item["init"]),
                load=_freeze(item["load"]),
                inputs=_freeze(item.get("inputs", {})),
                optional=item.get("optional", False),
                pending=(
                    Pending(
                        reason=item["pending"]["reason"],
                        missing_params=tuple(item["pending"]["missing_params"]),
                        step=item["pending"].get("step"),
                    )
                    if "pending" in item else None
                ),
                datasets=tuple(item["datasets"]) if "datasets" in item else None,
            )
            for key, item in raw["configurations"].items()
        }),
        tasks=MappingProxyType({
            code: Task(
                code=code,
                metric=Metric(
                    name=item["metric"]["name"],
                    aggregation=item["metric"]["aggregation"],
                    options=MappingProxyType({
                        name: value for name, value in item["metric"].items()
                        if name not in ("name", "aggregation")
                    }),
                ),
                members=MappingProxyType({
                    key: Member(level=_LEVELS[member["level"]], roles=frozenset(member["roles"]))
                    for key, member in item["configs"].items()
                }),
            )
            for code, item in raw["tasks"].items()
        }),
        h1a=MappingProxyType({
            code: tuple(
                Chain(
                    id=chain["id"], task=code, training=chain["training"],
                    configs=tuple(chain["configs"]),
                    approximations=_freeze(chain.get("approximations", {})),
                )
                for chain in chains
            )
            for code, chains in hypotheses["h1a"].items()
        }),
        h1b=H1b(
            baseline=h1b["baseline"], treatment=h1b["treatment"],
            legs=tuple(Leg(task=leg["task"], metric=leg["metric"]) for leg in h1b["legs"]),
            datasets=tuple(h1b["datasets"]), verdict=h1b["verdict"],
        ),
        h2a=MappingProxyType({
            code: tuple(
                Pair(
                    id=pair["id"], task=code,
                    baseline=pair["baseline"], treatment=pair["treatment"],
                )
                for pair in pairs
            )
            for code, pairs in hypotheses["h2a"].items() if code != "datasets"
        }),
        h2a_datasets=tuple(hypotheses["h2a"]["datasets"]),
        h2b=MappingProxyType({
            code: Transition(
                task=code, source=item["source"], target=item["target"],
                configs=tuple(item["configs"]),
            )
            for code, item in hypotheses["h2b"].items()
        }),
        h2c=MappingProxyType({
            code: H2c(
                task=code, baseline=item["baseline"], treatment=item["treatment"],
                tests=tuple(
                    Check(dataset=test["dataset"], decision=test["decision"])
                    for test in item["tests"]
                ),
            )
            for code, item in hypotheses["h2c"].items()
        }),
        d2=D2(
            task=d2["task"],
            configs=tuple(d2["configs"]),
            versus=tuple(d2["versus"]),
            datasets=tuple(d2["datasets"]),
            sample=Sample(
                sentences=d2["sample"]["sentences"],
                seed=d2["sample"]["seed"],
                max_tokens=d2["sample"]["max_tokens"],
                drawn_from=d2["sample"]["drawn_from"],
            ),
            report=tuple(d2["report"]),
        ),
        d3=D3(
            task=d3["task"], config=d3["config"],
            versus=tuple(d3["versus"]), metrics=tuple(d3["metrics"]),
        ),
        d4=D4(task=d4["task"], baseline=d4["baseline"], treatment=d4["treatment"]),
        statistics=Statistics(
            scale=statistics["scale"],
            delta=float(statistics["delta"]),
            delta_sensitivity=tuple(float(value) for value in statistics["delta_sensitivity"]),
            bootstrap=Bootstrap(
                method=bootstrap["method"],
                replicates=bootstrap["replicates"],
                confidence=float(bootstrap["confidence"]),
                seed=bootstrap["seed"],
            ),
            unit_sensitivity=statistics["unit_sensitivity"],
            decisions=MappingProxyType(dict(statistics["decisions"])),
            subset_sensitivity=statistics.get("subset_sensitivity"),
        ),
    )


# --- phase 2: meaning --------------------------------------------------------


def _wrapper_modules() -> tuple[str, ...]:
    """The modules of syntactic_analysis.wrappers, found without importing them."""
    from syntactic_analysis import wrappers

    return tuple(sorted(module.name for module in pkgutil.iter_modules(wrappers.__path__)))


def _named_parameters(target: Any) -> tuple[str, ...]:
    """The parameters `target` accepts by name.

    **kwargs never counts. Most wrappers refuse what arrives through it only
    when load_model() runs, which is too late for a check of the plan, and the
    spaCy and Stanza wrappers forward it to their library, where nothing here
    can check it. A plan passes only what a wrapper names. A positional-only
    parameter cannot be passed by name and does not count either.
    """
    named = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return tuple(
        parameter.name
        for parameter in inspect.signature(target).parameters.values()
        if parameter.kind in named and parameter.name != "self"
    )


def _inside(path: str) -> bool:
    """True for a relative POSIX path that stays inside the directory it is read from.

    No drive (`C:`), no backslash, no empty, `.` or `..` part: so a path in the
    plan names one place under the project root on every system, and two
    paths can be compared part by part.
    """
    parts = path.split("/")
    return "\\" not in path and ":" not in path and not any(
        part in ("", ".", "..") for part in parts
    )


class _Meaning:
    """Phase 2: the checks that need the whole plan, collected into one list.

    Each rule reports what it finds and skips what an earlier rule has already
    reported as unknown, so one mistake is not reported once per rule that
    reads it.
    """

    def __init__(self, plan: Plan) -> None:
        self.plan = plan
        self.errors: list[str] = []
        # Configurations whose wrapper did not resolve: already reported, so no
        # later rule compares their wrapper strings.
        self.unresolved: set[str] = set()

    def check(self) -> list[str]:
        self._wrappers()
        self._inputs()
        self._references()
        self._datasets()
        self._metrics()
        self._h1a()
        self._h1b()
        self._h2a()
        self._h2b()
        self._h2c()
        self._d2()
        self._d3()
        self._d4()
        self._restrictions()
        self._roles()
        self._statistics()
        return self.errors

    # helpers

    def _members(self, task: str, keys: Iterable[str], where: str) -> bool:
        """Report every key that is not a configuration in `task`; True if none."""
        members = self.plan.tasks[task].members
        known = True
        for key in keys:
            if key not in members:
                self.errors.append(
                    f"unknown configuration {key!r} in {where}; expected one of "
                    f"{list(members)} (the members of tasks.{task})"
                )
                known = False
            elif key not in self.plan.configurations:
                known = False  # reported by _references
        return known

    def _dataset(self, key: str, task: str, where: str) -> Dataset | None:
        """The dataset `key`, reported if it does not list `task`; None, reported, if unknown."""
        dataset = self.plan.datasets.get(key)
        if dataset is None:
            self.errors.append(
                f"unknown dataset {key!r} in {where}; expected one of {list(self.plan.datasets)}"
            )
            return None
        if task not in dataset.tasks:
            self.errors.append(f"dataset {key!r} in {where} does not list task {task}")
        return dataset

    def _annotation(self, key: str) -> str | None:
        """The annotation of a configuration's training data, None if either is unknown."""
        config = self.plan.configurations.get(key)
        training = self.plan.training_data.get(config.training) if config else None
        return training.annotation if training else None

    def _seed(self, value: int, where: str) -> None:
        if not 0 <= value <= _SEED_MAX:
            self.errors.append(f"{where} {value} is outside [0, {_SEED_MAX}]")

    # 1-2: wrappers and their signatures

    def _wrappers(self) -> None:
        modules = _wrapper_modules()
        imported: dict[str, Any] = {}
        for key, config in self.plan.configurations.items():
            where = f"configurations.{key}"
            init_key = config.init.get("key")
            if init_key is not None and init_key != key:
                self.errors.append(
                    f"init.key {init_key!r} in {where} differs from the configuration key {key!r}"
                )
            cls = self._resolve(config.wrapper, f"{where}.wrapper", modules, imported)
            if cls is None:
                self.unresolved.add(key)
            else:
                self._arguments(key, config, cls)

    def _resolve(self, wrapper: str, where: str, modules: tuple[str, ...],
                 imported: dict[str, Any]) -> type | None:
        """The analyzer class `wrapper` names, or None after reporting why not."""
        module_name, dot, class_name = wrapper.partition(".")
        if not (dot and module_name.isidentifier() and class_name.isidentifier()):
            self.errors.append(
                f"malformed wrapper {wrapper!r} in {where}; expected '<module>.<Class>' "
                f"with the module in syntactic_analysis.wrappers"
            )
            return None
        if module_name not in modules:
            self.errors.append(
                f"unknown wrapper module {module_name!r} in {where}; "
                f"expected one of {list(modules)}"
            )
            return None
        if module_name not in imported:
            try:
                imported[module_name] = importlib.import_module(
                    f"syntactic_analysis.wrappers.{module_name}"
                )
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                imported[module_name] = exc
        module = imported[module_name]
        if isinstance(module, Exception):
            self.errors.append(
                f"wrapper module {module_name!r} failed to import in {where}: "
                f"{type(module).__name__}: {module}"
            )
            return None
        cls = getattr(module, class_name, None)
        if cls is None:
            analyzers = sorted(
                name for name, value in vars(module).items()
                if inspect.isclass(value) and issubclass(value, BaseAnalyzer)
                and not inspect.isabstract(value)
            )
            self.errors.append(
                f"unknown class {class_name!r} in {where}; expected one of {analyzers} "
                f"(the analyzers of wrapper module {module_name!r})"
            )
            return None
        if not (inspect.isclass(cls) and issubclass(cls, BaseAnalyzer)) or inspect.isabstract(cls):
            self.errors.append(
                f"{wrapper!r} in {where} is not an analyzer: a wrapper names a concrete "
                f"subclass of BaseAnalyzer"
            )
            return None
        return cls

    def _arguments(self, key: str, config: Configuration, cls: type) -> None:
        """init, inputs and load against the signatures of the class and load_model."""
        where = f"configurations.{key}"
        accepted = _named_parameters(cls)
        # An input of an unknown name is reported once, by _inputs.
        given = {
            **{name: "init" for name in config.init},
            **{name: "inputs" for name in config.inputs if name in _INPUTS},
        }
        unbound = [name for name in given if name not in accepted]
        if config.pending is None:
            for name in unbound:
                self.errors.append(
                    f"unknown parameter {name!r} in {where}.{given[name]}; "
                    f"expected one of {sorted(accepted)}"
                )
        else:
            listed = config.pending.missing_params
            for name in listed:
                if name in accepted:
                    self.errors.append(
                        f"stale entry {name!r} in {where}.pending.missing_params: "
                        f"{cls.__name__} accepts it"
                    )
                elif name not in given:
                    self.errors.append(
                        f"entry {name!r} in {where}.pending.missing_params is not a key "
                        f"of init or inputs"
                    )
            for name in unbound:
                if name not in listed:
                    self.errors.append(
                        f"unknown parameter {name!r} in {where}.{given[name]} is not listed in "
                        f"pending.missing_params; expected one of {sorted(accepted)}"
                    )
        load_accepted = _named_parameters(cls.load_model)
        for name in config.load:
            if name not in load_accepted:
                self.errors.append(
                    f"unknown load parameter {name!r} in {where}.load; "
                    f"expected one of {sorted(load_accepted)}"
                )

    # 3: inputs

    def _inputs(self) -> None:
        configs = self.plan.configurations
        taggers = self.plan.tasks["pos"].members
        for key, config in configs.items():
            for name, target in config.inputs.items():
                where = f"configurations.{key}.inputs.{name}"
                if name not in _INPUTS:
                    self.errors.append(
                        f"unknown input {name!r} in configurations.{key}.inputs; "
                        f"expected one of {list(_INPUTS)}"
                    )
                    continue
                if name in config.init:
                    self.errors.append(
                        f"{name!r} is set in both configurations.{key}.init and {where}"
                    )
                if target == key:
                    self.errors.append(f"{where} names the configuration itself")
                elif target not in configs:
                    self.errors.append(
                        f"unknown configuration {target!r} in {where}; "
                        f"expected one of {list(configs)}"
                    )
                else:
                    if target not in taggers:
                        self.errors.append(
                            f"{where} {target!r} is not a member of task pos; "
                            f"a pos_source is a tagger the plan scores"
                        )
                    if config.ready and not configs[target].ready:
                        self.errors.append(
                            f"configurations.{key} is ready but its input {name} "
                            f"{target!r} is pending"
                        )

    # 4: references

    def _references(self) -> None:
        plan = self.plan
        configs = plan.configurations
        for key, training in plan.training_data.items():
            if training.annotation not in _ANNOTATIONS:
                self.errors.append(
                    f"unknown annotation {training.annotation!r} in "
                    f"training_data.{key}.annotation; expected one of {list(_ANNOTATIONS)}"
                )
        for key, config in configs.items():
            if config.training not in plan.training_data:
                self.errors.append(
                    f"unknown training {config.training!r} in configurations.{key}.training; "
                    f"expected one of {list(plan.training_data)}"
                )
        in_a_task = {key for task in plan.tasks.values() for key in task.members}
        for key in configs:
            if key not in in_a_task:
                self.errors.append(
                    f"configuration {key!r} is in no task; list it under tasks.<task>.configs "
                    f"or remove it"
                )
        for code, task in plan.tasks.items():
            for key, member in task.members.items():
                if key not in configs:
                    self.errors.append(
                        f"unknown configuration {key!r} in tasks.{code}.configs; "
                        f"expected one of {list(configs)}"
                    )
                elif configs[key].optional and member.roles != {"descriptive"}:
                    self.errors.append(
                        f"optional configuration {key!r} has role(s) "
                        f"{sorted(member.roles - {'descriptive'})} in tasks.{code}.configs.{key}; "
                        f"an optional configuration may only be descriptive"
                    )
        # An optional configuration may be dropped by a strict load, so nothing
        # the plan compares or builds may lean on it, roles or not.
        optional = {key for key, config in configs.items() if config.optional}
        leaning = [
            ("descriptive.d2.versus", plan.d2.versus),
            ("descriptive.d3.versus", plan.d3.versus),
            *((f"configurations.{key}.inputs", config.inputs.values())
              for key, config in configs.items()),
        ]
        for where, keys in leaning:
            for key in keys:
                if key in optional:
                    self.errors.append(
                        f"optional configuration {key!r} in {where}; "
                        f"an optional configuration may only be descriptive"
                    )

    # 5: datasets

    def _datasets(self) -> None:
        located: list[tuple[str, tuple[str, ...]]] = []
        for key, dataset in self.plan.datasets.items():
            where = f"datasets.{key}"
            if dataset.path is not None:
                if _inside(dataset.path):
                    # Without case: Windows, where the project runs, does not
                    # tell DATA/x from data/x.
                    located.append((key, tuple(dataset.path.casefold().split("/"))))
                else:
                    self.errors.append(
                        f"{where}.path {dataset.path!r} is not a relative path inside the project"
                    )
            if not dataset.tasks:
                self.errors.append(f"{where}.tasks is empty")
            for task in dataset.tasks:
                if task not in TASKS:
                    self.errors.append(
                        f"unknown task {task!r} in {where}.tasks; expected one of {list(TASKS)}"
                    )
            if dataset.status not in _STATUSES:
                self.errors.append(
                    f"unknown status {dataset.status!r} in {where}.status; "
                    f"expected one of {list(_STATUSES)}"
                )
            if dataset.path is not None and dataset.nltk_sections is not None:
                self.errors.append(
                    f"{where} has both path and nltk_sections; expected exactly one"
                )
            elif dataset.path is None and dataset.nltk_sections is None:
                self.errors.append(
                    f"{where} has neither path nor nltk_sections; expected exactly one"
                )
            if dataset.source is not None:
                self._source(dataset, where)
            elif dataset.status == "confirmatory":
                self.errors.append(
                    f"{where} is confirmatory and has no source; the data a confirmatory "
                    f"result rests on is downloaded from a fixed ref, so it can be named"
                )
            self._tokens(dataset, where)
            resampling = dataset.resampling
            unit_ids = _UNIT_IDS.get(resampling.unit)
            if unit_ids is None:
                self.errors.append(
                    f"unknown unit {resampling.unit!r} in {where}.resampling.unit; "
                    f"expected one of {list(_UNIT_IDS)}"
                )
            elif resampling.unit_id is None and unit_ids:
                self.errors.append(
                    f"missing key 'unit_id' in {where}.resampling; unit {resampling.unit!r} "
                    f"is read from one of {list(unit_ids)}"
                )
            elif resampling.unit_id is not None and resampling.unit_id not in unit_ids:
                expected = (
                    f"expected one of {list(unit_ids)}" if unit_ids else "it takes no unit_id"
                )
                self.errors.append(
                    f"unit_id {resampling.unit_id!r} does not fit unit {resampling.unit!r} "
                    f"in {where}.resampling; {expected}"
                )
            if resampling.expected_units is not None and resampling.expected_units < 1:
                self.errors.append(
                    f"{where}.resampling.expected_units must be >= 1, "
                    f"not {resampling.expected_units}"
                )
        # One local copy per dataset. A dataset whose path is, holds or lies in
        # another's copy reads that copy under its own status: an exploratory
        # entry at GENTLE's path would open GENTLE with no lock.
        for index, (first, parts) in enumerate(located):
            for second, other in located[index + 1:]:
                shorter = min(len(parts), len(other))
                if parts[:shorter] == other[:shorter]:
                    self.errors.append(
                        f"datasets.{first}.path {self.plan.datasets[first].path!r} and "
                        f"datasets.{second}.path {self.plan.datasets[second].path!r} overlap; "
                        f"each dataset has a local copy of its own"
                    )

    def _tokens(self, dataset: Dataset, where: str) -> None:
        """A constituency dataset says how it writes brackets and quotes; no other dataset does.

        The constituency stage gives every configuration its sentences in the
        PTB convention and converts from the one the dataset declares
        (ptb_conventions.parser_tokens), so a constituency dataset without the
        declaration could not be run, and a declaration on any other dataset
        would be a setting nothing reads.
        """
        tokens = dataset.tokens
        if "con" not in dataset.tasks:
            if tokens is not None:
                self.errors.append(
                    f"{where}.tokens is declared, but {where} does not list task con; only "
                    f"the sentences of a constituency dataset are converted"
                )
            return
        if tokens is None:
            self.errors.append(
                f"{where} lists task con and has no tokens; a constituency dataset declares "
                f"how it writes brackets and double quotes (tokens: {{brackets: ..., quotes: ...}}), "
                f"which decides what its sentences are converted from before they are parsed"
            )
            return
        if tokens.brackets not in BRACKET_CONVENTIONS:
            self.errors.append(
                f"unknown bracket convention {tokens.brackets!r} in {where}.tokens.brackets; "
                f"expected one of {list(BRACKET_CONVENTIONS)}: a bracketed tree writes a "
                f"parenthesis as -LRB- or -RRB-, since a raw one would be read as structure"
            )
        if tokens.quotes not in QUOTE_CONVENTIONS:
            self.errors.append(
                f"unknown quote convention {tokens.quotes!r} in {where}.tokens.quotes; "
                f"expected one of {list(QUOTE_CONVENTIONS)}"
            )

    def _source(self, dataset: Dataset, where: str) -> None:
        """A source scripts/fetch_data.py can download, into a local copy it can name."""
        source = dataset.source
        if dataset.path is None:
            self.errors.append(
                f"{where}.source needs {where}.path, where the local copy is kept"
            )
        where = f"{where}.source"
        if not _GITHUB.fullmatch(source.repository):
            self.errors.append(
                f"repository {source.repository!r} in {where} is not "
                f"https://github.com/<owner>/<repo>; scripts/fetch_data.py downloads from GitHub"
            )
        if source.ref in _MOVING_REFS:
            self.errors.append(
                f"ref {source.ref!r} in {where} names a branch, which moves; pin a tag or a commit"
            )
        elif not _REF.fullmatch(source.ref):
            self.errors.append(f"ref {source.ref!r} in {where} is not a tag or a commit")
        for index, file in enumerate(source.files):
            parts = file.split("/")
            if "\\" in file or any(part in ("", ".", "..") for part in parts):
                self.errors.append(
                    f"{where}.files[{index}] {file!r} is not a relative path inside the repository"
                )
        if len(source.files) > 1:
            names = [PurePosixPath(file).name for file in source.files]
            for name in sorted({name for name in names if names.count(name) > 1}):
                self.errors.append(
                    f"two files in {where}.files are named {name!r}; the local copy in "
                    f"{dataset.path} keeps their names"
                )

    # 6: metrics

    def _metrics(self) -> None:
        for code, task in self.plan.tasks.items():
            stated = {
                "name": task.metric.name,
                "aggregation": task.metric.aggregation,
                **task.metric.options,
            }
            for field, expected in _METRICS[code].items():
                if stated[field] != expected:
                    self.errors.append(
                        f"unknown {field} {stated[field]!r} in tasks.{code}.metric; "
                        f"expected one of [{expected!r}]"
                    )

    # 7: H1a and H1b

    def _h1a(self) -> None:
        plan = self.plan
        configs, training_data = plan.configurations, plan.training_data
        first: dict[str, str] = {}
        for code, chains in plan.h1a.items():
            members = plan.tasks[code].members
            for index, chain in enumerate(chains):
                where = f"hypotheses.h1a.{code}[{index}]"
                name = f"chain {chain.id!r} ({where})"
                if chain.id in first:
                    self.errors.append(
                        f"duplicate chain id {chain.id!r} in {where}; "
                        f"first used in {first[chain.id]}"
                    )
                else:
                    first[chain.id] = where
                if chain.training not in training_data:
                    self.errors.append(
                        f"unknown training {chain.training!r} in {where}.training; "
                        f"expected one of {list(training_data)}"
                    )
                if len(chain.configs) < 2:
                    self.errors.append(
                        f"{name} has {len(chain.configs)} configuration(s); "
                        f"a chain needs at least 2"
                    )
                self._members(code, chain.configs, f"{where}.configs")
                known = [key for key in chain.configs if key in members and key in configs]
                for key in known:
                    # The brief: L0 does not enter H1. The optional rdr_ewt is
                    # kept out by the rule on optional configurations; this
                    # keeps out any rule-based one.
                    if members[key].level == MethodLevel.RULE:
                        self.errors.append(f"{key!r} in {name} is L0; L0 does not enter H1")
                for lower, upper in zip(known, known[1:]):
                    if members[upper].level <= members[lower].level:
                        self.errors.append(
                            f"levels do not rise in {name}: {lower} {members[lower].level.code} "
                            f"then {upper} {members[upper].level.code}"
                        )
                for key in chain.approximations:
                    if key not in chain.configs:
                        self.errors.append(
                            f"approximations key {key!r} in {where} is not a configuration "
                            f"of the chain"
                        )
                for key in known:
                    self._chain_training(chain, key, name, where)
                allowed = {chain.training, *chain.approximations.values()}
                for key in known:
                    for input_name, target in configs[key].inputs.items():
                        if target in configs and configs[target].training not in allowed:
                            self.errors.append(
                                f"input {input_name} {target!r} of {key!r} in {name} is "
                                f"trained on {configs[target].training!r}, not the chain's "
                                f"{chain.training!r}"
                            )

    def _chain_training(self, chain: Chain, key: str, name: str, where: str) -> None:
        """A chain member trains on the chain's corpus, or on a listed release of it."""
        training_data = self.plan.training_data
        actual = self.plan.configurations[key].training
        listed = chain.approximations.get(key)
        if actual not in training_data:
            return  # reported by _references
        if actual == chain.training:
            if listed is not None:
                self.errors.append(
                    f"{key!r} is listed under approximations in {where} but is trained on the "
                    f"chain's own {chain.training!r}"
                )
            return
        if listed is None:
            self.errors.append(
                f"{key!r} in {name} is trained on {actual!r}, not the chain's {chain.training!r}; "
                f"list it under approximations if that is another release of the same corpus"
            )
        elif listed != actual:
            self.errors.append(
                f"approximations.{key} {listed!r} in {where} is not the training of {key!r}, "
                f"which is {actual!r}"
            )
        elif chain.training in training_data:
            family = training_data[actual].family
            chain_family = training_data[chain.training].family
            if family != chain_family:
                self.errors.append(
                    f"approximation {key!r} in {name}: {actual!r} (family {family}) is not a "
                    f"release of the chain's {chain.training!r} (family {chain_family})"
                )

    def _h1b(self) -> None:
        """One level step on one corpus, in two tasks, on datasets that hold both.

        Its two Deltas come from one bootstrap replicate, so both legs must
        resample the same unit. The plan resamples a dataset by one unit
        whatever task is scored on it, so that holds when every dataset lists
        the task of each leg, which is what is checked.
        """
        h1b, where = self.plan.h1b, "hypotheses.h1b"
        datasets = self.plan.datasets
        tasks: list[str] = []
        if len(h1b.legs) != 2:
            self.errors.append(
                f"{where}.legs has {len(h1b.legs)} leg(s); H1b compares one step in 2 tasks"
            )
        for index, leg in enumerate(h1b.legs):
            leg_where = f"{where}.legs[{index}]"
            if leg.task not in TASKS:
                self.errors.append(
                    f"unknown task {leg.task!r} in {leg_where}.task; expected one of {list(TASKS)}"
                )
            elif leg.task in tasks:
                self.errors.append(f"task {leg.task} is in two legs of {where}")
            else:
                tasks.append(leg.task)
                metric = self.plan.tasks[leg.task].metric.name
                if leg.metric != metric:
                    self.errors.append(
                        f"unknown metric {leg.metric!r} in {leg_where}.metric; "
                        f"expected one of [{metric!r}]"
                    )
        if h1b.baseline == h1b.treatment:
            self.errors.append(f"{where} compares {h1b.baseline!r} with itself")
        else:
            for task in tasks:
                self._step(task, h1b.baseline, h1b.treatment, "the H1b step", where, adjacent=True)
        for index, key in enumerate(h1b.datasets):
            dataset_where = f"{where}.datasets[{index}]"
            if key not in datasets:
                self.errors.append(
                    f"unknown dataset {key!r} in {dataset_where}; expected one of {list(datasets)}"
                )
                continue
            for task in tasks:
                if task not in datasets[key].tasks:
                    self.errors.append(
                        f"dataset {key!r} in {dataset_where} does not list task {task}; both "
                        f"legs are scored, and resampled, on the same documents"
                    )
        verdict = datasets.get(h1b.verdict)
        if h1b.verdict not in h1b.datasets:
            self.errors.append(
                f"verdict {h1b.verdict!r} in {where} is not one of its datasets "
                f"{list(h1b.datasets)}"
            )
        elif verdict is not None and verdict.status == "exploratory":
            self.errors.append(
                f"verdict {h1b.verdict!r} of {where} is exploratory; expected confirmatory"
            )

    def _step(self, task: str, lower: str, upper: str, label: str, where: str, *,
              adjacent: bool) -> None:
        """A step up in method level on one training corpus: H1b's, and D4's.

        `adjacent` asks for exactly one level up, as H1b's step to the
        pretrained transformer is; without it any rise will do.
        """
        if not self._members(task, (lower, upper), where):
            return
        configs, members = self.plan.configurations, self.plan.tasks[task].members
        if configs[lower].training != configs[upper].training:
            message = (
                f"{label} in {where}: {lower!r} is trained on {configs[lower].training!r} and "
                f"{upper!r} on {configs[upper].training!r}; it varies the method level only"
            )
            # The training does not depend on the task, and H1b checks its one
            # step once per leg: report it once, however many legs there are.
            if message not in self.errors:
                self.errors.append(message)
        low, high = members[lower].level, members[upper].level
        if high <= low or (adjacent and high != low + 1):
            self.errors.append(
                f"{label} in {where} is not {'one level' if adjacent else 'a level'} up in "
                f"task {task}: {lower} {low.code} then {upper} {high.code}"
            )

    # 8-10: H2a, H2b, H2c

    def _contrast(self, task: str, baseline: str, treatment: str, label: str, where: str) -> None:
        """Two configurations of one wrapper and one level on different training data."""
        if baseline == treatment:
            self.errors.append(f"{label} in {where} compares {baseline!r} with itself")
            return
        if not self._members(task, (baseline, treatment), where):
            return
        configs, members = self.plan.configurations, self.plan.tasks[task].members
        first, second = configs[baseline], configs[treatment]
        if first.wrapper != second.wrapper and not self.unresolved & {baseline, treatment}:
            self.errors.append(
                f"{label} in {where} compares different wrappers: {baseline} ({first.wrapper}) "
                f"and {treatment} ({second.wrapper}); it varies the training data only"
            )
        if members[baseline].level != members[treatment].level:
            self.errors.append(
                f"{label} in {where} compares different levels: {baseline} "
                f"{members[baseline].level.code} and {treatment} {members[treatment].level.code}"
            )
        if first.training == second.training:
            self.errors.append(
                f"{label} in {where}: {baseline!r} and {treatment!r} are both trained on "
                f"{first.training!r}; it contrasts training data"
            )

    def _h2a(self) -> None:
        """Each pair varies the training data only, on the datasets H2a names.

        A pair of a task is computed on each dataset of hypotheses.h2a.datasets
        that lists the task. So every dataset there must list the task of some
        pair, or nothing would be computed on it, and every task with pairs
        must be listed by one of them, or its pairs would be computed nowhere.
        The datasets were added on 24 Oct 2026, when H2a was limited to EWT
        test; until then it was computed on every dataset of its task.
        """
        first: dict[str, str] = {}
        for code, pairs in self.plan.h2a.items():
            for index, pair in enumerate(pairs):
                where = f"hypotheses.h2a.{code}[{index}]"
                if pair.id in first:
                    self.errors.append(
                        f"duplicate pair id {pair.id!r} in {where}; first used in {first[pair.id]}"
                    )
                else:
                    first[pair.id] = where
                self._contrast(code, pair.baseline, pair.treatment, f"pair {pair.id!r}", where)
        datasets, where = self.plan.datasets, "hypotheses.h2a.datasets"
        tasks = [code for code, pairs in self.plan.h2a.items() if pairs]
        for index, key in enumerate(self.plan.h2a_datasets):
            dataset = datasets.get(key)
            if dataset is None:
                self.errors.append(
                    f"unknown dataset {key!r} in {where}[{index}]; expected one of {list(datasets)}"
                )
            elif not set(dataset.tasks) & set(tasks):
                self.errors.append(
                    f"dataset {key!r} in {where}[{index}] lists none of the tasks of the H2a "
                    f"pairs {tasks}; no pair would be computed on it"
                )
        named = [datasets[key] for key in self.plan.h2a_datasets if key in datasets]
        for code in tasks:
            if not any(code in dataset.tasks for dataset in named):
                self.errors.append(
                    f"no dataset in {where} lists task {code}; the pairs of "
                    f"hypotheses.h2a.{code} would be computed on no dataset"
                )

    def _h2b(self) -> None:
        """Each transition covers exactly the configurations of its task's H1a chains.

        The criterion subtracts Delta_step,max, the largest step of those
        chains, from each model's loss out of genre: a model outside them would
        be measured against steps it takes no part in. Until 23 Oct 2026 a
        transition could list others too (supar_con for constituents); since
        the plan change of that day it is refused, and such a model is
        reported descriptively.
        """
        for code, transition in self.plan.h2b.items():
            where = f"hypotheses.h2b.{code}"
            for side, key, status in (
                ("source", transition.source, "exploratory"),
                ("target", transition.target, "confirmatory"),
            ):
                dataset = self._dataset(key, code, f"{where}.{side}")
                # An unknown status is reported by _datasets.
                if dataset is None or dataset.status not in _STATUSES:
                    continue
                if dataset.status != status:
                    self.errors.append(
                        f"{side} {key!r} of {where} is {dataset.status}; expected {status}"
                    )
            self._members(code, transition.configs, f"{where}.configs")
            members = self.plan.tasks[code].members
            in_chains: list[str] = []
            for chain in self.plan.h1a.get(code, ()):
                # A chain member outside the task is reported by _h1a.
                in_chains.extend(
                    key for key in chain.configs if key in members and key not in in_chains
                )
            missing = [key for key in in_chains if key not in transition.configs]
            if missing:
                self.errors.append(
                    f"{where}.configs lacks H1a member(s) {missing}; H2b covers exactly "
                    f"the configurations of the task's H1a chains"
                )
            # A configuration outside the task is reported by _members above.
            extra = [
                key for key in transition.configs if key in members and key not in in_chains
            ]
            if extra:
                self.errors.append(
                    f"{where}.configs has {extra}, which no H1a chain of task {code} holds; "
                    f"H2b covers exactly the configurations of the task's H1a chains"
                )

    def _h2c(self) -> None:
        for code, contrast in self.plan.h2c.items():
            where = f"hypotheses.h2c.{code}"
            self._contrast(code, contrast.baseline, contrast.treatment, "the H2c contrast", where)
            seen: set[str] = set()
            for index, test in enumerate(contrast.tests):
                test_where = f"{where}.tests[{index}]"
                self._dataset(test.dataset, code, test_where)
                if test.dataset in seen:
                    self.errors.append(
                        f"dataset {test.dataset!r} is tested twice in {where}.tests"
                    )
                seen.add(test.dataset)
                if test.decision not in _DECISIONS:
                    self.errors.append(
                        f"unknown decision {test.decision!r} in {test_where}; "
                        f"expected one of {list(_DECISIONS)}"
                    )

    # 12-13 and D4: descriptive sections (11 was D1, dropped on 23 Oct 2026)

    def _d2(self) -> None:
        d2, where = self.plan.d2, "descriptive.d2"
        configs = self.plan.configurations
        if d2.task != "con":
            self.errors.append(
                f"unknown task {d2.task!r} in {where}.task; expected one of ['con']"
            )
        self._members("con", d2.configs, f"{where}.configs")
        self._members("con", d2.versus, f"{where}.versus")
        both = [key for key in d2.configs if key in d2.versus]
        if both:
            self.errors.append(f"{both} in both {where}.configs and {where}.versus")
        for index, key in enumerate(d2.datasets):
            self._dataset(key, "con", f"{where}.datasets[{index}]")
        sample = d2.sample
        if sample.sentences < 1:
            self.errors.append(f"{where}.sample.sentences must be >= 1, not {sample.sentences}")
        self._seed(sample.seed, f"{where}.sample.seed")
        for key in d2.configs:
            if key not in configs:
                continue
            cap = configs[key].init.get("max_sentence_length")
            if cap is None:
                self.errors.append(
                    f"configurations.{key}.init has no max_sentence_length for "
                    f"{where}.sample.max_tokens {sample.max_tokens} to equal"
                )
            elif cap != sample.max_tokens:
                self.errors.append(
                    f"{where}.sample.max_tokens {sample.max_tokens} differs from "
                    f"configurations.{key}.init.max_sentence_length {cap}"
                )
        if sample.drawn_from not in _D2_DRAWN_FROM:
            self.errors.append(
                f"unknown drawn_from {sample.drawn_from!r} in {where}.sample; "
                f"expected one of {list(_D2_DRAWN_FROM)}"
            )
        for report in d2.report:
            if report not in _D2_REPORTS:
                self.errors.append(
                    f"unknown report {report!r} in {where}.report; "
                    f"expected one of {list(_D2_REPORTS)}"
                )

    def _d3(self) -> None:
        d3, where = self.plan.d3, "descriptive.d3"
        if d3.task != "dep":
            self.errors.append(
                f"unknown task {d3.task!r} in {where}.task; expected one of ['dep']"
            )
        if self._members("dep", (d3.config,), f"{where}.config"):
            annotation = self._annotation(d3.config)
            if annotation == "ud":
                self.errors.append(
                    f"{where}.config {d3.config!r} is trained on UD data; D3 compares a "
                    f"configuration trained on another scheme with UD-trained ones"
                )
        self._members("dep", d3.versus, f"{where}.versus")
        if d3.config in d3.versus:
            self.errors.append(
                f"{where}.versus lists {d3.config!r}, the configuration it is compared with"
            )
        for key in d3.versus:
            annotation = self._annotation(key)
            if annotation is not None and annotation != "ud":
                self.errors.append(
                    f"{where}.versus member {key!r} is trained on {annotation} data; "
                    f"expected a UD-trained configuration"
                )
        for metric in d3.metrics:
            if metric not in _D3_METRICS:
                self.errors.append(
                    f"unknown metric {metric!r} in {where}.metrics; "
                    f"expected one of {list(_D3_METRICS)}"
                )

    def _d4(self) -> None:
        """The step H1a leaves out: its upper configuration is in no H1a chain.

        The brief keeps spacy_ewt out of the POS chain because the literature
        predicts L1 ~ L2 there, and reports its step from nltk_ewt as D4
        instead. A chain that took spacy_ewt back would test the step D4 only
        describes, so the two cannot both hold it.
        """
        d4, where = self.plan.d4, "descriptive.d4"
        if d4.task not in TASKS:
            self.errors.append(
                f"unknown task {d4.task!r} in {where}.task; expected one of {list(TASKS)}"
            )
        elif d4.baseline == d4.treatment:
            self.errors.append(f"{where} compares {d4.baseline!r} with itself")
        else:
            self._step(d4.task, d4.baseline, d4.treatment, "the D4 step", where, adjacent=False)
            for index, chain in enumerate(self.plan.h1a.get(d4.task, ())):
                if d4.treatment in chain.configs:
                    self.errors.append(
                        f"{d4.treatment!r} of {where} is in chain {chain.id!r} "
                        f"(hypotheses.h1a.{d4.task}[{index}]); D4 is the step no H1a chain holds"
                    )

    # restrictions of configurations to datasets (24 Oct 2026)

    def _restrictions(self) -> None:
        """A configuration restricted to some datasets keeps every one a section needs it on.

        configurations.<key>.datasets names the datasets a configuration is
        scored on; without it, it is scored on every dataset of its tasks. A
        dataset named there must list a task the configuration is a member
        of, or nothing of it would be scored there, and each of its tasks
        must keep a dataset, or its membership of that task would be a
        setting nothing reads.

        A restriction may not leave out a dataset a section needs the
        configuration on: the members of an H1a chain on every dataset of the
        task (H1a is computed on all of them), the H1b step on its datasets,
        an H2a pair on hypotheses.h2a.datasets, an H2b transition on its
        source and target, H2c on the datasets of its tests, D2 on its
        datasets, and a configuration another takes as an input wherever that
        one is scored (it runs there as its part). D3 and D4 name no datasets:
        they are computed on the datasets of their task where every one of
        their members is scored (Plan.d3_datasets, Plan.d4_datasets), and a
        restriction may not leave them none. The rule was added on 24 Oct
        2026, when the owner restricted nltk_pre and spacy_sm to EWT test.
        """
        plan = self.plan
        configs, datasets = plan.configurations, plan.datasets
        tasks_of = {
            key: [code for code, task in plan.tasks.items() if key in task.members]
            for key in configs
        }
        scored: dict[str, set[str]] = {}
        for key, config in configs.items():
            own = {
                dataset for dataset, spec in datasets.items()
                if set(spec.tasks) & set(tasks_of[key])
            }
            scored[key] = own if config.datasets is None else own & set(config.datasets)
        restricted = {key: config.datasets for key, config in configs.items()
                      if config.datasets is not None}

        for key, listed in restricted.items():
            where = f"configurations.{key}.datasets"
            for index, name in enumerate(listed):
                spec = datasets.get(name)
                if spec is None:
                    self.errors.append(
                        f"unknown dataset {name!r} in {where}[{index}]; "
                        f"expected one of {list(datasets)}"
                    )
                elif not set(spec.tasks) & set(tasks_of[key]):
                    self.errors.append(
                        f"dataset {name!r} in {where}[{index}] lists none of the tasks of "
                        f"{key!r} {tasks_of[key]}; nothing of it would be scored there"
                    )
            for code in tasks_of[key]:
                if not any(code in datasets[name].tasks for name in listed if name in datasets):
                    self.errors.append(
                        f"{where} {list(listed)} names no dataset of task {code}; {key!r} is "
                        f"a member of tasks.{code} and would be scored on none of its datasets"
                    )
        if not restricted:
            return

        needs: list[tuple[str, str, str]] = []

        def need(keys: Iterable[str], dataset: str, section: str) -> None:
            needs.extend((key, dataset, section) for key in keys)

        for code, chains in plan.h1a.items():
            for index, chain in enumerate(chains):
                for dataset, spec in datasets.items():
                    if code in spec.tasks:
                        need(chain.configs, dataset,
                             f"hypotheses.h1a.{code}[{index}] (chain {chain.id!r}, computed on "
                             f"every dataset of task {code})")
        h1b = plan.h1b
        for dataset in h1b.datasets:
            spec = datasets.get(dataset)
            if spec is not None and any(leg.task in spec.tasks for leg in h1b.legs):
                need((h1b.baseline, h1b.treatment), dataset, "hypotheses.h1b")
        for code, pairs in plan.h2a.items():
            for dataset in plan.h2a_datasets_of(code):
                for index, pair in enumerate(pairs):
                    need((pair.baseline, pair.treatment), dataset,
                         f"hypotheses.h2a.{code}[{index}] (pair {pair.id!r}, computed on "
                         f"hypotheses.h2a.datasets)")
        for code, transition in plan.h2b.items():
            for dataset in (transition.source, transition.target):
                need(transition.configs, dataset, f"hypotheses.h2b.{code}")
        for code, contrast in plan.h2c.items():
            for index, test in enumerate(contrast.tests):
                need((contrast.baseline, contrast.treatment), test.dataset,
                     f"hypotheses.h2c.{code}.tests[{index}]")
        for dataset in plan.d2.datasets:
            need((*plan.d2.configs, *plan.d2.versus), dataset, "descriptive.d2")
        for key, config in configs.items():
            for name, target in config.inputs.items():
                for dataset in sorted(scored[key], key=list(datasets).index):
                    need((target,), dataset,
                         f"configurations.{key}.inputs.{name} ({target!r} runs wherever "
                         f"{key!r} is scored)")

        reported: set[tuple[str, str, str]] = set()
        for key, dataset, section in needs:
            # An unknown configuration or dataset is reported by the rules
            # that read the section.
            if key not in restricted or dataset not in datasets or dataset in scored[key]:
                continue
            if (key, dataset, section) in reported:
                continue
            reported.add((key, dataset, section))
            self.errors.append(
                f"configurations.{key}.datasets {list(restricted[key])} leaves out {dataset!r}, "
                f"where {section} needs {key!r}"
            )

        for section, task, keys in (
            ("descriptive.d3", plan.d3.task, (plan.d3.config, *plan.d3.versus)),
            ("descriptive.d4", plan.d4.task, (plan.d4.baseline, plan.d4.treatment)),
        ):
            # A task or member that is not one is reported by _d3 and _d4.
            members = plan.tasks[task].members if task in plan.tasks else {}
            if not all(key in members and key in configs for key in keys):
                continue
            narrowed = [key for key in keys if key in restricted]
            common = [
                dataset for dataset, spec in datasets.items()
                if task in spec.tasks and all(dataset in scored[key] for key in keys)
            ]
            if narrowed and not common:
                self.errors.append(
                    f"{section} is computed on the datasets of task {task} where every one of "
                    f"its members is scored, and the restrictions of {narrowed} "
                    f"(configurations.<key>.datasets) leave it none"
                )

    # 14: roles

    def _roles(self) -> None:
        """The roles each task declares equal the roles the plan's sections imply.

        Declared and derived both, because each is read by a different reader:
        the roles in `tasks` are what a person scans to see what a configuration
        is for, the sections are what the statistics compute. Requiring equality
        keeps the two from telling different stories.
        """
        plan = self.plan
        implied: dict[tuple[str, str], set[str]] = {}

        def imply(task: str, keys: Iterable[str], role: str) -> None:
            for key in keys:
                implied.setdefault((task, key), set()).add(role)

        for code, chains in plan.h1a.items():
            for chain in chains:
                imply(code, chain.configs, "h1a")
        for leg in plan.h1b.legs:
            imply(leg.task, (plan.h1b.baseline, plan.h1b.treatment), "h1b")
        for code, pairs in plan.h2a.items():
            for pair in pairs:
                imply(code, (pair.baseline, pair.treatment), "h2a")
        for code, transition in plan.h2b.items():
            imply(code, transition.configs, "h2b")
        for code, contrast in plan.h2c.items():
            imply(code, (contrast.baseline, contrast.treatment), "h2c")
        imply(plan.d2.task, plan.d2.configs, "d2")
        imply(plan.d3.task, (plan.d3.config,), "d3")
        imply(plan.d4.task, (plan.d4.baseline, plan.d4.treatment), "d4")

        for code, task in plan.tasks.items():
            for key, member in task.members.items():
                declared = set(member.roles) - {"descriptive"}
                derived = implied.get((code, key), set())
                where = f"tasks.{code}.configs.{key}.roles"
                missing = sorted(derived - declared)
                extra = sorted(declared - derived)
                if missing:
                    self.errors.append(f"{where} lacks {missing}, which the plan's sections imply")
                if extra:
                    self.errors.append(
                        f"{where} has {extra}, which no hypothesis or descriptive section implies"
                    )

    # 15: statistics

    def _statistics(self) -> None:
        statistics, where = self.plan.statistics, "statistics"
        if statistics.scale not in _SCALES:
            self.errors.append(
                f"unknown scale {statistics.scale!r} in {where}.scale; "
                f"expected one of {list(_SCALES)}"
            )
        delta = statistics.delta
        if delta <= 0:
            self.errors.append(f"{where}.delta must be > 0, not {delta}")
        sensitivity = statistics.delta_sensitivity
        if not sensitivity:
            self.errors.append(f"{where}.delta_sensitivity is empty")
        if len(set(sensitivity)) != len(sensitivity):
            self.errors.append(f"{where}.delta_sensitivity repeats a value: {list(sensitivity)}")
        for value in sensitivity:
            if value <= 0:
                self.errors.append(f"{where}.delta_sensitivity value {value} must be > 0")
        if delta in sensitivity:
            self.errors.append(
                f"{where}.delta_sensitivity {list(sensitivity)} contains delta {delta}; "
                f"a sensitivity value is an alternative to it"
            )
        bootstrap = statistics.bootstrap
        if bootstrap.method not in _BOOTSTRAP_METHODS:
            self.errors.append(
                f"unknown method {bootstrap.method!r} in {where}.bootstrap.method; "
                f"expected one of {list(_BOOTSTRAP_METHODS)}"
            )
        if bootstrap.replicates < _MIN_REPLICATES:
            self.errors.append(
                f"{where}.bootstrap.replicates must be >= {_MIN_REPLICATES}, "
                f"not {bootstrap.replicates}"
            )
        if not 0.5 < bootstrap.confidence < 1:
            self.errors.append(
                f"{where}.bootstrap.confidence must lie in (0.5, 1), not {bootstrap.confidence}"
            )
        self._seed(bootstrap.seed, f"{where}.bootstrap.seed")
        if statistics.unit_sensitivity not in _UNIT_IDS:
            self.errors.append(
                f"unknown unit {statistics.unit_sensitivity!r} in {where}.unit_sensitivity; "
                f"expected one of {list(_UNIT_IDS)}"
            )
        for decision, rule in statistics.decisions.items():
            if rule != _DECISION_RULES[decision]:
                self.errors.append(
                    f"unknown rule {rule!r} in {where}.decisions.{decision}; "
                    f"expected one of [{_DECISION_RULES[decision]!r}]"
                )
        subset = statistics.subset_sensitivity
        if subset is not None and subset not in _SUBSET_SENSITIVITIES:
            self.errors.append(
                f"unknown subset {subset!r} in {where}.subset_sensitivity; "
                f"expected one of {list(_SUBSET_SENSITIVITIES)}"
            )
        elif subset is not None:
            # A setting must have something to apply to: a Delta between two
            # configurations of a task scored by EVALB.
            evalb = [code for code, task in self.plan.tasks.items() if "evalb" in task.metric.options]
            if not any(code in self.plan.h1a or code in self.plan.h2b for code in evalb):
                self.errors.append(
                    f"{where}.subset_sensitivity {subset!r} is computed for the H1a steps and "
                    f"the H2b Delta_step,max of a task scored by EVALB ({evalb}), and the plan "
                    f"has none"
                )


# --- phase 3: strict ---------------------------------------------------------


def _runnable(plan: Plan, source: Path) -> Plan:
    """The plan a run follows: no pending configuration left in it.

    A pending configuration that is not optional is part of the design, so a
    run without it would answer a different question: refused. An optional one
    is descriptive only (phase 2 makes sure nothing compares against it or is
    built from it), so it is dropped from the configurations and from every
    task, and the run goes on without it.
    """
    pending = plan.pending()
    blocking = [key for key in pending if not plan.configurations[key].optional]
    if blocking:
        raise PlanError(
            source, [_pending_message(key, plan.configurations[key].pending) for key in blocking]
        )
    if not pending:
        return plan
    dropped = set(pending)
    return replace(
        plan,
        configurations=MappingProxyType({
            key: config for key, config in plan.configurations.items() if key not in dropped
        }),
        tasks=MappingProxyType({
            code: replace(task, members=MappingProxyType({
                key: member for key, member in task.members.items() if key not in dropped
            }))
            for code, task in plan.tasks.items()
        }),
    )
