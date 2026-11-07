"""What produced a number: versions, commit, corpus checksum, configuration.

Split out of evaluation/harness.py on 9 Oct 2026. A provenance record is what
makes a published figure checkable a year later: the library versions of THAT
run, the model versions, the Python build, the platform, the git commit,
whether the tree was clean, and the SHA-256 of the gold corpus the numbers were
computed against. The measuring stages of version 1.0 write theirs beside
their counts files (syntactic_analysis.stage); this module gives them, the
training scripts and scripts/smoke_plan.py the spellings below, and
`collect_provenance` the record scripts/smoke_plan.py writes.

`git_dirty` is read from `git status --porcelain`, which reports untracked
files too. On 8 Oct 2026 that turned a run log sitting beside its own run
directory into 26 sidecars claiming uncommitted changes no tracked file had;
the fix was a .gitignore rule, not a looser check, because a provenance flag
that under-reports is worse than one that over-reports.

Three fields are spelled by this module and by nothing else, so that every
record of a run reads the same whichever stage wrote it:

  timestamp  `stage_started()` - the moment the writing stage STARTED, as
             2026-10-10T08:38:59. A stage takes it once, first thing, and uses
             that one value for every sidecar and for its report header.
  script     `script_path(__file__)` - the script's path inside the repository,
             `scripts/pos_tagging.py`.
  gold_path  `recorded_path(gold.path)` - the gold corpus as a path inside the
             repository, `data/en_ewt-ud-test.conllu`, or from its `nltk_data`
             directory on for the Penn Treebank sample. The same function
             spells every path a report prints.

Until 11 Oct 2026 each script formatted the first two for itself. The
constituency stage wrote 20261010_140640 where the others wrote ISO; two stages
recorded a bare file name and four a path; and a timestamp meant the start of
the stage in four writers and the moment of writing in two, which is how a run
came to be documented as starting a minute before the commit it ran from. The
second spelling of the same instant, the stamp in a run directory's name, was
formatted by five scripts until 12 Oct 2026; run_stamp derives it from the
first. Until 12 Oct 2026 gold_path was the absolute path on the machine that
ran, home directory included.

Until 23 Oct 2026 this module also wrote and read the `.prov.json` sidecar
beside every CSV of the stages before version 1.0 (write_provenance,
read_sidecar, with_config); they went with the table writers and readers of
evaluation/tables.py and evaluation/inputs.py.
"""
from __future__ import annotations

import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from syntactic_analysis.base import AnalysisType, BaseAnalyzer
from syntactic_analysis.evaluation.gold import GoldSet
from syntactic_analysis.project import project_root


# --- provenance --------------------------------------------------------------

_PROVENANCE_PACKAGES = (
    "nltk", "spacy", "stanza", "supar", "benepar", "ufal.udpipe",
    "torch", "scipy", "numpy", "pandas", "matplotlib", "scikit-learn",
)

# The one spelling of a timestamp in a provenance record and in a report header.
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"

# The second spelling of that instant: the stamp in the name of a run directory,
# results/run_20261010_083859.
RUN_STAMP_FORMAT = "%Y%m%d_%H%M%S"


def stage_started() -> str:
    """Now, in the one spelling a provenance record uses.

    Called once, first thing in a stage's main(). The value is what the stage
    passes to `collect_provenance` and prints as `started` in its header, so the
    two cannot disagree and both mean the same instant: before any model was
    loaded and before any sentence was parsed.
    """
    return time.strftime(TIMESTAMP_FORMAT)


def run_stamp(timestamp: str) -> str:
    """The same instant as a directory name carries it.

    2026-10-10T08:38:59 -> 20261010_083859. Derived from the value
    `stage_started()` returned rather than taken from the clock a second time,
    so the directory a run writes into and the timestamp in its sidecars cannot
    name two different seconds.
    """
    return time.strftime(RUN_STAMP_FORMAT, time.strptime(timestamp, TIMESTAMP_FORMAT))


def script_path(file: str | Path) -> str:
    """How a provenance record names the script that wrote it.

    The path inside the repository, with forward slashes: `scripts/pos_tagging.py`.
    A script outside the repository - a copy run from somewhere else - is named
    by its file name, since there is no repository-relative path to give.
    """
    path = Path(file).resolve()
    try:
        return path.relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return path.name


def recorded_path(path: str | Path) -> str:
    """How a provenance record stores a path, and how a report prints one.

    Three rules, in this order:

    1. a path inside the repository is given relative to its root, with forward
       slashes: `results/run_20261012_090000/csv/pos_scores.csv`;
    2. a path outside it that has a component named `nltk_data` is given from
       that component on: `nltk_data/corpora/treebank/combined`. That is where
       the Penn Treebank sample lives, and everything before it is one
       machine's home directory;
    3. any other path is given resolved and absolute, with forward slashes.

    It is the rule `script_path` states for a script, extended to every path a
    run records or prints: the gold corpus, a report, an output file. The two
    are kept apart because they differ in the third case, where a script is
    named by its file name. Never raises: a path that cannot be resolved is
    returned as given, with forward slashes.
    """
    try:
        resolved = Path(path).resolve()
    except (OSError, RuntimeError, ValueError):
        return str(path).replace("\\", "/")
    try:
        return resolved.relative_to(project_root().resolve()).as_posix()
    except ValueError:
        pass
    parts = resolved.parts
    if "nltk_data" in parts:
        start = len(parts) - 1 - parts[::-1].index("nltk_data")
        return "/".join(parts[start:])
    return resolved.as_posix()


def _git_describe() -> tuple[str, bool]:
    """Return (commit sha, dirty flag). Empty sha when git is unavailable."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root(), capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root(), capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return sha, bool(status)
    except (OSError, subprocess.SubprocessError):
        return "", False


@dataclass(frozen=True)
class Provenance:
    """Everything needed to reproduce, or to explain, a set of numbers.

    The four gold fields are None for a record written without a gold corpus,
    and a JSON record stores that as null. Until 16 Oct 2026 such a record held
    "" and 0, and a gold_sentences of 0 read as a corpus of no sentences rather
    than as no corpus at all.
    """

    timestamp: str
    script: str
    git_sha: str
    git_dirty: bool
    python_version: str
    platform: str
    packages: dict[str, str]
    models: dict[str, str]
    gold_path: str | None
    gold_sha256: str | None
    gold_sentences: int | None
    gold_tokens: int | None
    config: dict[str, object]


def collect_provenance(
    *,
    script: str,
    gold: GoldSet | None = None,
    analyzers: Iterable[BaseAnalyzer] = (),
    config: Mapping[str, object] | None = None,
    timestamp: str | None = None,
    task: AnalysisType | None = None,
) -> Provenance:
    """Gather the run's identity.

    The run log used to record the date, the gold path, the sentence count, alpha
    and the mapping flag — but not one library version, Python version or commit.
    With unpinned dependencies that made a later discrepancy impossible to
    attribute: a changed number could be a bug or a library upgrade, with no way
    to tell.

    Called once per stage, after the last model is loaded and before the first
    sentence is analysed - it reads the commit and the dirty flag, and those have
    to describe the tree the stage started from.

    `task` decides how an analyzer without a model version is recorded in
    `models`: by the training corpus of that task (`training_corpus_for`) when it
    is given, by the tool-wide string when it is not. The three measuring stages
    pass theirs, so a POS sidecar no longer names the corpus a tagger's
    constituency half was trained on; until 12 Oct 2026 every record held the
    tool-wide string.
    """
    import importlib.metadata as md

    versions: dict[str, str] = {}
    for name in _PROVENANCE_PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            continue

    models: dict[str, str] = {}
    for analyzer in analyzers:
        version = analyzer.model_version
        models[analyzer.name] = version or (
            analyzer.training_corpus_for(task) if task is not None
            else analyzer.training_corpus
        )

    sha, dirty = _git_describe()
    return Provenance(
        timestamp=timestamp or stage_started(),
        script=script,
        git_sha=sha,
        git_dirty=dirty,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        packages=versions,
        models=models,
        gold_path=recorded_path(gold.path) if gold else None,
        gold_sha256=gold.sha256 if gold else None,
        gold_sentences=gold.n_sentences if gold else None,
        gold_tokens=gold.n_tokens if gold else None,
        config=dict(config or {}),
    )
