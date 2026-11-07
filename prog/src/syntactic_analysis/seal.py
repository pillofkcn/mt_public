"""The seal on the confirmatory datasets, and the one way a stage opens a dataset.

plan/v1.yaml marks GENTLE and MCTB_en confirmatory: the hypotheses were
written before anything was measured on them, and what they say is only a test
if it stays that way until the plan is locked (step 8 of version 1.0). Until
then they may be downloaded, hashed and counted - scripts/fetch_data.py does
exactly that - and nothing else.

open_dataset() is how a stage gets a dataset to score. An exploratory dataset
opens freely. A confirmatory one opens only with confirmatory=True, which a
stage passes when it is started with --confirmatory, and only after
verify_lock() has found that plan/LOCK.json holds:

  * plan_sha256  equals the digest of plan/v1.yaml, and of the plan the
                 caller loaded: plan.plan_sha256, the SHA-256 of its bytes
                 with CRLF read as LF, so a CRLF checkout of the locked plan
                 is the locked plan, as it is to git;
  * commit       is a full commit hash in this repository, and the code is
                 unchanged since it: `git diff --quiet <commit> -- src scripts
                 plan/v1.yaml` succeeds in the project root, and git status
                 shows nothing uncommitted or untracked under those paths;
  * data         maps each confirmatory dataset to the SHA-256 of each of its
                 local files (Dataset.files), and every one matches; since
                 25 Oct 2026, before the lock, also each exploratory dataset,
                 which the confirmatory run scores again (dataset_files): EWT
                 test by its file under data/, WSJ 01 by the 100 treebank
                 files its gold set is read from, each by its path under the
                 entry of nltk.data.path NLTK finds it in. Every dataset of
                 the plan is there and no other, no file is missing or extra,
                 and every one matches;
  * models       maps each configuration whose model this project trains
                 (build.trained_models: udpipe_ewt, nltk_ewt, nltk_arc,
                 spacy_ewt, rdr_ewt) to the SHA-256 of each file its wrapper
                 declares (BaseAnalyzer.trained_model), by its path relative
                 to the project folder; every configuration is there, every
                 file is listed (none missing, none extra) and every one
                 matches (since 24 Oct 2026);
  * downloaded_models
                 maps every configuration of the plan to the downloaded files
                 its load reads (build.downloaded_models: what each wrapper
                 declares, BaseAnalyzer.downloaded_files, and since
                 25 Oct 2026 for every configuration the data files of the
                 libraries every stage imports, build.STAGE_LIBRARIES), root
                 by root - the roots of downloads.ROOTS, each found on the
                 machine that checks the lock - and, under each root, every
                 file by its path relative to the root to its SHA-256; a
                 configuration that reads nothing downloaded would map to {}
                 (until 25 Oct 2026 the five trained here). Every configuration
                 is there, every root and every file is listed (none missing,
                 none extra) and every one matches (since 24 Oct 2026, before
                 the lock);

and LOCK.json itself is committed and unchanged since: git tracks it, and
`git diff --quiet HEAD -- plan/LOCK.json` succeeds. Step 8 commits the lock
before the first confirmatory run, so the history shows that the lock came
before any confirmatory result; a lock that exists only in the working tree
would leave no trace of when it was written. Since 23 Oct 2026.

LOCK.json also holds the rest of what step 8 of the brief records:

  * versions     a JSON object: the versions of the libraries and the models
                 at the lock (the libraries, each configuration's
                 model_version, the Stanza resources and the date of its
                 models);
  * locked_at    when the lock was written, ISO 8601 with a UTC offset
                 ("2026-10-25T09:30:00+02:00").

Their form is checked, their content is not: it is a record, and is not
compared with the running environment, whose versions each stage's
provenance records beside the lock. Since 24 Oct 2026; before, a lock that
held these two keys was refused for unknown keys.

The models are checked twice in a confirmatory run: verify_lock checks all
of them, at the start of the run and of every stage, and verify_model checks
one configuration's files again right before build() loads it (a stage
passes it as build's before_load), so a model that changed during the run
stops its stage before it is loaded. The same holds for the downloaded files
(since 24 Oct 2026): verify_lock checks every configuration's, verify_model
the files of the configuration about to be loaded, and a file that differs,
is missing or is not recorded stops the stage. model_digests and
downloaded_digests compute what `models` and `downloaded_models` hold from
the wrappers' declarations; step 8 writes the lock with them
(scripts/lock_plan.py).
A model this project trains is never trained during a build or a run: its
training script writes it before the lock (the owner's decision of
24 Oct 2026), and a missing one fails its build.

An exploratory dataset opens freely, but not through a confirmatory one's
local copy: a dataset whose path is, holds or lies inside that copy is
refused whatever its status says. plan.load() refuses such a plan already;
the check is repeated here, on resolved paths, because this is where the
data is opened.

Why "the code is unchanged since the locked commit" rather than "HEAD is the
locked commit": LOCK.json is written after the commit it names - a commit
cannot contain its own hash - and is then committed itself, so recording the
lock moves HEAD one commit past it. Requiring HEAD to equal the commit would
refuse every confirmatory run. What the seal has to guarantee is that the code
and the plan that run are the ones that were locked, so that is what it asks
git: a commit that touches only other paths (LOCK.json, documentation, the
thesis) leaves it intact, and any change to src/, scripts/ or the plan,
committed or not, breaks it.

LOCK.json does not exist before step 8, and nothing here creates it.

The GENTLE loader (evaluation.gold.load_gold_pretok) reads any CoNLL-U path
it is given; no stage script takes a gold path today, so open_dataset is the
only way one could reach GENTLE. The MCTB_en loader is private to
evaluation/mctb.py and called from here alone.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from syntactic_analysis.base import ModelNotTrainedError, TrainedModel
from syntactic_analysis.checksum import sha256_of
from syntactic_analysis.downloads import DownloadedFiles
from syntactic_analysis.plan import Dataset, Plan, plan_sha256
from syntactic_analysis.project import project_root

__all__ = [
    "SealedError", "LOCK_KEYS", "CODE_PATHS", "default_lock_path", "verify_lock", "open_dataset",
    "dataset_files", "model_files", "model_digests", "downloaded_digests", "verify_model",
]

# The keys LOCK.json holds, each required. Any other key is refused, as plan.py
# refuses one: a key nothing reads is a guarantee its author believes is
# checked. plan_sha256, commit, data, models and downloaded_models are
# verified; versions and locked_at are the record step 8 adds, checked for
# their form only (_read_lock). models and downloaded_models since
# 24 Oct 2026.
LOCK_KEYS: tuple[str, ...] = (
    "plan_sha256", "commit", "data", "models", "downloaded_models", "versions", "locked_at",
)

# The paths, relative to the project root, that must not change after the lock.
CODE_PATHS: tuple[str, ...] = ("src", "scripts", "plan/v1.yaml")

# A full commit hash: SHA-1 (40) or SHA-256 (64) hex. A short hash or a ref
# name such as HEAD would make "unchanged since the commit" a moving target.
_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class SealedError(RuntimeError):
    """A confirmatory dataset that may not be opened, and every reason why.

    Attributes:
        dataset: The dataset that was asked for, or None for verify_lock alone.
        reasons: One message per problem found.
    """

    def __init__(self, dataset: str | None, reasons: Iterable[str]) -> None:
        self.dataset = dataset
        self.reasons: tuple[str, ...] = tuple(reasons)
        super().__init__(dataset, self.reasons)

    def __str__(self) -> str:
        subject = f"dataset {self.dataset!r} is sealed" if self.dataset else "the lock does not hold"
        lines = [f"{subject}:"]
        lines.extend(f"  {number}. {reason}" for number, reason in enumerate(self.reasons, 1))
        return "\n".join(lines)


def default_lock_path(root: Path | None = None) -> Path:
    """Where the lock is recorded: <root>/plan/LOCK.json."""
    return (project_root() if root is None else Path(root)) / "plan" / "LOCK.json"


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    """git run in `root`, output captured; FileNotFoundError when git is missing."""
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, timeout=120,
    )


def _read_lock(path: Path) -> dict[str, Any]:
    """LOCK.json as a mapping, or SealedError saying why it cannot be read."""
    if not path.is_file():
        raise SealedError(None, [
            f"{path} does not exist; the plan is not locked (step 8 writes it)"
        ])
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SealedError(None, [f"{path} cannot be read: {exc}"]) from None
    if not isinstance(lock, dict):
        raise SealedError(None, [f"{path} must hold a JSON object"])
    problems = [
        f"unknown key {key!r} in {path.name}; expected one of {list(LOCK_KEYS)}"
        for key in lock if key not in LOCK_KEYS
    ]
    problems += [f"missing key {key!r} in {path.name}" for key in LOCK_KEYS if key not in lock]
    if "versions" in lock and (not isinstance(lock["versions"], dict) or not lock["versions"]):
        problems.append(
            f"versions in {path.name} must be a non-empty JSON object (the library and "
            f"model versions at the lock)"
        )
    if "locked_at" in lock and not _is_timestamp(lock["locked_at"]):
        problems.append(
            f"locked_at in {path.name} must be an ISO 8601 date and time with a UTC offset, "
            f"not {lock['locked_at']!r}"
        )
    if problems:
        raise SealedError(None, problems)
    return lock


def _is_timestamp(value: Any) -> bool:
    """An ISO 8601 date and time with a UTC offset, as datetime.fromisoformat reads it."""
    if not isinstance(value, str):
        return False
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return False
    return moment.tzinfo is not None


def _lock_committed(root: Path, path: Path) -> list[str]:
    """Why the lock at `path` is not the committed one, or [] if it is."""
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return [
            f"{path} is outside the project at {root}, so git cannot show that the lock "
            f"was committed before the run"
        ]
    try:
        tracked = _git(root, "ls-files", "--error-unmatch", "--", relative)
        if tracked.returncode != 0:
            return [
                f"{relative} is not committed; step 8 commits the lock before the first "
                f"confirmatory run, so the history shows it came before the results"
            ]
        diff = _git(root, "diff", "--quiet", "HEAD", "--", relative)
    except FileNotFoundError:
        return ["git is not installed, so the lock cannot be shown to be committed"]
    if diff.returncode == 1:
        return [f"{relative} differs from HEAD; a run checks the committed lock"]
    if diff.returncode != 0:
        return [f"git diff of {relative} against HEAD failed: {diff.stderr.strip()}"]
    return []


def _code_unchanged(root: Path, commit: Any) -> list[str]:
    """Why the code under CODE_PATHS differs from `commit`, or [] if it does not."""
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        return [f"commit {commit!r} in LOCK.json is not a full commit hash"]
    try:
        exists = _git(root, "cat-file", "-e", f"{commit}^{{commit}}")
        if exists.returncode != 0:
            return [f"commit {commit} in LOCK.json is not a commit in the repository at {root}"]
        diff = _git(root, "diff", "--quiet", commit, "--", *CODE_PATHS)
        status = _git(root, "status", "--porcelain", "--untracked-files=all", "--", *CODE_PATHS)
    except FileNotFoundError:
        return ["git is not installed, so the code cannot be compared with the locked commit"]
    problems = []
    if diff.returncode == 1:
        problems.append(
            f"the code differs from the locked commit {commit[:12]} under {list(CODE_PATHS)}"
        )
    elif diff.returncode != 0:
        problems.append(f"git diff against {commit[:12]} failed: {diff.stderr.strip()}")
    if status.returncode != 0:
        problems.append(f"git status failed: {status.stderr.strip()}")
    elif status.stdout.strip():
        changed = len(status.stdout.strip().splitlines())
        problems.append(
            f"{changed} uncommitted or untracked path(s) under {list(CODE_PATHS)}"
        )
    return problems


def dataset_files(dataset: Dataset, root: str | Path | None = None) -> dict[str, Path]:
    """The files of one dataset as LOCK.json's `data` records them: {recorded path: where it is}.

    A dataset with a local copy (`path`: ewt_test, gentle, mctb_en): its
    Dataset.files, relative to the project folder, under `root`
    (project_root() when None), each whether it is there or not. A dataset
    read through NLTK (`nltk_sections`: wsj_01): the treebank files its gold
    set is read from (ptb_conventions.section_files, the function that also
    names the WSJ 00 files the PCFG is induced from), each by its path
    relative to the entry of nltk.data.path NLTK finds it in, as
    `downloaded_models` records the WSJ 00 files under nltk_data: never a
    path of this machine. Since 25 Oct 2026, before the lock.

    Raises:
        LookupError: If NLTK does not find the treebank sample.
    """
    root = project_root() if root is None else Path(root)
    if dataset.nltk_sections is None:
        return {file: root / file for file in dataset.files}
    from syntactic_analysis import ptb_conventions
    from syntactic_analysis.downloads import nltk_data_files

    found: dict[str, Path] = {}
    for item in nltk_data_files(ptb_conventions.section_files(dataset.nltk_sections)):
        found.update(item.paths())
    return dict(sorted(found.items()))


def _data_unchanged(plan: Plan, root: Path, data: Any) -> list[str]:
    """Why the data of the plan's datasets differs from the digests LOCK.json records, or [].

    Every dataset of the plan must be there: the confirmatory ones as since
    23 Oct 2026, and since 25 Oct 2026 the exploratory ones too, which the
    confirmatory run scores again (dataset_files). No other key may be.
    """
    if not isinstance(data, dict):
        return ["data in LOCK.json must map dataset ids to {file: sha256}"]
    confirmatory = {
        key: dataset for key, dataset in plan.datasets.items() if dataset.status == "confirmatory"
    }
    problems = [
        f"LOCK.json records data of {key!r}, which is not a dataset of the plan"
        for key in data if key not in plan.datasets
    ]
    for key, dataset in confirmatory.items():
        recorded = data.get(key)
        if not isinstance(recorded, dict):
            problems.append(f"LOCK.json records no digests for confirmatory dataset {key!r}")
            continue
        if sorted(recorded) != sorted(dataset.files):
            problems.append(
                f"LOCK.json records the files {sorted(recorded)} for {key!r}; "
                f"the plan's are {sorted(dataset.files)}"
            )
            continue
        for file in dataset.files:
            expected, local = recorded[file], root / file
            if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
                problems.append(f"the digest of {file} in LOCK.json is not a SHA-256")
            elif not local.is_file():
                problems.append(f"{file} is missing; run scripts/fetch_data.py")
            else:
                actual = sha256_of(local)
                if actual != expected:
                    problems.append(
                        f"{file} has sha256 {actual}, LOCK.json records {expected}"
                    )
    for key, dataset in plan.datasets.items():
        if key not in confirmatory:
            problems += _exploratory_data_problems(key, dataset, data.get(key), root)
    return problems


def _exploratory_data_problems(key: str, dataset: Dataset, recorded: Any, root: Path) -> list[str]:
    """Why the files of one exploratory dataset differ from what LOCK.json records, or [].

    Its files are dataset_files: none may be missing from the record or
    extra in it, and each must have the recorded digest. A file is named by
    its recorded path, with "nltk_data/" in front for a dataset read through
    NLTK. Since 25 Oct 2026, before the lock.
    """
    if not isinstance(recorded, dict):
        return [
            f"LOCK.json records no digests for exploratory dataset {key!r}, which the "
            f"confirmatory run scores again"
        ]
    try:
        files = dataset_files(dataset, root)
    except LookupError:
        return [
            f"NLTK does not find the treebank sample (corpora/treebank) that {key!r} is read "
            f"from; setup.sh downloads it"
        ]
    through_nltk = dataset.nltk_sections is not None

    def shown(name: str) -> str:
        return f"nltk_data/{name}" if through_nltk else name

    problems = []
    for name in recorded:
        if name in files:
            continue
        if through_nltk:
            problems.append(
                f"LOCK.json records {shown(name)} for {key!r}, which NLTK does not find among "
                f"the files of WSJ section(s) {', '.join(dataset.nltk_sections)}"
            )
        else:
            problems.append(
                f"LOCK.json records {name} for {key!r}; the plan's files are {sorted(files)}"
            )
    for name, path in files.items():
        if name not in recorded:
            problems.append(
                f"{shown(name)} is a file of {key!r} that LOCK.json does not record"
            )
            continue
        expected = recorded[name]
        if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
            problems.append(f"the digest of {shown(name)} in LOCK.json is not a SHA-256")
        elif not path.is_file():
            problems.append(f"{shown(name)} of {key!r} is missing; setup.sh downloads it")
        else:
            actual = sha256_of(path)
            if actual != expected:
                problems.append(
                    f"{shown(name)} has sha256 {actual}, LOCK.json records {expected}"
                )
    return problems


def _relative(path: Path) -> str:
    """A declared model path as LOCK.json records it: relative to the project folder.

    Raises:
        ValueError: If the path is outside the project folder.
    """
    try:
        return Path(path).resolve().relative_to(project_root().resolve()).as_posix()
    except ValueError:
        raise ValueError(
            f"{path} is outside the project folder, so LOCK.json cannot record it"
        ) from None


def model_files(model: TrainedModel, root: str | Path | None = None) -> dict[str, Path]:
    """The files of a declared model, by their path relative to the project folder.

    Each declared file whether it is there or not, and every file now under a
    declared directory, at any depth; a declared directory that is not there,
    or holds no file, adds nothing (_missing_directories finds it). The
    relative paths are what LOCK.json records; the values are where the files
    are under `root` (project_root() when None), which is where they are
    checked.

    Raises:
        ValueError: If a declared path is outside the project folder.
    """
    root = project_root() if root is None else Path(root)
    found = {_relative(path): root / _relative(path) for path in model.files}
    for directory in model.directories:
        base = root / _relative(directory)
        if base.is_dir():
            for item in sorted(base.rglob("*")):
                if item.is_file():
                    found[item.relative_to(root).as_posix()] = item
    return dict(sorted(found.items()))


def _missing_directories(model: TrainedModel, root: Path) -> list[str]:
    """The declared directories of a model that are not under `root` or hold no file.

    The rule of TrainedModel.missing(), which load_model applies through
    require_trained: an empty spaCy model-best/ is a model that is not
    trained, not a model without files, so model_digests refuses it before a
    lock records it, and a lock check reports it as missing. Since 24 Oct 2026.
    """
    absent = []
    for directory in model.directories:
        base = root / _relative(directory)
        if not base.is_dir() or not any(item.is_file() for item in base.rglob("*")):
            absent.append(f"{_relative(directory)}/")
    return absent


def model_digests(plan: Plan, *, root: str | Path | None = None) -> dict[str, dict[str, str]]:
    """What LOCK.json records under `models`: {configuration: {file: sha256}}.

    For every configuration of the plan whose model this project trains
    (build.trained_models), the SHA-256 of every file its wrapper declares
    (model_files), by its path relative to the project folder, read under
    `root` (project_root() when None). Step 8 writes the lock with it; nothing
    here writes a file. Since 24 Oct 2026.

    Raises:
        ModelNotTrainedError: If a declared file is missing, or a declared
            directory is missing or holds no file; it names every one, and
            the script that trains it.
        ValueError: If a declared path is outside the project folder.
    """
    from syntactic_analysis.build import trained_models

    root = project_root() if root is None else Path(root)
    digests: dict[str, dict[str, str]] = {}
    absent: list[str] = []
    for key, model in trained_models(plan).items():
        files = model_files(model, root)
        missing = [rel for rel, path in files.items() if not path.is_file()]
        missing += _missing_directories(model, root)
        if missing:
            absent.append(f"{key}: {', '.join(missing)} (train it with {model.script})")
            continue
        digests[key] = {rel: sha256_of(path) for rel, path in files.items()}
    if absent:
        raise ModelNotTrainedError(
            "the models this project trains are not all on disk, so the lock cannot "
            "record them: " + "; ".join(absent)
        )
    return digests


def _model_problems(key: str, model: TrainedModel, recorded: Any, root: Path) -> list[str]:
    """Why the files of one model differ from what LOCK.json records for it, or []."""
    if not isinstance(recorded, dict) or not recorded:
        return [
            f"LOCK.json records no model files for {key!r}, whose model {model.script} "
            f"trains"
        ]
    try:
        current = model_files(model, root)
        missing_directories = _missing_directories(model, root)
    except ValueError as exc:
        return [f"{key}: {exc}"]
    declared = set(current)
    problems = []
    for directory in missing_directories:
        problems.append(f"{directory} of {key!r} is missing")
    for rel, digest in recorded.items():
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            problems.append(f"the digest of {rel} of {key!r} in LOCK.json is not a SHA-256")
        elif rel not in declared:
            inside = any(
                rel.startswith(f"{_relative(directory)}/") for directory in model.directories
            )
            problems.append(
                f"{rel} of {key!r} is missing" if inside else
                f"LOCK.json records {rel} for {key!r}, which is not a file of its model"
            )
        elif not current[rel].is_file():
            problems.append(f"{rel} of {key!r} is missing")
        else:
            actual = sha256_of(current[rel])
            if actual != digest:
                problems.append(f"{rel} has sha256 {actual}, LOCK.json records {digest}")
    for rel in current:
        if rel not in recorded:
            problems.append(
                f"{rel} is a file of the model of {key!r} that LOCK.json does not record"
            )
    return problems


def _models_unchanged(plan: Plan, root: Path, models: Any) -> list[str]:
    """Why the models this project trains differ from what LOCK.json records, or [].

    Every configuration whose model this project trains must be there, with
    exactly the files its wrapper declares and their digests; no other
    configuration may be.
    """
    from syntactic_analysis.build import trained_models

    if not isinstance(models, dict):
        return ["models in LOCK.json must map configuration keys to {file: sha256}"]
    declared = trained_models(plan)
    problems = [
        f"LOCK.json records model files of {key!r}, which is not a configuration of the "
        f"plan whose model this project trains"
        for key in models if key not in declared
    ]
    for key, model in declared.items():
        problems += _model_problems(key, model, models.get(key), root)
    return problems


# --- the downloaded files (24 Oct 2026) ---------------------------------------------


def _digest(path: Path, cache: dict[str, str]) -> str:
    """The SHA-256 of `path`, read once per check however many configurations read the file."""
    key = os.path.normcase(os.path.abspath(path))
    if key not in cache:
        cache[key] = sha256_of(path)
    return cache[key]


def _by_root(
    declared: Iterable[DownloadedFiles],
) -> tuple[dict[str, dict[str, Path]], dict[str, list[str]], list[str]]:
    """A configuration's downloaded files root by root, its declared directories, and what is missing.

    Returns ({root: {relative path: where it is}}, {root: [declared
    directories]}, [missing]): every declared file whether it is there or
    not, and every file now under a declared directory; a missing file as
    "<root>/<path>", a missing or empty directory as "<root>/<path>/".

    Raises:
        ValueError: If two declarations put two different files at one path of one root.
    """
    files: dict[str, dict[str, Path]] = {}
    directories: dict[str, list[str]] = {}
    missing: list[str] = []
    for item in declared:
        under = files.setdefault(item.root, {})
        directories.setdefault(item.root, []).extend(item.directories)
        for rel, path in item.paths().items():
            if rel in under and os.path.normcase(str(under[rel])) != os.path.normcase(str(path)):
                raise ValueError(
                    f"{item.root}/{rel} is declared twice, as two different files"
                )
            under[rel] = path
        missing += [f"{item.root}/{rel}" for rel in item.missing()]
    return files, directories, missing


def downloaded_digests(plan: Plan) -> dict[str, dict[str, dict[str, str]]]:
    """What LOCK.json records under `downloaded_models`: {configuration: {root: {file: sha256}}}.

    For every configuration of the plan, the SHA-256 of every downloaded file
    its load reads (build.downloaded_models), under its root (downloads.ROOTS)
    by its path relative to the root; {} for a configuration whose load reads
    nothing downloaded. Step 8 writes the lock with it
    (scripts/lock_plan.py); nothing here writes a file. A file several
    configurations read is read once. Since 24 Oct 2026.

    Raises:
        FileNotFoundError: If a declared file is missing, or a declared
            directory is missing or holds no file; it names every one.
        ValueError: If two declarations of one configuration put two files at
            one path.
    """
    from syntactic_analysis.build import downloaded_models

    cache: dict[str, str] = {}
    digests: dict[str, dict[str, dict[str, str]]] = {}
    absent: list[str] = []
    for key, declared in downloaded_models(plan).items():
        files, _directories, missing = _by_root(declared)
        if missing:
            absent.append(f"{key}: {', '.join(missing)}")
            continue
        digests[key] = {
            root: {rel: _digest(path, cache) for rel, path in under.items()}
            for root, under in files.items()
        }
    if absent:
        raise FileNotFoundError(
            "the downloaded files the configurations load are not all on disk, so the lock "
            "cannot record them (setup.sh downloads them): " + "; ".join(absent)
        )
    return digests


def _downloaded_problems(
    key: str, declared: Iterable[DownloadedFiles], recorded: Any, cache: dict[str, str],
) -> list[str]:
    """Why the downloaded files one configuration's load reads differ from what LOCK.json records, or []."""
    if not isinstance(recorded, dict):
        return [
            f"LOCK.json records no downloaded files for {key!r} ({{}} for a configuration "
            f"whose load reads none)"
        ]
    try:
        current, directories, missing = _by_root(declared)
    except ValueError as exc:
        return [f"{key}: {exc}"]
    problems = [f"{name} of {key!r} is missing" for name in missing]
    absent = set(missing)
    problems += [
        f"LOCK.json records files under {root} for {key!r}, whose load reads nothing there"
        for root in recorded if root not in current
    ]
    for root, files in current.items():
        listed = recorded.get(root)
        if not isinstance(listed, dict):
            problems.append(
                f"LOCK.json records no files under {root} for {key!r}, whose load reads "
                f"{len(files)} there"
            )
            continue
        for rel, digest in listed.items():
            name = f"{root}/{rel}"
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                problems.append(f"the digest of {name} of {key!r} in LOCK.json is not a SHA-256")
            elif rel not in files:
                inside = any(rel.startswith(f"{folder}/") for folder in directories[root])
                problems.append(
                    f"{name} of {key!r} is missing" if inside else
                    f"LOCK.json records {name} for {key!r}, which its load does not read"
                )
            elif name not in absent:
                actual = _digest(files[rel], cache)
                if actual != digest:
                    problems.append(f"{name} has sha256 {actual}, LOCK.json records {digest}")
        problems += [
            f"{root}/{rel} is a file the load of {key!r} reads that LOCK.json does not record"
            for rel in files if rel not in listed and f"{root}/{rel}" not in absent
        ]
    return problems


def _downloads_unchanged(plan: Plan, downloads: Any) -> list[str]:
    """Why the downloaded files of the plan's configurations differ from what LOCK.json records, or [].

    Every configuration of the plan must be there, with exactly the roots
    and the files its wrapper declares and their digests; no other key may
    be. A declaration that cannot be made is a problem of its own.
    """
    from syntactic_analysis.build import build, downloaded_files

    if not isinstance(downloads, dict):
        return ["downloaded_models in LOCK.json must map configuration keys to "
                "{root: {file: sha256}}"]
    problems = [
        f"LOCK.json records downloaded files of {key!r}, which is not a configuration of "
        f"the plan"
        for key in downloads if key not in plan.configurations
    ]
    cache: dict[str, str] = {}
    built: dict = {}
    for key in plan.configurations:
        analyzer = build(plan, key, load=False, built=built)
        try:
            declared = downloaded_files(plan, key, analyzer)
        except Exception as exc:  # any failure: the files cannot be shown to be the locked ones
            problems.append(
                f"the downloaded files of {key!r} cannot be declared: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        problems += _downloaded_problems(key, declared, downloads.get(key), cache)
    return problems


def verify_model(
    lock: Mapping[str, Any], key: str, model: TrainedModel | None,
    downloaded: Iterable[DownloadedFiles] = (), *,
    root: str | Path | None = None,
) -> None:
    """Refuse to load configuration `key` unless its model files are the ones the lock records.

    A confirmatory stage passes it to build() as before_load, with the lock
    verify_lock returned, so it runs right before each configuration is
    loaded, inputs included (since 24 Oct 2026). `model` is the model this
    project trains that the load will read (build.trained_model), None for a
    published one, which `models` does not record; the files are those under
    `root` (project_root() when None), as model_files finds them.
    `downloaded` is what build.downloaded_files declares for the load, ()
    for none: its files must be exactly the ones `downloaded_models`
    records for `key`, with the same digests (since 24 Oct 2026, before the
    lock).

    Raises:
        SealedError: Listing every file that is missing, extra or changed, or
            saying that the lock records no files for a model this project
            trains or for the configuration's downloaded files, or records
            files for a configuration or a root that has none.
    """
    root = project_root() if root is None else Path(root)
    models = lock.get("models")
    if not isinstance(models, dict):
        raise SealedError(None, ["models in LOCK.json must map configuration keys to "
                                 "{file: sha256}"])
    downloads = lock.get("downloaded_models")
    if not isinstance(downloads, dict):
        raise SealedError(None, ["downloaded_models in LOCK.json must map configuration keys "
                                 "to {root: {file: sha256}}"])
    recorded = models.get(key)
    problems: list[str] = []
    if model is None:
        if recorded is not None:
            problems.append(
                f"LOCK.json records model files of {key!r}, whose wrapper declares no "
                f"model this project trains"
            )
    else:
        problems += _model_problems(key, model, recorded, root)
    problems += _downloaded_problems(key, tuple(downloaded), downloads.get(key), {})
    if problems:
        raise SealedError(None, problems)


def verify_lock(
    plan: Plan, lock_path: str | Path | None = None, *, root: str | Path | None = None,
) -> dict[str, Any]:
    """Check that the plan, the code, the data of every dataset and the models are the locked ones.

    The plan is compared by plan.plan_sha256, which reads CRLF as LF, as
    Plan.sha256 is: the code check asks git, which ignores line endings, and
    a digest of the raw bytes would refuse a Windows checkout of a plan that
    was locked on LF (or the other way round). Since 23 Oct 2026.

    Args:
        plan:       The plan the run follows.
        lock_path:  The lock file; <root>/plan/LOCK.json when None. Wherever
                    it is, it must lie in the project and be committed.
        root:       The project root, where git runs and the plan, the code
                    and the data paths are found; project_root() when None.

    Returns the lock, read, when every check passes.

    Raises:
        SealedError: Listing every check that failed.
    """
    root = project_root() if root is None else Path(root)
    path = default_lock_path(root) if lock_path is None else Path(lock_path)
    lock = _read_lock(path)

    problems = _lock_committed(root, path)
    recorded = lock["plan_sha256"]
    plan_file = root / "plan" / "v1.yaml"
    if not isinstance(recorded, str) or not _SHA256.fullmatch(recorded):
        problems.append("plan_sha256 in LOCK.json is not a SHA-256")
    else:
        if plan.sha256 != recorded:
            problems.append(
                f"the plan loaded from {plan.path} has sha256 {plan.sha256}, "
                f"LOCK.json records {recorded}"
            )
        if not plan_file.is_file():
            problems.append(f"{plan_file} does not exist")
        else:
            actual = plan_sha256(plan_file.read_bytes())
            if actual != recorded:
                problems.append(f"{plan_file} has sha256 {actual}, LOCK.json records {recorded}")
    problems += _code_unchanged(root, lock["commit"])
    problems += _data_unchanged(plan, root, lock["data"])
    problems += _models_unchanged(plan, root, lock["models"])
    problems += _downloads_unchanged(plan, lock["downloaded_models"])
    if problems:
        raise SealedError(None, problems)
    return lock


def open_dataset(
    plan: Plan,
    dataset_id: str,
    *,
    confirmatory: bool = False,
    root: str | Path | None = None,
    lock_path: str | Path | None = None,
):
    """The gold set of one dataset of the plan, if the seal lets it be opened.

    Args:
        plan:         The plan the run follows.
        dataset_id:   A key of plan.datasets.
        confirmatory: True only in a run started with --confirmatory. A
                      confirmatory dataset is refused without it, and with it
                      opens only if verify_lock() passes.
        root:         The project root the dataset's paths are relative to;
                      project_root() when None.
        lock_path:    Passed on to verify_lock.

    Returns a GoldSet for a CoNLL-U dataset (ewt_test, gentle) and a
    ConstituencyGoldSet for a treebank (wsj_01 through NLTK, mctb_en from its
    domain files).

    Raises:
        ValueError: An unknown dataset, or one whose number of resampling units
            differs from the plan's expected_units.
        SealedError: A confirmatory dataset opened without confirmatory=True,
            or with a lock that does not hold; any other dataset whose path
            is, holds or lies inside a confirmatory dataset's local copy.
    """
    dataset = plan.datasets.get(dataset_id)
    if dataset is None:
        raise ValueError(
            f"unknown dataset {dataset_id!r}; expected one of {list(plan.datasets)}"
        )
    root = project_root() if root is None else Path(root)
    if dataset.status == "confirmatory":
        if not confirmatory:
            raise SealedError(dataset_id, [
                "it is confirmatory: it opens only after the plan is locked "
                "(plan/LOCK.json, step 8) and in a run started with --confirmatory"
            ])
        try:
            verify_lock(plan, lock_path, root=root)
        except SealedError as exc:
            raise SealedError(dataset_id, exc.reasons) from None
    else:
        reasons = _sealed_copies(plan, dataset, root)
        if reasons:
            raise SealedError(dataset_id, reasons)
    return _load(dataset, root)


def _sealed_copies(plan: Plan, dataset: Dataset, root: Path) -> list[str]:
    """Why `dataset` would read a confirmatory dataset's local copy, or [].

    Compared on resolved paths, so a link counts as what it points to, and
    with os.path.normcase, so on Windows a path that differs only in case is
    the same path.
    """
    if dataset.path is None:
        return []

    def canonical(path: str) -> Path:
        return Path(os.path.normcase((root / path).resolve()))

    mine = canonical(dataset.path)
    reasons = []
    for key, other in plan.datasets.items():
        if key == dataset.id or other.status != "confirmatory" or other.path is None:
            continue
        sealed = canonical(other.path)
        if mine == sealed or sealed in mine.parents or mine in sealed.parents:
            reasons.append(
                f"its path {dataset.path} is, holds or lies inside {other.path}, the local "
                f"copy of confirmatory dataset {key!r}"
            )
    return reasons


def _load(dataset: Dataset, root: Path):
    """The gold set of `dataset`, with its resampling units counted against the plan.

    The evaluation package is imported here rather than at module level: it
    costs over a second, and verify_lock needs none of it.
    """
    from syntactic_analysis.evaluation.gold import load_gold_pretok, load_gold_wsj
    from syntactic_analysis.evaluation.mctb import _gold_set

    expected = dataset.resampling.expected_units
    if dataset.nltk_sections is not None:
        gold = load_gold_wsj(dataset.nltk_sections)
        units = len(gold.fileids)
    elif "con" in dataset.tasks:
        gold = _gold_set([root / file for file in dataset.files])
        units = gold.n_sentences
    else:
        gold = load_gold_pretok(root / dataset.path)
        if dataset.resampling.unit == "document":
            missing = sum(1 for sentence in gold.parse.sentences if not sentence.doc_id)
            if missing:
                raise ValueError(
                    f"{dataset.path}: {missing} sentence(s) belong to no document, "
                    f"and {dataset.id} is resampled by document"
                )
            units = len({sentence.doc_id for sentence in gold.parse.sentences})
        else:
            units = gold.n_sentences
    if expected is not None and units != expected:
        raise ValueError(
            f"{dataset.id} has {units} {dataset.resampling.unit} units, "
            f"the plan expects {expected}"
        )
    return gold
