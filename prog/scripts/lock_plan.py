"""Compute plan/LOCK.json, the lock of step 8: print it, and write it only with --write.

    python scripts/lock_plan.py --commit <hash>            # print the lock
    python scripts/lock_plan.py --commit <hash> --write    # and write plan/LOCK.json

The lock holds the keys seal.LOCK_KEYS names, each computed here:

  plan_sha256        the digest of plan/v1.yaml as Plan.sha256 takes it (CRLF
                     read as LF);
  commit             the full hash of HEAD. HEAD must be the commit named with
                     --commit (a full or a short hash, or a ref), and nothing
                     in the project folder may be uncommitted or untracked
                     (`git status --porcelain -- .` in prog/), so the lock
                     names exactly the code it was computed from;
  data               for each confirmatory dataset of the plan, the SHA-256 of
                     each of its files (Dataset.files) as scripts/fetch_data.py
                     recorded it (Dataset.digests, data/*.sha256), and each
                     local file must still have it: it is hashed
                     (checksum.sha256_of, the bytes go into the digest and
                     nowhere else). Nothing of the data is read otherwise, and
                     nothing but digests and file names is printed. Since
                     25 Oct 2026, before the lock, also for each exploratory
                     dataset, which the confirmatory run scores again, the
                     SHA-256 of each of its files (seal.dataset_files): EWT
                     test's data/en_ewt-ud-test.conllu, and the 100 treebank
                     files of WSJ 01 by their path under the nltk_data entry
                     NLTK finds them in (corpora/treebank/combined/
                     wsj_0100.mrg ...);
  models             seal.model_digests: the files of the five models this
                     project trains, by their path under prog/;
  downloaded_models  seal.downloaded_digests: the downloaded files every
                     configuration's load reads, root by root (downloads.py);
  versions           the environment as the provenance of a stage records it
                     (stage.environment: Python, its implementation, the
                     platform, the CPU and the version of every library a
                     wrapper imports), the version of Stanza's resources index
                     the Stanza models are resolved against (1.11.0) and the
                     date of those models (wrappers/stanza.py,
                     MODELS_BUILT_FOR: built for Stanza 1.10.0, release notes
                     of 23 Dec 2024). The models themselves are identified by
                     the digests of their files under models and
                     downloaded_models;
  locked_at          now, ISO 8601 with the UTC offset of this machine.

It refuses, writing nothing and with exit status 1, when plan/LOCK.json
exists already, when HEAD is not the commit named or the project folder has
uncommitted or untracked changes (checked first, before any file is read),
when a confirmatory file is missing or does not match its record, when a
file of an exploratory dataset is missing or NLTK does not find the treebank
sample, and when a model file is missing. Without --write it prints the lock
and writes nothing; with --write it also writes plan/LOCK.json (UTF-8, LF,
the keys in the order of seal.LOCK_KEYS). The lock then has to be
committed: seal.verify_lock accepts only a committed lock, whose `commit`
is the commit before it, with the code under src/, scripts/ and
plan/v1.yaml unchanged since. Step 8 of version 1.0 proposes that commit
and the tag plan-v1-lock to the owner and waits for his consent.

--root runs it on another copy of the project (a test's temporary
repository); the default is the project this script belongs to. Added
24 Oct 2026, before the lock.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

from syntactic_analysis.checksum import sha256_of
from syntactic_analysis.plan import Dataset, Plan, load
from syntactic_analysis.project import project_root
from syntactic_analysis.seal import (
    LOCK_KEYS,
    dataset_files,
    default_lock_path,
    downloaded_digests,
    model_digests,
)


class Refused(RuntimeError):
    """The lock cannot be computed as it stands, and every reason why."""

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = list(reasons)
        super().__init__("\n".join(self.reasons))


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=120)


def check_commit(root: Path, named: str) -> str:
    """The full hash of HEAD, if it is the commit `named` and the project folder is clean.

    Raises:
        Refused: If `named` is not a commit, HEAD is another one, or
            `git status --porcelain --untracked-files=all -- .` in `root`
            lists anything.
    """
    problems = []
    try:
        head = _git(root, "rev-parse", "--verify", "HEAD")
        wanted = _git(root, "rev-parse", "--verify", "--quiet", f"{named}^{{commit}}")
        status = _git(root, "status", "--porcelain", "--untracked-files=all", "--", ".")
    except FileNotFoundError:
        raise Refused(["git is not installed"]) from None
    if head.returncode != 0:
        raise Refused([f"the project at {root} is in no git repository with a commit"])
    commit = head.stdout.strip()
    if wanted.returncode != 0:
        problems.append(f"{named!r} is not a commit of the repository at {root}")
    elif wanted.stdout.strip() != commit:
        problems.append(
            f"HEAD is {commit}, not the commit named ({wanted.stdout.strip()}); the lock names "
            f"the commit it is computed at"
        )
    if status.returncode != 0:
        problems.append(f"git status failed: {status.stderr.strip()}")
    elif status.stdout.strip():
        changed = [line for line in status.stdout.splitlines() if line.strip()]
        shown = ", ".join(line[3:] for line in changed[:10])
        more = f" and {len(changed) - 10} more" if len(changed) > 10 else ""
        problems.append(
            f"{len(changed)} uncommitted or untracked path(s) in the project folder ({shown}"
            f"{more}); the lock names a commit, so everything has to be committed first"
        )
    if problems:
        raise Refused(problems)
    return commit


def _fetch_data() -> ModuleType:
    """scripts/fetch_data.py, whose records of the confirmatory data the digests come from."""
    name = "fetch_data_for_the_lock"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("fetch_data.py"))
    module = importlib.util.module_from_spec(spec)
    # Registered first: dataclasses look their module up in sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def data_digests(plan: Plan, root: Path) -> dict[str, dict[str, str]]:
    """{dataset: {file: sha256}} for every dataset of the plan, in the plan's order.

    A confirmatory dataset: from fetch_data.py's records, each file checked.
    The file paths are the plan's (Dataset.files, relative to the project
    folder), as seal.verify_lock compares them; the record names them
    relative to its own directory (data/). Each local file is hashed and
    must have the recorded digest.

    An exploratory dataset, which the confirmatory run scores again (since
    25 Oct 2026, before the lock): each of its files as seal.dataset_files
    names it, hashed - EWT test's file under data/, and the treebank files
    of WSJ 01 by their path under the nltk_data entry NLTK finds them in.

    Raises:
        Refused: Listing every record that is missing or unreadable or lists
            other files than the plan, every file that is missing or
            differs from its record, every exploratory file that is missing,
            and a treebank sample NLTK does not find.
    """
    read_digests = _fetch_data().read_digests
    problems: list[str] = []
    data: dict[str, dict[str, str]] = {}
    for key, dataset in plan.datasets.items():
        if dataset.status != "confirmatory":
            digests = _exploratory_digests(key, dataset, root, problems)
            if digests is not None:
                data[key] = digests
            continue
        record = root / dataset.digests
        if not record.is_file():
            problems.append(f"{dataset.digests} does not exist; run scripts/fetch_data.py")
            continue
        try:
            recorded = read_digests(record)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        parent = PurePosixPath(dataset.digests).parent
        names = {PurePosixPath(file).relative_to(parent).as_posix(): file for file in dataset.files}
        if sorted(recorded) != sorted(names):
            problems.append(
                f"{dataset.digests} lists {sorted(recorded)}; the plan's files are {sorted(names)}"
            )
            continue
        digests: dict[str, str] = {}
        for name, file in names.items():
            path = root / file
            if not path.is_file():
                problems.append(f"{file} is missing; run scripts/fetch_data.py")
                continue
            actual = sha256_of(path)
            if actual != recorded[name]:
                problems.append(
                    f"{file} has sha256 {actual}, {dataset.digests} records {recorded[name]}"
                )
                continue
            digests[file] = actual
        data[key] = digests
    if problems:
        raise Refused(problems)
    return data


def _exploratory_digests(
    key: str, dataset: Dataset, root: Path, problems: list[str],
) -> dict[str, str] | None:
    """{file: sha256} of an exploratory dataset (seal.dataset_files); None, `problems` extended."""
    try:
        files = dataset_files(dataset, root)
    except LookupError:
        problems.append(
            f"NLTK does not find the treebank sample (corpora/treebank) that {key!r} is read "
            f"from; setup.sh downloads it"
        )
        return None
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        problems += [f"{name} of {key!r} is missing; setup.sh downloads it" for name in missing]
        return None
    if not files:
        problems.append(f"{key!r} has no file to record")
        return None
    return {name: sha256_of(path) for name, path in files.items()}


def versions() -> dict[str, Any]:
    """The record of `versions`: the environment as a stage's provenance has it, and Stanza's."""
    import stanza

    from syntactic_analysis.stage import environment
    from syntactic_analysis.wrappers.stanza import MODELS_BUILT_FOR

    return {
        **environment(),
        "stanza_resources": stanza.__resources_version__,
        "stanza_models": MODELS_BUILT_FOR,
    }


def compute_lock(root: Path, named: str) -> dict[str, Any]:
    """The whole lock for the project at `root`, at the commit `named`.

    Raises:
        Refused: If the lock exists, the commit or the tree is not the one
            to lock, or a data or model file is not as it must be.
    """
    if default_lock_path(root).exists():
        raise Refused([
            f"{default_lock_path(root)} exists already; a plan is locked once (a deviation "
            f"after the lock goes to plan/deviations.md)"
        ])
    commit = check_commit(root, named)
    plan = load(root / "plan" / "v1.yaml", strict=True)
    data = data_digests(plan, root)
    try:
        models = model_digests(plan, root=root)
        downloaded = downloaded_digests(plan)
    except (OSError, LookupError, RuntimeError, ValueError) as exc:
        raise Refused([str(exc)]) from None
    lock = {
        "plan_sha256": plan.sha256,
        "commit": commit,
        "data": data,
        "models": models,
        "downloaded_models": downloaded,
        "versions": versions(),
        "locked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if tuple(lock) != LOCK_KEYS:
        raise RuntimeError(f"the lock has the keys {list(lock)}, seal.LOCK_KEYS {list(LOCK_KEYS)}")
    return lock


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute plan/LOCK.json (step 8) and print it; write it only with --write.",
    )
    parser.add_argument(
        "--commit", required=True,
        help="the commit to lock, which must be HEAD (a full or short hash, or a ref)",
    )
    parser.add_argument(
        "--write", action="store_true",
        help="write plan/LOCK.json as well (refused if it exists); without it nothing is written",
    )
    parser.add_argument(
        "--root", type=Path, default=None,
        help="the project folder (default: the one this script belongs to)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = project_root() if args.root is None else args.root.resolve()
    try:
        lock = compute_lock(root, args.commit)
    except Refused as exc:
        print("refused; nothing was written:", file=sys.stderr)
        for number, reason in enumerate(exc.reasons, 1):
            print(f"  {number}. {reason}", file=sys.stderr)
        return 1
    text = json.dumps(lock, indent=2, ensure_ascii=False) + "\n"
    print(text, end="")
    if args.write:
        path = default_lock_path(root)
        with open(path, "x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        print(
            f"wrote {path.relative_to(root).as_posix()}; commit it before the confirmatory run "
            f"(seal.verify_lock accepts only a committed lock)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
