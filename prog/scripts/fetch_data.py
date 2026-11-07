"""Download the datasets the plan pins to a source, record their digests, and count them.

    python scripts/fetch_data.py                      # every dataset with a source
    python scripts/fetch_data.py --dataset gentle     # one (repeatable)
    python scripts/fetch_data.py --verify-only        # no network: check the local copy

What is fetched is not written here: each dataset of plan/v1.yaml with a
`source` names a GitHub repository, a ref (a tag or a commit) and the files to
take from it, and Dataset.files says where the local copy goes. Today these are
the two confirmatory test sets, GENTLE (UD 2.18) and MCTB_en.

Both are sealed until the plan is locked (syntactic_analysis.seal), so this
script may download them, hash them and count them, and it prints nothing but
counts, digests and file names - not a sentence, a word, a tag or a tree, and
not in an error message either: the counters return numbers and line numbers
only. The counts are the format check. For CoNLL-U: sentences, tokens
(multiword-token ranges, plus the word lines no range covers), syntactic words
(word lines with an integer id), documents (`# newdoc id`, which must equal
the plan's expected_units), malformed lines (must be none), and the same
sentences, syntactic words and documents again as data_loader.parse_conllu
reads them, so the loader a stage uses is known to read the file whole. UD
counts a multiword token once as a token and once per word it holds, and the
tools tag and parse words; until 23 Oct 2026 this script printed the words as
"tokens", the number UD reports as syntactic words. For a
one-tree-per-line treebank: trees and tokens per domain file, the lines that do
not parse as one tree under a TOP-like root (must be none), and the NML nodes
of all the files together - a count per domain would be a per-domain statistic
of a sealed set beyond its size.

parse_conllu is not a counter: on a line it cannot read it raises, and
Python's message quotes the field (`invalid literal for int(): '...'`). So it
reads a file only after count_conllu has found every line well formed, and an
exception it raises all the same is reported by its type alone. Since 23 Oct
2026; until then it read a file with malformed lines too.

The digests are recorded in <path>.sha256 (Dataset.digests), in sha256sum
format with paths relative to data/, so `cd data && sha256sum -c <file>` checks
them too. A record, once written, is the seal's reference:

  * a local file that matches it is kept, and not downloaded again;
  * a local file that does not match it is refused and left as it is - never
    overwritten - and the script exits non-zero;
  * a missing file is downloaded and refused unless it matches.

Without a record (a first download), the files are downloaded, counted, and the
record is written only if every count passes. --verify-only downloads nothing:
it needs the record and the files, and checks digests and counts.

Exit status: 0 when every dataset checked passes, 1 when any problem is found,
2 for a usage error.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

from syntactic_analysis.checksum import ChecksumMismatch, download_verified, sha256_of
from syntactic_analysis.plan import Dataset, Plan, load
from syntactic_analysis.project import project_root

# One line of a sha256sum file: the digest, a space, a space or "*" (binary
# mode), the path.
_DIGEST_LINE = re.compile(r"([0-9a-f]{64}) [ *](.+)")

# The ids a CoNLL-U word line may have: an integer, a multiword range "1-2",
# an empty node "1.1". The first is a syntactic word; a token is a range, or
# a word no range of its sentence covers; an empty node is neither.
_WORD_ID = re.compile(r"[1-9][0-9]*")
_RANGE_ID = re.compile(r"([1-9][0-9]*)-([1-9][0-9]*)")
_EMPTY_ID = re.compile(r"[0-9]+\.[1-9][0-9]*")
# The HEAD of a word line: an integer (0 for the root), or "_", which
# parse_conllu reads as no head. Anything else it cannot read.
_HEAD = re.compile(r"[0-9]+|_")

# How many line numbers of a malformed or unparsable line are printed.
_SHOWN_LINES = 10

Download = Callable[..., None]


# --- counting ----------------------------------------------------------------


@dataclass(frozen=True)
class ConlluCounts:
    """What count_conllu found: counts and line numbers, nothing of the text.

    Attributes:
        sentences:  Blocks of lines with at least one word line.
        tokens:     Surface tokens: multiword-token ranges, plus the word
                    lines no range of their sentence covers (UD's "tokens").
        words:      Syntactic words: word lines with an integer id, no ranges,
                    no empty nodes (UD's "syntactic words"). What
                    parse_conllu reads and the tools tag and parse.
        documents:  `# newdoc id` lines.
        malformed:  1-based numbers of the lines that are not a comment, not
                    blank and not ten tab-separated fields under a valid id,
                    and of the word lines whose HEAD is not an integer or _.
    """

    sentences: int
    tokens: int
    words: int
    documents: int
    malformed: tuple[int, ...]


def count_conllu(path: Path) -> ConlluCounts:
    """Count a CoNLL-U file without returning any of its text."""
    sentences = tokens = words = documents = 0
    malformed: list[int] = []
    in_sentence = False
    covered: set[int] = set()   # word ids inside a range of the current sentence
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            in_sentence = False
            covered = set()
            continue
        if line.startswith("#"):
            if line.startswith("# newdoc id ="):
                documents += 1
            continue
        fields = line.split("\t")
        if len(fields) != 10:
            malformed.append(number)
            continue
        if _WORD_ID.fullmatch(fields[0]):
            if not _HEAD.fullmatch(fields[6]):
                malformed.append(number)
                continue
            words += 1
            if int(fields[0]) not in covered:
                tokens += 1
            if not in_sentence:
                sentences += 1
                in_sentence = True
        elif match := _RANGE_ID.fullmatch(fields[0]):
            tokens += 1
            covered.update(range(int(match.group(1)), int(match.group(2)) + 1))
        elif not _EMPTY_ID.fullmatch(fields[0]):
            malformed.append(number)
    return ConlluCounts(sentences, tokens, words, documents, tuple(malformed))


def _lines(numbers: tuple[int, ...]) -> str:
    """Line numbers for a message: the first few and how many in all."""
    shown = ", ".join(str(number) for number in numbers[:_SHOWN_LINES])
    more = f" and {len(numbers) - _SHOWN_LINES} more" if len(numbers) > _SHOWN_LINES else ""
    return f"line(s) {shown}{more}"


def _check_conllu(dataset: Dataset, paths: list[Path]) -> list[str]:
    """Count each CoNLL-U file, print the counts, and return what fails.

    parse_conllu reads a file only when count_conllu found no malformed line,
    and an exception it raises is reported by its type, never its message:
    see the module docstring.
    """
    from syntactic_analysis.data_loader import load_conllu

    problems = []
    for path in paths:
        counts = count_conllu(path)
        expected = dataset.resampling.expected_units
        print(
            f"  {path.name}: {counts.documents} documents, {counts.sentences} sentences, "
            f"{counts.tokens} tokens, {counts.words} syntactic words, "
            f"{len(counts.malformed)} malformed lines"
        )
        if counts.malformed:
            problems.append(f"{path.name}: malformed {_lines(counts.malformed)}")
        if not counts.sentences:
            problems.append(f"{path.name} holds no sentence")
        if dataset.resampling.unit == "document" and expected is not None:
            if counts.documents != expected:
                problems.append(
                    f"{path.name} has {counts.documents} documents; the plan expects {expected}"
                )
        if counts.malformed:
            print(f"  {path.name} not read by parse_conllu: it has malformed lines")
            continue
        try:
            parse = load_conllu(path)
        except Exception as exc:  # noqa: BLE001 - reported, without its message
            problems.append(
                f"parse_conllu cannot read {path.name} ({type(exc).__name__}; the message "
                f"is not shown, as it may quote the file)"
            )
            continue
        read_sentences = len(parse.sentences)
        # Sentence.tokens holds one entry per syntactic word.
        read_words = sum(len(sentence.tokens) for sentence in parse.sentences)
        read_documents = len({sentence.doc_id for sentence in parse.sentences})
        print(
            f"  {path.name} as parse_conllu reads it: {read_documents} documents, "
            f"{read_sentences} sentences, {read_words} syntactic words"
        )
        if (read_sentences, read_words) != (counts.sentences, counts.words):
            problems.append(
                f"parse_conllu reads {read_sentences} sentences and {read_words} syntactic "
                f"words from {path.name}; counted {counts.sentences} and {counts.words}"
            )
        if counts.documents and read_documents != counts.documents:
            problems.append(
                f"parse_conllu assigns {read_documents} documents in {path.name}; "
                f"counted {counts.documents}"
            )
    return problems


def _check_trees(dataset: Dataset, paths: list[Path]) -> list[str]:
    """Count each one-tree-per-line file, print the counts, and return what fails."""
    from syntactic_analysis.evaluation.mctb import count_trees, domain_of

    problems = []
    trees = tokens = nml = 0
    for path in paths:
        counts = count_trees(path)
        trees, tokens, nml = trees + counts.trees, tokens + counts.tokens, nml + counts.nml
        # NML only in the total: see the module docstring.
        print(
            f"  {domain_of(path.name)}: {counts.trees} trees, {counts.tokens} tokens, "
            f"{len(counts.unparsable)} unparsable lines, {len(counts.blank)} blank lines"
        )
        if counts.unparsable:
            problems.append(
                f"{path.name}: {_lines(counts.unparsable)} do not parse as one tree under a "
                f"TOP-like root"
            )
        if counts.blank:
            problems.append(f"{path.name}: blank {_lines(counts.blank)}")
        if not counts.trees:
            problems.append(f"{path.name} holds no tree")
    print(f"  {len(paths)} domains, {trees} trees, {tokens} tokens, {nml} NML nodes")
    return problems


def _counter(dataset: Dataset) -> Callable[[Dataset, list[Path]], list[str]] | None:
    """The format check of a dataset: CoNLL-U for pos/dep, tree lines for con."""
    if dataset.files and all(file.endswith(".conllu") for file in dataset.files):
        return _check_conllu
    if dataset.tasks == ("con",):
        return _check_trees
    return None


# --- digests -----------------------------------------------------------------


def read_digests(record: Path) -> dict[str, str]:
    """A sha256sum file as {relative path: digest}.

    Raises:
        ValueError: On a line that is not `<sha256>  <path>`, or a path listed twice.
    """
    digests: dict[str, str] = {}
    for number, line in enumerate(record.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = _DIGEST_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"{record.name} line {number} is not '<sha256>  <path>'")
        digest, name = match.groups()
        if name in digests:
            raise ValueError(f"{record.name} lists {name} twice")
        digests[name] = digest
    return digests


def write_digests(record: Path, digests: dict[str, str]) -> None:
    """Write a sha256sum file, LF line endings on every platform.

    .gitattributes keeps *.sha256 at LF in the checkout too: sha256sum reads
    the path to the end of the line, and with CRLF it would look for a file
    whose name ends in a carriage return.
    """
    with open(record, "w", encoding="utf-8", newline="\n") as handle:
        for name, digest in digests.items():
            handle.write(f"{digest}  {name}\n")


def _raw_url(repository: str, ref: str, file: str) -> str:
    """The raw.githubusercontent.com URL of one file at one ref."""
    owner_repo = repository.removeprefix("https://github.com/")
    return f"https://raw.githubusercontent.com/{owner_repo}/{ref}/{file}"


# --- one dataset ---------------------------------------------------------------


def fetch_dataset(
    dataset: Dataset,
    root: Path,
    *,
    verify_only: bool = False,
    download: Download = download_verified,
) -> list[str]:
    """Download (unless verify_only), verify and count one dataset; return its problems.

    Prints one line per file (size, digest, what was done) and the counts.
    `download` is checksum.download_verified; a test passes its own.
    """
    source = dataset.source
    record = root / dataset.digests
    local = [root / file for file in dataset.files]
    # The record names each file relative to its own directory (data/).
    names = [PurePosixPath(file).relative_to(PurePosixPath(dataset.digests).parent).as_posix()
             for file in dataset.files]
    print(f"{dataset.id}: {source.repository} at {source.ref}")

    recorded: dict[str, str] | None = None
    if record.exists():
        try:
            recorded = read_digests(record)
        except ValueError as exc:
            return [str(exc)]
        if sorted(recorded) != sorted(names):
            return [
                f"{dataset.digests} lists {sorted(recorded)}; the plan's files are {sorted(names)}"
            ]
    elif verify_only:
        return [f"{dataset.digests} does not exist; run scripts/fetch_data.py without --verify-only"]

    problems: list[str] = []
    digests: dict[str, str] = {}
    for name, relative, path, file in zip(names, dataset.files, local, source.files):
        expected = recorded.get(name) if recorded else None
        if path.exists() and expected is not None:
            actual, action = sha256_of(path), "present, matches the record"
            if actual != expected:
                problems.append(
                    f"{relative} has sha256 {actual}, {dataset.digests} records {expected}; "
                    f"left as it is"
                )
                action = "MISMATCH"
        elif verify_only:
            problems.append(f"{relative} is missing")
            print(f"  {name}: missing")
            continue
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                download(_raw_url(source.repository, source.ref, file), path, sha256=expected)
            except ChecksumMismatch as exc:
                problems.append(
                    f"the download of {file} has sha256 {exc.actual}, {dataset.digests} "
                    f"records {exc.expected}; nothing was written"
                )
                print(f"  {name}: download refused, sha256 {exc.actual}")
                continue
            except OSError as exc:  # urllib.error.URLError is an OSError
                problems.append(f"the download of {file} failed: {exc}")
                print(f"  {name}: download failed")
                continue
            actual, action = sha256_of(path), "downloaded"
            if expected is not None:
                action += ", matches the record"
        digests[name] = actual
        print(f"  {name}: {path.stat().st_size} bytes, sha256 {actual}, {action}")

    if problems:
        return problems
    counter = _counter(dataset)
    if counter is None:
        return [f"{dataset.id}: no format check for a dataset of tasks {list(dataset.tasks)}"]
    problems = counter(dataset, local)
    if problems:
        return problems
    if recorded is None:
        write_digests(record, digests)
        print(f"  digests recorded in {dataset.digests}")
    else:
        print(f"  digests match {dataset.digests}")
    return []


# --- the script ----------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the datasets plan/v1.yaml pins to a source, record their "
                    "SHA-256 and check them by counting. Prints counts and digests only.",
    )
    parser.add_argument(
        "--dataset", action="append", metavar="ID",
        help="a dataset id of the plan with a source; repeatable (default: every one)",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="download nothing; check the local files against the recorded digests and counts",
    )
    parser.add_argument(
        "--plan", type=Path, default=None,
        help="the plan file (default: plan/v1.yaml)",
    )
    return parser.parse_args(argv)


def run(
    plan: Plan,
    ids: list[str] | None = None,
    *,
    root: Path | None = None,
    verify_only: bool = False,
    download: Download = download_verified,
) -> int:
    """Fetch or verify the datasets `ids` (every one with a source when None); the exit status."""
    root = project_root() if root is None else Path(root)
    sourced = [key for key, dataset in plan.datasets.items() if dataset.source is not None]
    ids = sourced if not ids else list(dict.fromkeys(ids))
    unknown = [key for key in ids if key not in sourced]
    if unknown:
        print(
            f"error: {unknown} is not a dataset with a source; expected one of {sourced}",
            file=sys.stderr,
        )
        return 2
    failed = False
    for key in ids:
        problems = fetch_dataset(
            plan.datasets[key], root, verify_only=verify_only, download=download,
        )
        for problem in problems:
            print(f"  PROBLEM: {problem}")
        failed = failed or bool(problems)
    print("FAILED" if failed else "OK")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run(load(args.plan), args.dataset, verify_only=args.verify_only)


if __name__ == "__main__":
    sys.exit(main())
