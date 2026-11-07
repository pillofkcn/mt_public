"""Retrain RDRPOSTagger on the EWT training split with its own learner: the plan's `rdr_ewt`.

Why this stage exists
---------------------
The RDRPOSTagger model the wrapper used to load (`en_ewt_upos`, published by
its authors; removed from the wrapper on 24 Oct 2026) was trained by them on
UD_English-EWT r2.4 train. Every other in-domain
configuration of the plan is trained on the repository's EWT train file
(data/en_ewt-ud-train.conllu, a later snapshot), so this stage trains the
tagger on that file, with the learner its authors published, and nothing of
this project's between the two.

The learner is GPL-3.0-or-later and is not part of this repository: this
script downloads it, file by file, from the official repository
(github.com/datquocnguyen/RDRPOSTagger) at a pinned commit, checks every file
against the SHA-256 recorded below, keeps it OUTSIDE the repository
(--learner-dir, by default ~/.cache/rdrpostagger/<commit>), and runs it as a
separate program with this interpreter. Nothing of it is imported here or by
the wrapper, which reads the two files it writes (.RDR, the rules, and .DICT,
the lexicon) with its own reader (wrappers/rdrpos.py).

What it does
------------
  1. Fetches the learner if it is not there yet, and checks it.
  2. Converts the EWT train CoNLL-U into the learner's format: one sentence
     per line, `form/UPOS` separated by spaces. Multiword-token range lines
     and empty nodes are skipped, so the tagger is trained on the syntactic
     words the gold is scored on. A form with whitespace in it cannot be
     written in that format and is refused, not altered (EWT train has none).
  3. Runs `python RDRPOSTagger.py train <file>` in the learner's pSCRDRtagger
     folder, as its README says, with the thresholds of its Utility/Config.py
     unchanged (THRESHOLD = (3, 2)), and with a fixed hash seed (HASH_SEED).
     The learner has no seed of its own, but it is not deterministic under
     Python 3 without one: it collects the candidate rules of a word in a
     set of strings, counts them in the order it iterates that set, and of
     two rules with the same count keeps the first. Python 3 randomizes the
     hash of a string in every new process (the Python 2 the learner was
     written for did not), so two runs on the same data broke ties
     differently and wrote two different rule trees (23 Oct 2026: 76 384 and
     76 170 bytes, the same lexicon). PYTHONHASHSEED fixes that order; the
     process gets no other PYTHON* variable of the caller's environment.
     Python runs without site (-S), without the script's folder on the path
     (-P; the learner adds its own root itself), in UTF-8 mode (-X utf8) -
     the learner opens its files without naming an encoding, and EWT is
     UTF-8 - and unbuffered (-u), so the log shows which tag it is learning
     rules for.
  4. Checks the result: the learner prints an error and still exits with 0,
     so the run counts only if it printed "Done!" and wrote both files.
  5. Writes training_record.json beside the model: the commit, the digests of
     the learner's files, of the training file and of the two outputs, the
     command, the versions, the counts and the training time. Then, on the
     dev split (never the test split), it records two things: the UPOS
     accuracy through the wrapper, and whether the wrapper's reader tags
     every dev token as the learner's own tagger does (`python
     RDRPOSTagger.py tag`), the check the wrapper's docstring reports, on
     the test split, for the published model it loaded until 24 Oct 2026.

The output goes to model_cache/rdrpos_ewt/, which git ignores. The wrapper
loads it as RdrPosAnalyzer().load_model("en_ewt_train") and refuses it
without the record, or when a file is not the one the record names.

Usage
-----
    python scripts/train_rdrpos.py            # fetch, convert, train, record
    python scripts/train_rdrpos.py --force    # train again over model_cache/rdrpos_ewt
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

from syntactic_analysis.checksum import download_verified, sha256_of
from syntactic_analysis.evaluation.provenance import recorded_path, stage_started
from syntactic_analysis.evaluation.report import Reporter
from syntactic_analysis.project import ewt_split_path, project_root
from syntactic_analysis.wrappers.rdrpos import (
    EWT_LEXICON,
    EWT_RULES,
    EWT_TRAIN_FILE,
    EWT_TRAINING_RECORD,
    EWT_WORK_DIR,
    TRAINED_MODEL,
)

# The commit of the official repository the learner is taken from: the head of
# its master branch since 22 Jun 2019, which is also the commit the wrapper's
# published model files were pinned to until the wrapper stopped loading them
# (24 Oct 2026).
LEARNER_COMMIT = "eb8b11d2d75b8c39665dc5500b10e06e5a73a1e9"
LEARNER_REPOSITORY = "https://github.com/datquocnguyen/RDRPOSTagger"
_RAW_BASE = f"https://raw.githubusercontent.com/datquocnguyen/RDRPOSTagger/{LEARNER_COMMIT}/"

# The learner's files at that commit, with their SHA-256: the program and its
# licence, nothing else of the repository (no pre-trained model).
LEARNER_FILES: dict[str, str] = {
    "License.txt": "b5323e2b378e8fd827c91d8a892eb6595433d993b03de346e130162f24f6c37b",
    "InitialTagger/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "InitialTagger/InitialTagger.py": "997fd9c029e48ec14eee44a4254bd467f84d720b0b926289f31a69adddbeb0ee",
    "SCRDRlearner/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "SCRDRlearner/Node.py": "5f07dd5e16800358a73e208b8c2aa782a9c858799c3abdb61096207a867791b2",
    "SCRDRlearner/Object.py": "faf70eb26d4dbe2ac73696bd83c1719d8f441851949423968d443877806d8b3d",
    "SCRDRlearner/SCRDRTree.py": "20513265993947a22e9331bd09302c7c0d2c23ae34be0de681a27a290357660c",
    "SCRDRlearner/SCRDRTreeLearner.py": "483526a25b1725e0ecdeeca80f1218c13e7d56a8f8986fb5f1e5ef1aa72877c6",
    "Utility/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "Utility/Config.py": "d01b22bab43e5c242d0c007b1f1263a4d77d5ef41ee4e2eb7ee011a97118f977",
    "Utility/LexiconCreator.py": "7b1df40136e1adbb974dc8bc3e59c55784c738c7d8f6d2a9adf33abf830bdd74",
    "Utility/Utils.py": "05b85b44b53e44eff162ea3d6a3850c2a63deff1626928c7ab9450f8cda9ec17",
    "pSCRDRtagger/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "pSCRDRtagger/RDRPOSTagger.py": "43b2b710d97bdf461589694352c276d8c3ed70cce7a34ffa9bdc754a949c5439",
}

DEFAULT_LEARNER_DIR = Path.home() / ".cache" / "rdrpostagger" / LEARNER_COMMIT[:12]

# The seed of Python's string hashing in the learner's process (see step 3):
# the project's seed, as spaCy's training uses it.
HASH_SEED = 20260101

# How the learner is run (see step 3): no site, no script folder on the path,
# UTF-8 mode, unbuffered, so that its progress lines reach the log as it
# prints them; and its environment, below.
_PYTHON = [sys.executable, "-S", "-P", "-X", "utf8", "-u"]
_PYTHON_SHOWN = f"PYTHONHASHSEED={HASH_SEED} python -S -P -X utf8 -u"


def learner_environment() -> dict[str, str]:
    """The caller's environment without its PYTHON* variables, with PYTHONHASHSEED set."""
    environment = {
        name: value for name, value in os.environ.items()
        if not name.upper().startswith("PYTHON")
    }
    environment["PYTHONHASHSEED"] = str(HASH_SEED)
    return environment


def fetch_learner(directory: Path) -> None:
    """Download the learner's files that are missing or not the recorded ones.

    Raises:
        ValueError:   If `directory` is inside the project (the learner is GPL
                      and stays out of this repository).
        RuntimeError: If a file cannot be downloaded or is not the recorded one.
    """
    root = project_root().resolve()
    resolved = directory.resolve()
    if resolved == root or root in resolved.parents:
        raise ValueError(
            f"--learner-dir {directory} is inside the project ({root}); the learner "
            f"is GPL-3.0-or-later and is kept outside this repository"
        )
    for relative, digest in LEARNER_FILES.items():
        path = directory / relative
        if path.is_file() and sha256_of(path) == digest:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {relative} ({LEARNER_COMMIT[:12]})", flush=True)
        try:
            download_verified(_RAW_BASE + relative, path, sha256=digest)
        except Exception as exc:  # noqa: BLE001 - reported with the file it was
            raise RuntimeError(f"could not fetch {_RAW_BASE + relative}: {exc}") from exc


def learner_threshold(directory: Path) -> str:
    """The THRESHOLD line of the learner's Utility/Config.py, as written there."""
    text = (directory / "Utility" / "Config.py").read_text(encoding="utf-8")
    match = re.search(r"^THRESHOLD\s*=\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def conllu_sentences(path: Path) -> list[list[tuple[str, str]]]:
    """The (form, UPOS) pairs of each sentence: syntactic words only."""
    sentences: list[list[tuple[str, str]]] = []
    words: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if not line:
                if words:
                    sentences.append(words)
                    words = []
                continue
            if line.startswith("#"):
                continue
            columns = line.split("\t")
            if "-" in columns[0] or "." in columns[0]:
                continue
            words.append((columns[1], columns[3]))
    if words:
        sentences.append(words)
    return sentences


def write_training_file(sentences: list[list[tuple[str, str]]], path: Path) -> None:
    """The learner's format: one sentence per line, `form/UPOS` separated by spaces.

    Raises:
        ValueError: If a form contains whitespace or a tag is empty.
    """
    lines = []
    for number, words in enumerate(sentences, 1):
        for form, tag in words:
            if not form or any(char.isspace() for char in form) or not tag or tag == "_":
                raise ValueError(
                    f"sentence {number}: the word {form!r}/{tag!r} cannot be written in "
                    f"the learner's format (whitespace in the form or no UPOS)"
                )
        lines.append(" ".join(f"{form}/{tag}" for form, tag in words))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_learner(arguments: list[str], learner_dir: Path, log) -> str:
    """Run RDRPOSTagger.py in its pSCRDRtagger folder; return what it printed.

    Raises:
        RuntimeError: If it exits with a non-zero status or prints its error line.
    """
    command = [*_PYTHON, "RDRPOSTagger.py", *arguments]
    shown = f"{_PYTHON_SHOWN} RDRPOSTagger.py " + " ".join(
        recorded_path(part) if Path(part).is_absolute() else part for part in arguments
    )
    print(f"$ {shown}", flush=True)
    log.write(f"$ {shown}\n")
    lines = []
    with subprocess.Popen(
        command, cwd=learner_dir / "pSCRDRtagger", env=learner_environment(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            lines.append(line)
    output = "".join(lines)
    if process.returncode != 0 or "ERROR ==>" in output:
        raise RuntimeError(f"{shown} failed (status {process.returncode}): {output.strip()[-500:]}")
    return output


def dev_checks(dev_path: Path, learner_dir: Path, log) -> dict:
    """UPOS accuracy on dev through the wrapper, and agreement with the learner's tagger.

    The learner's tagger reads raw text, one sentence per line, and splits it
    on whitespace; the dev split has no form with whitespace in it (refused
    otherwise), so its tokens line up with the gold words one to one.
    """
    from syntactic_analysis.evaluation.gold import load_gold_pretok
    from syntactic_analysis.evaluation.metrics import corpus_metric, pos_sentence_counts
    from syntactic_analysis.wrappers.rdrpos import RdrPosAnalyzer

    gold = load_gold_pretok(dev_path)
    analyzer = RdrPosAnalyzer(key="rdr_ewt")
    analyzer.load_model(TRAINED_MODEL)
    parse = analyzer.analyze_dependency(gold.pretok)
    pos = [pos_sentence_counts(p, g) for p, g in zip(parse.sentences, gold.parse.sentences)]

    for words in gold.pretok:
        if any(char.isspace() for word in words for char in word):
            raise ValueError("a dev form contains whitespace; the learner's tagger would split it")
    raw = EWT_WORK_DIR / "dev.RAW"
    raw.write_text("\n".join(" ".join(words) for words in gold.pretok) + "\n",
                   encoding="utf-8", newline="\n")
    run_learner(["tag", str(EWT_RULES), str(EWT_LEXICON), str(raw)], learner_dir, log)
    tagged_path = raw.with_name(raw.name + ".TAGGED")
    learner_tags = [
        [pair.rsplit("/", 1)[1] for pair in line.split()]
        for line in tagged_path.read_text(encoding="utf-8").splitlines()
    ]
    wrapper_tags = [[token.pos for token in sentence.tokens] for sentence in parse.sentences]
    if len(learner_tags) != len(wrapper_tags):
        raise RuntimeError(
            f"the learner tagged {len(learner_tags)} dev sentences, the wrapper "
            f"{len(wrapper_tags)}"
        )
    tokens = sum(len(tags) for tags in wrapper_tags)
    differing = 0
    for ours, theirs in zip(wrapper_tags, learner_tags):
        if len(ours) != len(theirs):
            differing += len(ours)  # misaligned: every token of it counts as differing
        else:
            differing += sum(a != b for a, b in zip(ours, theirs))
    raw.unlink()
    tagged_path.unlink()
    return {
        "sentences": gold.n_sentences,
        "tokens": gold.n_tokens,
        "upos": round(corpus_metric(pos), 2),
        "learner_tagger_agreement": {"tokens": tokens, "differing": differing},
        "model_version": analyzer.model_version,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RDRPOSTagger on the EWT train split with its official learner."
    )
    parser.add_argument(
        "--learner-dir", type=Path, default=DEFAULT_LEARNER_DIR,
        help=f"Where the learner is kept, outside the repository (default {DEFAULT_LEARNER_DIR}).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Train again over model_cache/rdrpos_ewt.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = stage_started()
    train_path = ewt_split_path("train")
    dev_path = ewt_split_path("dev")
    learner_dir = args.learner_dir

    Reporter().header(
        "STAGE train_rdrpos: RDRPOSTAGGER RETRAINED ON THE EWT TRAINING SPLIT",
        {
            "started": started,
            "train": recorded_path(train_path),
            "dev": recorded_path(dev_path),
            "learner": f"{LEARNER_REPOSITORY} @ {LEARNER_COMMIT[:12]}",
            "learner dir": str(learner_dir),
            "output": recorded_path(EWT_WORK_DIR),
        },
    )
    for path in (train_path, dev_path):
        if not path.exists():
            print(f"not found: {path}. Run setup.sh to download the treebank.", file=sys.stderr)
            return 1
    if EWT_TRAINING_RECORD.exists() and not args.force:
        print(f"Model already present: {recorded_path(EWT_WORK_DIR)}. Pass --force to train again.")
        return 0

    fetch_learner(learner_dir)
    EWT_WORK_DIR.mkdir(parents=True, exist_ok=True)
    for path in EWT_WORK_DIR.iterdir():
        if path.is_file():
            path.unlink()  # an earlier run's files: the record must describe this one

    training_file = EWT_WORK_DIR / EWT_TRAIN_FILE
    sentences = conllu_sentences(train_path)
    write_training_file(sentences, training_file)

    log_path = EWT_WORK_DIR / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        training_started = time.perf_counter()
        output = run_learner(["train", str(training_file)], learner_dir, log)
        elapsed = time.perf_counter() - training_started
        if "Done!" not in output or not (EWT_RULES.is_file() and EWT_LEXICON.is_file()):
            raise RuntimeError(f"the learner did not finish; see {recorded_path(log_path)}")

        rules = EWT_RULES.read_text(encoding="utf-8").splitlines()
        record = {
            "started": started,
            "learner_repository": LEARNER_REPOSITORY,
            "learner_commit": LEARNER_COMMIT,
            "learner_files_sha256": dict(LEARNER_FILES),
            "learner_threshold": learner_threshold(learner_dir),
            "hash_seed": HASH_SEED,
            "python": platform.python_version(),
            "os": platform.platform(),
            "cpu": platform.processor(),
            "command": f"{_PYTHON_SHOWN} RDRPOSTagger.py train {recorded_path(training_file)}",
            "train_path": recorded_path(train_path),
            "train_sha256": sha256_of(train_path),
            "train_sentences": len(sentences),
            "train_tokens": sum(len(words) for words in sentences),
            "training_file_sha256": sha256_of(training_file),
            "training_minutes": round(elapsed / 60, 2),
            # One rule per line; the first line is the always-true root.
            "rules": sum(1 for line in rules[1:] if line.strip()),
            "files": {
                "rules": {"path": recorded_path(EWT_RULES), "sha256": sha256_of(EWT_RULES)},
                "lexicon": {"path": recorded_path(EWT_LEXICON), "sha256": sha256_of(EWT_LEXICON)},
            },
            "log": recorded_path(log_path),
        }
        EWT_TRAINING_RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\nTrained in {elapsed / 60:.1f} min; record: {recorded_path(EWT_TRAINING_RECORD)}")

        record["dev"] = dev_checks(dev_path, learner_dir, log)
    EWT_TRAINING_RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record["dev"], indent=2))
    if record["dev"]["learner_tagger_agreement"]["differing"]:
        print("The wrapper's reader and the learner's tagger disagree; see the record.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
