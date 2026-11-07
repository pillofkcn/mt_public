"""Where the repository root is, for everything that needs a default path.

`project_root()` lived in evaluation/harness.py, which made every script that
only wanted a default output directory import the measurement path to get it.
It is not an evaluation concern and it is not specific to one task, so it sits
at the package root with the default paths that are built on it: the model
cache, the EWT splits and the two subdirectories of a run directory. Until
12 Oct 2026 five wrappers and one script each counted parents[] for
themselves.
"""
from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Repository root, derived from this file rather than the working directory.

    This file is src/syntactic_analysis/project.py, so the root is three
    components up: project.py -> syntactic_analysis -> src -> root. It was
    parents[3] while the function lived one directory deeper, in
    evaluation/harness.py, and moving it without re-counting pointed every
    default path one level above the repository - which is how the gold corpus
    came to be looked for in Desktop/School/data/.
    """
    return Path(__file__).resolve().parents[2]


def model_cache_dir() -> Path:
    """Where downloaded and locally trained models are kept: <root>/model_cache.

    Absolute, so the cache is the same directory whatever the working directory
    of the process is. Nothing is created here: a caller that writes into the
    cache makes the directory itself.
    """
    return project_root() / "model_cache"


# The three splits of the UD English Web Treebank that setup.sh puts in data/.
EWT_SPLITS: tuple[str, ...] = ("train", "dev", "test")


def ewt_split_path(split: str) -> Path:
    """The CoNLL-U file of one EWT split: <root>/data/en_ewt-ud-<split>.conllu.

    The path is built, not checked: whether the file is there is for the loader
    that opens it to say.

    Raises:
        ValueError: If `split` is not one of EWT_SPLITS.
    """
    if split not in EWT_SPLITS:
        raise ValueError(
            f"unknown EWT split {split!r}; expected one of {sorted(EWT_SPLITS)}"
        )
    return project_root() / "data" / f"en_ewt-ud-{split}.conllu"


# The two subdirectories of a run directory. The report lands in log/, beside
# the numbers it describes: until 8 Oct 2026 it defaulted into log/scratch/,
# which named the normal case scratch and filed the report away from the CSVs
# it reports on. A third, trees/, held the tree dumps of the constituency stage
# until version 1.0, which writes no trees.
CSV_SUBDIR, LOG_SUBDIR = "csv", "log"
