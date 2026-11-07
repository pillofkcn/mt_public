"""MCTB_en, the confirmatory constituency test set: one bracketed tree per line.

MCTB_en (Yang, Cui, Ning, Wu and Zhang 2022, "Challenges to Open-Domain
Constituency Parsing", Findings of ACL 2022) is five domain files - dialogue,
forum, law, literature and review - each holding one tree per line under a
TOP-like root. The annotators followed the PTB guideline and corrected the
trees a parser built on the Berkeley Neural Parser had predicted (section 4.1
of the paper). Its identity, licence and counts are in data/DATASET.md.

It is confirmatory and sealed until the plan is locked, so this module has two
readers for two purposes, and they share one way of reading a line
(`parse_tree_line`):

  * `count_trees` reads a file only to count it - trees, tokens, NML nodes,
    and the numbers of the lines that do not parse - and returns nothing
    else, so nothing it returns can show a sentence. scripts/fetch_data.py
    checks the format of the download with it.
  * `_gold_set` turns the files into the ConstituencyGoldSet a stage scores
    against. It is private: its one caller is syntactic_analysis.seal, so a
    stage gets MCTB_en only through open_dataset(), which refuses it until
    the plan is locked and the run is started with --confirmatory.

A tree is prepared as the WSJ gold is (evaluation/ptb.py): traces removed,
function tags stripped, and the unary root wrapper dropped, so that a tree
starts at its first real constituent as an NLTK WSJ tree does. The resampling
unit is the sentence (plan/v1.yaml); the domain travels with each sentence as
metadata, in PtbSentence.domain.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from syntactic_analysis.normalize import ROOT_WRAPPERS, strip_unary_root_wrapper
from syntactic_analysis.ptb_conventions import clean_gold_tree, strip_function_tags

if TYPE_CHECKING:  # pragma: no cover - typing only
    from nltk import Tree

    from syntactic_analysis.evaluation.gold import ConstituencyGoldSet
    from syntactic_analysis.evaluation.ptb import PtbSentence

# The label the revised Penn Treebank gives the nominal modifiers it brackets
# inside a flat NP. The original PTB guideline that MCTB_en follows has none.
NML_LABEL = "NML"


def domain_of(file_name: str) -> str:
    """The domain a file holds: the part of its name before the first dot.

    "law.cleaned.txt" -> "law".
    """
    return file_name.split(".", 1)[0]


def parse_tree_line(line: str) -> "Tree":
    """One line as one tree, or ValueError saying what is wrong but not what the line holds.

    The line must be exactly one bracketed tree whose root is a TOP-like
    wrapper (normalize.ROOT_WRAPPERS) over at least one constituent. NLTK's own
    message quotes the text around the error, so it is replaced: on a sealed
    file, an error message is output too.
    """
    from nltk import Tree

    try:
        tree = Tree.fromstring(line)
    except ValueError:
        raise ValueError("not one bracketed tree") from None
    if tree.label() not in ROOT_WRAPPERS:
        raise ValueError(f"the root is not one of {list(ROOT_WRAPPERS)}")
    if not len(tree) or any(isinstance(child, str) for child in tree):
        raise ValueError("the root holds no constituent")
    return tree


@dataclass(frozen=True)
class TreeCounts:
    """What `count_trees` found in one file: counts and line numbers, nothing else.

    Attributes:
        trees:      Lines that parse as one tree.
        tokens:     Their words, after trace removal: what a parser is given.
        nml:        Nodes labelled NML (function tags stripped).
        unparsable: The 1-based numbers of the lines that do not parse.
        blank:      The 1-based numbers of the empty lines; one tree per line
                    leaves no room for one.
    """

    trees: int
    tokens: int
    nml: int
    unparsable: tuple[int, ...]
    blank: tuple[int, ...]

    @property
    def clean(self) -> bool:
        """True when every line is one tree."""
        return not self.unparsable and not self.blank


def count_trees(path: str | Path) -> TreeCounts:
    """Count a one-tree-per-line file without returning any of its text."""
    trees = tokens = nml = 0
    unparsable: list[int] = []
    blank: list[int] = []
    text = Path(path).read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            blank.append(number)
            continue
        try:
            tree = parse_tree_line(line)
        except ValueError:
            unparsable.append(number)
            continue
        cleaned = clean_gold_tree(tree)
        if cleaned is None:  # nothing but traces: not a sentence a parser can be given
            unparsable.append(number)
            continue
        trees += 1
        tokens += len(cleaned.leaves())
        nml += sum(
            1 for node in tree.subtrees() if strip_function_tags(node.label()) == NML_LABEL
        )
    return TreeCounts(trees, tokens, nml, tuple(unparsable), tuple(blank))


def _read_domain(path: Path) -> list["PtbSentence"]:
    """The gold sentences of one domain file, in file order.

    Private: reached only through `_gold_set`, which only
    syntactic_analysis.seal calls. Each sentence's id is "<domain>-<n>", n
    counting lines from 1.

    Raises:
        ValueError: Naming the file and line number of a line that does not
            parse, and nothing of its text.
    """
    from syntactic_analysis.evaluation.ptb import PtbSentence

    domain = domain_of(path.name)
    out: list[PtbSentence] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        try:
            tree = parse_tree_line(line)
        except ValueError as exc:
            raise ValueError(f"{path.name} line {index + 1}: {exc}") from None
        cleaned = clean_gold_tree(tree)
        if cleaned is None:
            raise ValueError(f"{path.name} line {index + 1}: nothing but traces")
        cleaned = strip_unary_root_wrapper(cleaned)
        words, tags = zip(*cleaned.pos())
        out.append(
            PtbSentence(
                words=tuple(words),
                tags=tuple(tags),
                tree=" ".join(str(cleaned).split()),
                fileid=path.name,
                index=index,
                sent_id=f"{domain}-{index + 1}",
                domain=domain,
            )
        )
    return out


def _gold_set(paths: Sequence[Path]) -> "ConstituencyGoldSet":
    """MCTB_en as a ConstituencyGoldSet: the domain files `paths`, in that order.

    Private: syntactic_analysis.seal.open_dataset is its one caller, and it
    checks the lock first. `sha256` is taken over the raw bytes of the files
    in order, the way corpus_digest takes the WSJ sample's.
    """
    from syntactic_analysis.evaluation.gold import ConstituencyGoldSet

    digest = hashlib.sha256()
    sentences: list[PtbSentence] = []
    for path in paths:
        digest.update(path.read_bytes())
        sentences.extend(_read_domain(path))
    pretok = [list(sentence.words) for sentence in sentences]
    domains = [domain_of(path.name) for path in paths]
    return ConstituencyGoldSet(
        sentences=tuple(sentences),
        pretok=pretok,
        trees=[sentence.tree for sentence in sentences],
        tags=[list(sentence.tags) for sentence in sentences],
        sections=(),
        fileids=tuple(path.name for path in paths),
        sha256=digest.hexdigest(),
        n_sentences=len(sentences),
        n_tokens=sum(len(words) for words in pretok),
        source=(
            f"MCTB_en (Yang et al. 2022), domains {', '.join(domains)}; "
            f"{len(paths)} files, one tree per line, traces removed, function tags stripped"
        ),
        directory=paths[0].parent,
    )
