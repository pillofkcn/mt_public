"""The gold corpora: the UD EWT splits, and the Penn Treebank sample sections.

Split out of evaluation/harness.py on 9 Oct 2026. That module had grown four
unrelated jobs - loading gold data, building analyzers, running the measurement
loops, and writing CSVs with provenance - under a name that described none of
them. This is the first: the two gold sets a stage is scored against, how many
sentences each must contain, and what its checksum is. The preparation of the
Penn Treebank sample is evaluation/ptb.py.

The confirmatory test sets of version 1.0 (plan/v1.yaml) reach a stage by
another door: syntactic_analysis.seal.open_dataset, which refuses them until
the plan is locked. GENTLE is CoNLL-U and is read by load_gold_pretok; MCTB_en
is read by evaluation/mctb.py into a ConstituencyGoldSet.

Both loaders assert the sentence count on a full run. A corpus that silently
loads short turns every score into an average over a different denominator than
the one the thesis reports, and nothing downstream would notice.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from syntactic_analysis.checksum import sha256_of
from syntactic_analysis.data_loader import load_conllu
from syntactic_analysis.evaluation.ptb import (
    PtbSentence,
    WSJ_TEST_SECTIONS,
    corpus_digest,
    load_wsj_sentences,
    section_fileids,
)
from syntactic_analysis.models import DependencyParse
from syntactic_analysis.project import ewt_split_path
from syntactic_analysis.ptb_conventions import SECTION_SENTENCES


# --- gold standard -----------------------------------------------------------

# The UD_English-EWT test split, as the snapshot data/DATASET.md records it:
# what load_gold_pretok asserts when it loads the default split in full. Until
# 12 Oct 2026 the sentence count was a constant of four scripts, under three
# names, and of setup.sh.
EWT_TEST_SENTENCES = 2077
EWT_TEST_TOKENS = 25094


@dataclass(frozen=True)
class GoldSet:
    """The gold corpus plus everything needed to identify it later.

    Attributes:
        parse:       The loaded gold parse.
        pretok:      Word lists, one per sentence — the input every analyzer gets,
                     so tokenisation is identical for all of them.
        path:        Where it was loaded from.
        sha256:      Digest of the file, recorded in provenance. A corpus that
                     changes between releases silently changes every score.
        n_sentences: Sentence count.
        n_tokens:    Token count.
    """

    parse: DependencyParse
    pretok: list[list[str]]
    path: Path
    sha256: str
    n_sentences: int
    n_tokens: int

    def sentence(self, index: int) -> DependencyParse:
        """One gold sentence wrapped as a DependencyParse, for the metrics."""
        return DependencyParse(
            sentences=[self.parse.sentences[index]], analyzer_name="gold"
        )


def load_gold_pretok(
    path: str | Path | None = None,
    *,
    max_sentences: int | None = None,
    expect_sentences: int | None = None,
) -> GoldSet:
    """Load the gold corpus and build the pre-tokenised input.

    Args:
        path: CoNLL-U file. Defaults to the EWT test split.
        max_sentences: Truncate for a quick run. Use None for the full corpus;
            note the check is `is not None`, so 0 means "no sentences" rather than
            being silently read as "all of them".
        expect_sentences: Assert the corpus has this many sentences. Defaults to
            EWT_TEST_SENTENCES when the default split is loaded in full, which
            is what catches an HTML error page left in the .conllu by a failed
            download.

    Raises:
        FileNotFoundError: If the corpus is missing.
        ValueError: If `expect_sentences` does not match.
    """
    if not path and max_sentences is None and expect_sentences is None:
        expect_sentences = EWT_TEST_SENTENCES
    path = Path(path) if path else ewt_split_path("test")
    if not path.exists():
        raise FileNotFoundError(
            f"gold corpus not found: {path}. Run setup.sh to download it."
        )

    digest = sha256_of(path)
    parse = load_conllu(path)

    if max_sentences is not None:
        parse.sentences = parse.sentences[:max_sentences]

    pretok = [[tok.text for tok in sent.tokens] for sent in parse.sentences]
    n_tokens = sum(len(words) for words in pretok)

    if expect_sentences is not None and len(parse.sentences) != expect_sentences:
        raise ValueError(
            f"{path} has {len(parse.sentences)} sentences, expected "
            f"{expect_sentences}. The corpus may be truncated or a different "
            f"release; see data/DATASET.md."
        )

    return GoldSet(
        parse=parse,
        pretok=pretok,
        path=path,
        sha256=digest,
        n_sentences=len(parse.sentences),
        n_tokens=n_tokens,
    )


@dataclass(frozen=True)
class ConstituencyGoldSet:
    """A constituency gold standard: trees, and the input that produced them.

    Deliberately exposes `pretok` and `n_sentences` under the same names as
    `GoldSet`, so a stage reads the input of either one the same way. What it
    adds is `trees` — the thing the EWT does not have.

    Attributes:
        sentences:   The prepared gold sentences (see ptb.PtbSentence).
        pretok:      Word lists, one per sentence: the parsers' input.
        trees:       Cleaned single-line gold trees, aligned with `pretok`.
        tags:        Gold PTB POS tags, aligned with `pretok`.
        sections:    WSJ sections this came from, e.g. ("01",).
        fileids:     The corpus files read, in order.
        sha256:      Digest over those files' raw text (the corpus lives in
                     nltk_data, not in this repository).
        n_sentences: Sentence count.
        n_tokens:    Token count.
        source:      Human-readable identification for logs and provenance.
        directory:   Where the files were read from, for a corpus outside
                     nltk_data (MCTB_en, read by evaluation/mctb.py, with no
                     WSJ sections); None for the WSJ sample.
    """

    sentences: tuple[PtbSentence, ...]
    pretok: list[list[str]]
    trees: list[str]
    tags: list[list[str]]
    sections: tuple[str, ...]
    fileids: tuple[str, ...]
    sha256: str
    n_sentences: int
    n_tokens: int
    source: str
    directory: Path | None = None

    @property
    def path(self) -> Path:
        """Directory the treebank was read from, for the log header."""
        if self.directory is not None:
            return self.directory
        import nltk.corpus

        return Path(nltk.corpus.treebank.abspaths(self.fileids[0])[0]).parent


def load_gold_wsj(
    sections: Sequence[str] = WSJ_TEST_SECTIONS,
    *,
    max_sentences: int | None = None,
    expect_sentences: int | None = None,
) -> ConstituencyGoldSet:
    """Load WSJ gold trees from the NLTK Penn Treebank sample.

    Section 01 is the constituency test set: the NLTK sample is sections 00-01,
    and every neural parser here is trained on 02-21, so 01 is held out for all
    of them, while 00 is what the PCFG is induced from. See
    `syntactic_analysis.evaluation.ptb` for what "gold" means concretely — traces
    removed, function tags stripped, treebank bracket escapes kept.

    Args:
        sections: WSJ sections to load (default: the held-out "01").
        max_sentences: Truncate for a quick run; None for all.
        expect_sentences: Assert the count; defaults to the sample's known size
            when the whole of one or more sections is loaded.

    Raises:
        ValueError: On an unknown section or a sentence-count mismatch.
        LookupError: If the treebank sample is not installed (run setup.sh).
    """
    if expect_sentences is None and max_sentences is None:
        expect_sentences = sum(SECTION_SENTENCES[s] for s in sections)

    sentences = load_wsj_sentences(
        sections, max_sentences=max_sentences, expect_sentences=expect_sentences
    )
    fileids = section_fileids(sections)
    pretok = [list(s.words) for s in sentences]
    return ConstituencyGoldSet(
        sentences=tuple(sentences),
        pretok=pretok,
        trees=[s.tree for s in sentences],
        tags=[list(s.tags) for s in sentences],
        sections=tuple(sections),
        fileids=fileids,
        sha256=corpus_digest(fileids),
        n_sentences=len(sentences),
        n_tokens=sum(len(words) for words in pretok),
        source=(
            f"Penn Treebank sample (nltk), WSJ section(s) {', '.join(sections)}; "
            f"{len(fileids)} files, traces removed, function tags stripped"
        ),
    )

