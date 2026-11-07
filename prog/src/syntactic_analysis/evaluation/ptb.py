"""Penn Treebank gold trees — the constituency ground truth this project can use.

The EWT has no constituency layer, which is why the constituency stage scores
inter-parser agreement rather than accuracy. But NLTK ships a 10% sample of the
Penn Treebank, and that sample is **WSJ sections 00 and 01** (`wsj_0001.mrg` …
`wsj_0199.mrg`): 1921 and 1993 sentences. Every neural constituency parser in this
comparison — Stanza, Benepar, SuPar — is trained on the standard split, sections
**02-21**, so section 01 is data none of them was trained on and all of them can
be scored against with real EVALB bracket F1. The PCFG is induced from section 00,
so it is held out for that parser too.

That gives the constituency stage two test sets answering two different questions:

  * WSJ 01  — labeled bracket F1 against gold trees, in-domain newswire. An
              absolute quality measurement, comparable to published numbers.
  * EWT     — inter-parser agreement and coverage on web text, out-of-domain for
              every parser. No ground truth, so no accuracy claim.

and, running both, a check on whether agreement tracks quality at all.

What this module does NOT do is pretend the sample is the full treebank. It is
10% of it, sections 00-01 only, distributed with NLTK under the LDC sample
licence; the thesis says so wherever a WSJ number appears.

Gold-tree preparation follows the standard evaluation conventions:

  * `-NONE-` subtrees (empty categories: `*T*-1`, `*`, `0`, `*U*`) are removed,
    because no parser under test emits them and they are not part of any
    parser's output vocabulary. A node left with no children after the removal is
    removed in turn.
  * Function tags and coindexing are stripped from nonterminal labels
    (`NP-SBJ-1` -> `NP`, `PP-LOC-PRD` -> `PP`), because parsers predict bare
    categories. Preterminal (POS) labels are left alone — `-NONE-` is a POS tag,
    not a function tag, and tags like `PRP$` must survive intact.
  * Leaves keep the treebank's own bracket escapes (`-LRB-`, `-RRB-`). They are
    the input the parsers are given as well: all three neural parsers were
    trained on PTB text where a parenthesis IS the token `-LRB-`, and the PCFG's
    lexicon contains that form and not `(`. Feeding the surface `(` instead would
    turn a tokenisation convention into an out-of-vocabulary failure.

The preparation itself lives in `syntactic_analysis.ptb_conventions`, shared
with `wrappers/nltk.py`, which induces a PCFG from the same corpus. Until
9 Oct 2026 the two had a copy each and this docstring warned that "a change to
either one has to be checked against the other by hand"; they were compared
over the whole sample before being merged - 707 labels, 3914 trees and 167545
productions, with no disagreement anywhere.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

from syntactic_analysis.ptb_conventions import (
    TRACE_TAG,
    clean_gold_tree,
    section_fileids,
)

# The held-out WSJ section. 00 is what the PCFG is induced from and 02-21 is what
# every neural parser was trained on, so 01 is the one section none of the four
# has seen.
WSJ_TEST_SECTIONS: tuple[str, ...] = ("01",)


def corpus_digest(fileids: Iterable[str]) -> str:
    """SHA-256 over the raw text of the given treebank files, in order.

    The gold corpus for the WSJ run lives inside nltk_data rather than in this
    repository, so this is the only way its identity can be recorded in
    provenance: a sidecar of the WSJ run carries this digest of section 01 as
    its gold_sha256, where a sidecar of the EWT carries that file's checksum.
    """
    import nltk.corpus

    digest = hashlib.sha256()
    for fileid in fileids:
        digest.update(nltk.corpus.treebank.raw(fileid).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class PtbSentence:
    """One prepared gold sentence.

    Attributes:
        words:  Leaves after trace removal — the pre-tokenised input every
                parser is given, in treebank form (`-LRB-`, not `(`).
        tags:   Gold PTB POS tags, aligned 1:1 with `words`. Used as the optional
                gold-tag input to the PCFG, which isolates the grammar's quality
                from its tagger's errors.
        tree:   The cleaned gold tree as a single-line bracketed string.
        fileid: Source file, e.g. "wsj_0101.mrg".
        index:  Position of the sentence within that file.
        sent_id: The sentence's id where the corpus has no file structure to
                stand in for one: "<domain>-<n>" in MCTB_en (evaluation/mctb.py),
                "" for WSJ, where fileid and index identify a sentence.
        domain: The MCTB_en domain the sentence comes from, "" for WSJ.
    """

    words: tuple[str, ...]
    tags: tuple[str, ...]
    tree: str
    fileid: str
    index: int
    sent_id: str = ""
    domain: str = ""


def load_wsj_sentences(
    sections: Sequence[str] = WSJ_TEST_SECTIONS,
    *,
    max_sentences: int | None = None,
    expect_sentences: int | None = None,
) -> list[PtbSentence]:
    """Load and prepare the gold sentences of the given WSJ sections.

    Args:
        sections: Which sections of the NLTK sample to load. The default "01" is
            the held-out test section; "00" is what the PCFG is induced from and
            must not be used as a test set for it.
        max_sentences: Truncate for a quick run (None = all).
        expect_sentences: Assert the prepared count. load_gold_wsj passes the
            sample's known size on a full load, so a stage does not have to.

    Returns the sentences in corpus order. Sentences that consisted only of empty
    categories are dropped (there are none in sections 00-01, but the code does
    not assume that).

    Raises:
        ValueError: On an unknown section or a failed `expect_sentences` check.
    """
    fileids = section_fileids(sections)

    import nltk.corpus

    out: list[PtbSentence] = []
    for fileid in fileids:
        for index, tree in enumerate(nltk.corpus.treebank.parsed_sents(fileid)):
            cleaned = clean_gold_tree(tree)
            if cleaned is None:
                continue
            pos = [
                (word, tag)
                for word, tag in tree.pos()
                if tag != TRACE_TAG
            ]
            words = tuple(word for word, _ in pos)
            if words != tuple(cleaned.leaves()):
                # Would mean trace removal and the pos() filter disagree, i.e. a
                # tree shape this preparation does not understand. Loud, because
                # a silent misalignment here shifts every span index.
                raise ValueError(
                    f"{fileid}[{index}]: gold tags do not align with gold leaves"
                )
            out.append(
                PtbSentence(
                    words=words,
                    tags=tuple(tag for _, tag in pos),
                    tree=" ".join(str(cleaned).split()),
                    fileid=fileid,
                    index=index,
                )
            )
            if max_sentences is not None and len(out) >= max_sentences:
                return out

    if expect_sentences is not None and len(out) != expect_sentences:
        raise ValueError(
            f"WSJ sections {tuple(sections)} prepared {len(out)} sentences, "
            f"expected {expect_sentences}; the installed nltk treebank sample "
            f"differs from the one this project was measured on"
        )
    return out
