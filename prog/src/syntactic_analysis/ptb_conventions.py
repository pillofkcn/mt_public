"""Penn Treebank annotation conventions: its sections, its labels, its traces, its tokens.

Three rules that two very different callers both need, and both had written out
themselves until 9 Oct 2026:

  * `evaluation/ptb.py`, which prepares the gold trees the constituency
    experiment is scored against;
  * `wrappers/nltk.py`, which induces a PCFG from the same corpus.

The two copies had a byte-identical regex and two different guards in front of
it, and `evaluation/ptb.py` said so in its own docstring: "two separate code
paths ... has to be checked against the other by hand". They are one path now.

A wrapper must not import from `evaluation/`, which is why this sits at the top
level of the package beside `checksum.py` rather than inside the evaluation
subpackage. It is also why it holds only conventions: no corpus is loaded here,
no score is computed, and nothing depends on which experiment is running.

Before the two were merged, their outputs were compared over the whole sample:
707 distinct node labels, 3914 trees and 167545 induced productions, with no
disagreement anywhere. Unifying them therefore moved no number, and that is a
measured claim rather than an argument from reading the code.

The token convention of a constituency test set (`parser_tokens`,
`restore_words`) was added on 24 Oct 2026: how a treebank writes brackets and
double quotes, which plan/v1.yaml declares per dataset, and the one convention
every constituency configuration is given its input in.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from syntactic_analysis.normalize import BRACKET_ESCAPES
from syntactic_analysis.offline import nltk_package

# How many sentences each section of the NLTK sample holds, after trace removal.
# The sample is sections 00 and 01 only; anything else is a caller's mistake,
# and an unknown section silently resolves to zero fileids, which would induce a
# grammar from nothing or score against nothing.
SECTION_SENTENCES: dict[str, int] = {"00": 1921, "01": 1993}

# Fileids look like "wsj_0001.mrg"; characters 4:6 are the section.
SECTION_FILE_SLICE = slice(4, 6)

# The POS tag marking an empty category. Its subtree carries no words.
TRACE_TAG = "-NONE-"

# Labels that ARE hyphenated categories. They start with the hyphen, so the
# suffix pattern below cannot match them anyway - the set is kept because it
# states the intent, and because a label like "-FOO-BAR" would otherwise be
# silently shortened. Until 12 Oct 2026 the seven were spelled out here a
# second time.
PROTECTED_LABELS = frozenset(BRACKET_ESCAPES.values()) | {TRACE_TAG}

# Everything from the first "-" or "=" that follows a category letter, to the
# end. PTB stacks suffixes (NP-SBJ-1, PP-LOC-PRD, ADVP-PRD-LOC=3), so this has
# to be greedy rather than a single "-[A-Z]+" group. The lookbehind means a
# label that STARTS with "-" is never touched, and the restricted character
# class means CNF and collapsed-unary artefacts (NP@<DT-NN>, S^<VP>) fail to
# match and stay visible instead of being mangled into something plausible.
FUNCTION_TAG_RE = re.compile(r"(?<=[A-Z])[-=][A-Z0-9=\-]*$")


def strip_function_tags(label: str) -> str:
    """Reduce a PTB node label to its bare category: 'NP-SBJ-1' -> 'NP'.

    Function tags, coindexing indices and gap markers are annotation that no
    parser here emits, so a span labelled "NP-SBJ" could never match another
    parser's "NP" and would be scored as a miss for both sides. Induced into a
    grammar they fragment one category into dozens.

    Returns the base category, or the label unchanged when it is protected or is
    not a category-plus-suffix at all.

    >>> [strip_function_tags(x) for x in ("NP-SBJ", "NP-SBJ-1", "PP-LOC-PRD")]
    ['NP', 'NP', 'PP']
    >>> [strip_function_tags(x) for x in ("VP-2", "NP=2", "ADVP-PRD-LOC=3")]
    ['VP', 'NP', 'ADVP']
    >>> [strip_function_tags(x) for x in ("NP", "PRP$", "WP$", "S", ",", "``")]
    ['NP', 'PRP$', 'WP$', 'S', ',', '``']
    >>> [strip_function_tags(x) for x in ("-LRB-", "-RRB-", "-NONE-")]
    ['-LRB-', '-RRB-', '-NONE-']
    """
    if label in PROTECTED_LABELS or label.startswith("-"):
        return label
    return FUNCTION_TAG_RE.sub("", label, count=1)


def strip_traces(tree):
    """Remove `-NONE-` subtrees, returning None when nothing is left.

    Empty categories are coindexing annotation. Induced as they stand they
    become lexical rules over terminals ("*-1", "*T*-2", "0", "*U*") that no
    input can match, and because induction is plain maximum likelihood that
    probability mass comes straight off the rules that could have fired. They
    also distort the phrasal rules around them: `(NP (-NONE- *))` counts as a
    genuine one-child NP.

    Removing a trace can empty its parent, so the parent goes too - an empty
    bracket is not something any parser can be scored on.

    Args:
        tree: An nltk Tree, or a leaf string (returned unchanged).
    """
    from nltk import Tree

    if isinstance(tree, str):
        return tree
    if tree.label() == TRACE_TAG:
        return None
    children = [c for c in (strip_traces(child) for child in tree) if c is not None]
    if not children:
        return None
    return Tree(tree.label(), children)


def clean_gold_tree(tree):
    """Both gold preparations: traces removed, then function tags stripped.

    Returns a new Tree, or None if the sentence was nothing but empty
    categories. Preterminal labels are left untouched: a POS tag is not a
    phrase label, and stripping it would turn "PRP$" into "PRP".
    """
    from nltk import Tree

    stripped = strip_traces(tree)
    if stripped is None:
        return None

    def relabel(node):
        if isinstance(node, str):
            return node
        is_preterminal = len(node) == 1 and isinstance(node[0], str)
        label = node.label() if is_preterminal else strip_function_tags(node.label())
        return Tree(label, [relabel(child) for child in node])

    return relabel(stripped)


def section_fileids(sections: Sequence[str], *, fetch: bool = True) -> tuple[str, ...]:
    """The sample's fileids for the given WSJ sections, in corpus order.

    Returns a tuple, which is hashable - the PCFG induction makes it part of an
    lru_cache key.

    The sample is NLTK's data package treebank, downloaded when it is missing
    except during a run, which refuses instead (offline.nltk_package). With
    fetch=False nothing is downloaded at all, in a run or not: the
    declaration of the files a load reads (NltkAnalyzer.downloaded_files,
    since 24 Oct 2026) reads what is on disk and fetches nothing.

    Raises:
        ValueError: If a section is not in the sample. Caught here rather than
            at use time, because an unknown section resolves to zero fileids
            and the caller would then work from the whole corpus or from
            nothing without being told.
        offline.NotCachedError: During a run, if the package is not installed.
        LookupError: With fetch=False, if the package is not installed.
    """
    unknown = [s for s in sections if s not in SECTION_SENTENCES]
    if unknown:
        raise ValueError(
            f"unknown WSJ section(s) {unknown}; expected one of "
            f"{sorted(SECTION_SENTENCES)}, the sections of the nltk treebank sample"
        )
    import nltk.corpus

    if fetch:
        nltk_package("treebank", "corpora/treebank")
    wanted = set(sections)
    return tuple(
        fileid
        for fileid in nltk.corpus.treebank.fileids()
        if fileid[SECTION_FILE_SLICE] in wanted
    )


def section_files(sections: Sequence[str]) -> list[Path]:
    """The files of the treebank sample NLTK reads for the given WSJ sections, where it finds them.

    The corpus reader's own paths (nltk.corpus.treebank.abspath) for the
    fileids of the sections (section_fileids), in corpus order, or the
    package's zip file when NLTK reads it zipped. They are the files the PCFG
    is induced from (WSJ 00, NltkAnalyzer.downloaded_files) and the files the
    gold set of a dataset read through NLTK is read from (WSJ 01,
    evaluation/ptb.py, load_wsj_sentences), which plan/LOCK.json records under
    `data` (seal.dataset_files). Nothing is downloaded. Moved here from
    wrappers/nltk.py (_treebank_files) on 25 Oct 2026, before the lock, so
    that both declarations read the same function.

    Raises:
        ValueError: If a section is not in the sample.
        LookupError: If NLTK does not find the package.
    """
    fileids = section_fileids(sections, fetch=False)
    import nltk.corpus

    reader = nltk.corpus.treebank
    zipped = getattr(reader.root, "zipfile", None)
    if zipped is not None:
        return [Path(zipped.filename)]
    return [Path(reader.abspath(fileid).path) for fileid in fileids]


# --- the token convention of a constituency test set -------------------------
# How a treebank writes the tokens that are punctuation in running text. Each
# constituency dataset of plan/v1.yaml declares it (datasets.<id>.tokens), and
# plan.py accepts these values only:
#
#   brackets  ptb  a parenthesis is the leaf -LRB- or -RRB-
#                  (normalize.BRACKET_ESCAPES). The one convention a bracketed
#                  tree can hold: a raw "(" leaf would be read as structure.
#   quotes    ptb  a double quote is `` where it opens and '' where it closes,
#                  as WSJ 01 writes it;
#             raw  a double quote is " either way, as MCTB_en writes it (the
#                  one format check allowed before the lock, 24 Oct 2026:
#                  1016 leaves ", none `` or ''; CHANGELOG_v1.md).
#
# Every constituency configuration is given its sentence in the PTB
# convention, the one it is given on WSJ 01 (`parser_tokens`), so that a
# difference between WSJ 01 and a test set written otherwise is not a
# difference of token convention. The tree it returns is scored against the
# gold tree with the gold words (`restore_words`).
BRACKET_CONVENTIONS: tuple[str, ...] = ("ptb",)
QUOTE_CONVENTIONS: tuple[str, ...] = ("ptb", "raw")
PTB_OPENING_QUOTE = "``"
PTB_CLOSING_QUOTE = "''"
RAW_QUOTE = '"'


def parser_tokens(words: Sequence[str], *, brackets: str, quotes: str) -> list[str]:
    """One sentence of a test set as a constituency parser is given it: in the PTB convention.

    Args:
        words:    The sentence's gold tokens, as the test set writes them.
        brackets: The test set's bracket convention (BRACKET_CONVENTIONS).
        quotes:   Its quote convention (QUOTE_CONVENTIONS).

    With quotes "ptb" the tokens are returned unchanged: on WSJ 01 the
    conversion is the identity. With "raw", every token that is exactly a
    double quote becomes `` or '' by its place among the sentence's double
    quotes: the first opens, the second closes, the third opens again, and so
    on. Every other token is returned as it is.

    The rule reads the sentence's tokens and nothing else - no gold tag, no
    gold tree - so nothing of the gold standard a parser is scored against
    reaches its input. Its known weakness is a quotation that spans sentences:
    a sentence that begins inside a quotation has its quotes the wrong way
    round, and a sentence with an odd number of them ends on a quote that
    may be the wrong one. Measured on WSJ 01, with its `` and '' read as ",
    the rule gives back the treebank's own token for 587 of its 601 double
    quotes; the 14 others are in 14 of the 302 sentences that hold a double
    quote (37 of the 302 hold an odd number). Whitespace, which a tokenizer
    of running text reads an opening quote from, is not in a treebank.

    Raises:
        ValueError: For a convention this module has no rule for.
    """
    if brackets not in BRACKET_CONVENTIONS:
        raise ValueError(
            f"unknown bracket convention {brackets!r}; expected one of {list(BRACKET_CONVENTIONS)}"
        )
    if quotes not in QUOTE_CONVENTIONS:
        raise ValueError(
            f"unknown quote convention {quotes!r}; expected one of {list(QUOTE_CONVENTIONS)}"
        )
    if quotes == "ptb":
        return list(words)
    given: list[str] = []
    seen = 0
    for word in words:
        if word == RAW_QUOTE:
            word = PTB_OPENING_QUOTE if seen % 2 == 0 else PTB_CLOSING_QUOTE
            seen += 1
        given.append(word)
    return given


def restore_words(tree: str, words: Sequence[str], given: Sequence[str]) -> str:
    """A parser's tree with the gold words put back where `parser_tokens` changed them.

    Args:
        tree:  The tree the parser returned for `given`, one leaf per token;
               "" when it returned none.
        words: The sentence's gold tokens.
        given: What the parser was given: parser_tokens(words, ...).

    Leaf i of the tree is token i of the sentence, so the gold word goes back
    position by position, and only at the positions where `given` differs from
    `words`. Every other leaf stays as the parser wrote it, exactly as on a
    test set with nothing to convert, so a parser's tree is scored the same
    way on WSJ 01 and on a test set written otherwise. A tree with nothing to
    put back is returned as it is, without being parsed: on WSJ 01 nothing
    reaches the scoring that did not reach it before.

    Raises:
        ValueError: If `words` and `given` differ in length, or the tree
            cannot be read or has not one leaf per token: the gold words could
            not then be put back position by position, and the tree is not
            scored.
    """
    if len(words) != len(given):
        raise ValueError(
            f"{len(given)} tokens given for {len(words)} gold words; the conversion "
            f"changes no token count"
        )
    changed = [index for index, (word, sent) in enumerate(zip(words, given)) if word != sent]
    if not tree or not changed:
        return tree
    from nltk import Tree

    try:
        parsed = Tree.fromstring(tree)
    except (ValueError, IndexError) as exc:
        raise ValueError(f"unparseable constituency tree: {exc}") from exc
    leaves = parsed.treepositions("leaves")
    if len(leaves) != len(words):
        raise ValueError(
            f"tree has {len(leaves)} leaves for {len(words)} tokens; the gold words cannot "
            f"be put back position by position"
        )
    for index in changed:
        parsed[leaves[index]] = words[index]
    return " ".join(str(parsed).split())
