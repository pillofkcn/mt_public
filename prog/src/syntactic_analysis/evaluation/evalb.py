"""Constituency scoring primitives: the EVALB parameter system and brackets.

Split out of evaluation/metrics.py on 9 Oct 2026. That module was named for
metrics and held two unrelated families: the POS and dependency metrics, which
compare token-aligned parses, and this - a configurable implementation of the
EVALB protocol, where which labels are deleted and which are treated as equal
is itself a parameter of the comparison. Keeping them together meant a reader
looking for "how is bracket F1 computed" had to walk past four attachment
scores first, and the EVALB parameter objects read as an aside.

Nothing here runs an analyzer or touches a corpus. The functions take tree
strings and return bracket counts; syntactic_analysis.stage is what runs a
parser and records them.

Version 1.0 scores through `con_sentence_counts`: one `ConCounts` per
sentence, under the parameter set the plan names (`plan_parameter_set`),
pooled by `metrics.corpus_metric`. It counts brackets as the EVALB program
does, repeats included. Until 23 Oct 2026 this module also held the scoring of
the stages before version 1.0, which kept a SET of spans per tree
(extract_spans, evalb_scores, evalb_counts, bracket_f1), and the helpers of
their reports and figures (tree depth, constituent counts, the consensus of
the parser-agreement run); they went with those stages.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from nltk import Tree

Span = tuple[str, int, int]


# --- Constituency ------------------------------------------------------------

# The root wrapper a parser puts around a whole sentence is not a constituent:
# its label is in the delete_labels of COLLINS (TOP, ROOT, S1), so it is deleted
# like any other such node and its children are scored.

# Preterminal labels that carry no part of speech. SuPar's constituency model
# predicts structure only and writes "_" in every preterminal, so a rule that
# deletes punctuation by its tag cannot see SuPar's punctuation at all; see
# _brackets_with_params for what that cost before it was noticed.
UNLABELLED_PRETERMINALS: frozenset[str] = frozenset({"_", "", "X", "XX"})

# Surface forms the Penn Treebank tags as punctuation, for the fallback above.
# Checked against WSJ section 01: this set deletes exactly the tokens the tag
# rule deletes, with no over-deletion and 5 misses out of 5206 punctuation
# tokens. The misses are a closing single quote, which cannot be added because
# the same character is the possessive marker 24 times over, tagged POS and
# rightly kept. A character-class rule ("all non-alphanumeric") was the obvious
# alternative and is worse: it also swallows $, %, & and #, which the treebank
# tags as ordinary words, costing 759 wrong deletions instead of 5 misses.
# Version 1.0's `con_sentence_counts` needs this list only where a parser's
# words are not the gold words: otherwise it first gives every untagged
# preterminal the gold tag of its word (`_with_gold_tags`), and the closing
# quote is deleted by its tag like any other punctuation.
PUNCTUATION_LEAVES: frozenset[str] = frozenset({
    ",", ":", ".", "?", "!", ";", "--", "-", "...", "``", "''", "`",
})

# Where evalb.c's modify_label cuts a nonterminal label: NP-SBJ-1 is NP.
_EVALB_LABEL_END = re.compile(r"[-=]")


class _FrozenLabelMap(Mapping):
    """A hashable, read-only ``str -> str`` mapping.

    ``EvalbParams`` is a frozen dataclass, so every field has to be hashable if the
    params are to be used as a dict key or cached; a plain dict is not. The pairs
    are kept sorted so that two maps built in a different order hash and compare
    equal.
    """

    __slots__ = ("_map", "_pairs")

    def __init__(self, pairs: Mapping[str, str] | Iterable[tuple[str, str]] = ()):
        self._map: dict[str, str] = dict(pairs)
        self._pairs: tuple[tuple[str, str], ...] = tuple(sorted(self._map.items()))

    def __getitem__(self, key: str) -> str:
        return self._map[key]

    def __iter__(self):
        return iter(self._map)

    def __len__(self) -> int:
        return len(self._map)

    def __hash__(self) -> int:
        return hash(self._pairs)

    def __eq__(self, other) -> bool:
        if isinstance(other, _FrozenLabelMap):
            return self._pairs == other._pairs
        if isinstance(other, Mapping):
            return self._map == dict(other)
        return NotImplemented

    def __repr__(self) -> str:
        return repr(dict(self._pairs))


@dataclass(frozen=True)
class EvalbParams:
    """Scoring parameters in the style of an EVALB ``.prm`` file.

    EVALB does not compare raw bracket sets. It first applies the rules in its
    parameter file, and the standard file for constituency work is COLLINS.prm,
    distributed with EVALB itself (https://nlp.cs.nyu.edu/evalb/) and used for the
    published PTB bracket F1 numbers. Without those rules the scores are not
    comparable to the literature, and worse, they charge parsers for annotation
    conventions instead of for parsing.

    Attributes:
        delete_labels: Labels whose node is deleted before scoring. The semantics
            differ by node type, exactly as in EVALB and in SuPar's
            ``Tree.factorize``:

              * **Nonterminal** — only the BRACKET disappears; the children are
                promoted to the parent and the yield is unchanged. ``TOP`` is
                deleted this way, which is what makes a wrapper cost nothing.
              * **Preterminal** — the LEAF goes with it. The word leaves the
                sentence entirely, so every span index to its right shifts down.
                That is what the DELETE_LABEL lines for the punctuation tags in
                COLLINS.prm mean: punctuation does not count toward span
                boundaries, so a parser is neither rewarded nor punished for
                where it attaches a comma.

            ``-NONE-`` is a preterminal here too, so empty-element traces are
            removed from the yield; a node whose whole yield was traces then has
            an empty span and produces no bracket at all.
        equal_labels: Label normalisation applied to the surviving nonterminals
            before comparison (EVALB EQ_LABEL). ``{"PRT": "ADVP"}`` means a PRT
            bracket is scored as if it were ADVP. The direction is irrelevant to
            the score as long as both sides are normalised the same way — it is a
            canonical form, not a rewrite of one side.
        name: Short identifier for CSV column names and provenance records, e.g.
            "collins", "collins_nml".

    Function tags are NOT handled here. ``NP-SBJ`` is stripped to ``NP`` by the
    caller before scoring (``strip_function_tags`` in ptb_conventions.py), for
    every tree on both sides, so by the time spans are extracted the labels are
    already bare categories.
    """

    delete_labels: frozenset[str] = frozenset()
    equal_labels: Mapping[str, str] = _FrozenLabelMap()
    name: str = "custom"

    def __post_init__(self) -> None:
        """Freeze whatever the caller passed, so presets can be written literally."""
        object.__setattr__(self, "delete_labels", frozenset(self.delete_labels))
        object.__setattr__(self, "equal_labels", _FrozenLabelMap(self.equal_labels))

    def normalize_label(self, label: str) -> str:
        """Apply ``equal_labels`` to one label."""
        return self.equal_labels.get(label, label)


COLLINS = EvalbParams(
    name="collins",
    delete_labels=frozenset({
        "TOP", "ROOT", "S1", "-NONE-", ",", ":", "``", "''", ".", "?", "!",
    }),
    equal_labels={"PRT": "ADVP"},
)
"""COLLINS.prm: DELETE_LABEL TOP, DELETE_LABEL -NONE-, the five punctuation tags
(comma, colon, open and close quote, period) and EQ_LABEL ADVP PRT.

Extended by ROOT and S1 — wrappers the .prm file never needed but that Stanza and
SuPar emit — and by ? and !, which are punctuation by exactly the argument that
deletes "." and which the delete set in SuPar's ``Tree.factorize`` also carries.
SuPar's set has the mirror-image gap: it lists TOP and S1 but not ROOT.

With the wrappers in ``delete_labels`` no wrapper has to be stripped first —
deleting a nonterminal promotes its children, which for a unary wrapper is
precisely stripping it, and it also works when the root has several children,
as in (TOP (S ...) (. .)).

The empty label "" that raw PTB files carry (``( (S ...))``) is not in the set;
NLTK's treebank reader already resolves it to the real root label. A corpus read
some other way needs it added.
"""

COLLINS_NML = EvalbParams(
    name="collins_nml",
    delete_labels=COLLINS.delete_labels | {"NML"},
    equal_labels=COLLINS.equal_labels,
)
"""COLLINS plus NML deletion — the headline configuration for the gold-tree run.

NML marks a nominal modifier inside an NP ("(NP (DT the) (NML (NN stock) (NN
market)))"). It does not exist in the original Penn Treebank annotation that the
NLTK sample ships, and was introduced by the 2006 revision that Benepar's
``benepar_en3`` and Stanza's constituency model are trained on. Scored strictly,
every NML bracket those parsers emit is a false positive against a gold tree that
had no way to contain one — a difference of annotation guideline, not a parsing
error, and one that hits two of the four parsers only.

Deleting NML promotes its children, so the surrounding NP is still scored and the
parsers are compared on the structure the gold standard actually annotates.
Version 1.0 scores constituents under this set only (plan/v1.yaml tasks.con.metric);
the strict COLLINS score was reported by the stages before it.
"""


def _parsed_tree(tree_str: str, expected_leaves: int | None):
    """Parse a bracketed tree and check its leaf count. None when there is none.

    ``Tree.fromstring`` does NOT raise on unescaped bracket tokens — it silently
    drops them and shifts every subsequent offset — so without the leaf count a
    corrupt tree yields plausible-looking brackets. The check is made against
    the ORIGINAL leaves, before any deletion: it is a contract check on what the
    parser emitted, not on what is scored.

    Raises:
        ValueError: If the tree cannot be parsed, or its leaf count disagrees
            with `expected_leaves`.
    """
    if not tree_str or not tree_str.strip():
        return None

    try:
        tree = Tree.fromstring(tree_str)
    except (ValueError, IndexError) as exc:
        raise ValueError(f"unparseable constituency tree: {exc}") from exc

    if expected_leaves is not None and len(tree.leaves()) != expected_leaves:
        raise ValueError(
            f"tree has {len(tree.leaves())} leaves, expected {expected_leaves} — "
            f"unescaped bracket tokens are the usual cause"
        )
    return tree


def _brackets_with_params(
    tree, params: EvalbParams, *, evalb_labels: bool = False
) -> tuple[list[Span], list[str]]:
    """Every bracket under the parameter set, repeats kept, and the surviving words.

    Deletion is applied during the traversal rather than by rebuilding the tree,
    which is the same thing: not emitting a node's bracket while still walking its
    children IS promoting the children to the parent. That holds at the root too —
    a deleted root with one child leaves that child as the outermost bracket, and
    a deleted root with several children leaves a forest whose members are all
    scored, with no span for the root itself.

    A node whose yield is empty after deletion produces no span: ``(NP (-NONE- *))``
    is not a zero-width NP bracket, it is an NP that is not there at all.

    Punctuation is deleted by its TAG, which is what EVALB's ``DELETE_LABEL``
    means — but one parser here emits no tags at all. SuPar's constituency model
    predicts structure only and writes ``_`` in every preterminal, so tag-based
    deletion removed punctuation from the gold and left it in SuPar's trees,
    shifting every span index after the first comma and scoring the parser at
    0.35 where it belongs near 0.94. When a preterminal carries no usable tag the
    deletion therefore falls back to the WORD, which is the rule's actual intent:
    punctuation tokens do not count toward span boundaries. The punctuation
    labels and the punctuation words coincide (``,`` ``.`` ``''``), so the same
    set answers both questions.

    A unary chain that repeats one label over one span - (S (S ...) (. .)) once
    the period is deleted, (NP (NP ...)) - is two brackets here, as it is in
    EVALB (calc_result matches each gold bracket against a test bracket not yet
    matched, so repeats are a multiset). It returns the words rather than their
    number because EVALB compares the words of the two trees once the deletions
    are applied, and calls a sentence whose words differ an error ("Words
    unmatch") even where their number agrees.

    With `evalb_labels`, a nonterminal label is first cut at its first "-" or
    "=", as evalb.c's modify_label does before both the DELETE_LABEL check and
    the comparison: NP-SBJ and NP=2 are NP, NML-1 is NML, and a nonterminal
    labelled -NONE- becomes a bracket with an empty label. A preterminal label
    is used as it is, as read_line uses it.
    """
    brackets: list[Span] = []
    words: list[str] = []

    def _walk(node) -> None:
        if isinstance(node, str):
            words.append(node)
            return
        label = node.label()
        if len(node) == 1 and isinstance(node[0], str):
            deleted = label in params.delete_labels
            if not deleted and label in UNLABELLED_PRETERMINALS:
                deleted = node[0] in PUNCTUATION_LEAVES
            # Preterminal: the leaf is deleted along with the tag, so the word
            # stops occupying a position and later spans shift left.
            if not deleted:
                words.append(node[0])
            return
        if evalb_labels:
            label = _EVALB_LABEL_END.split(label, maxsplit=1)[0]
        deleted = label in params.delete_labels
        start = len(words)
        for child in node:
            _walk(child)
        end = len(words)
        if not deleted and end > start:
            brackets.append((params.normalize_label(label), start, end))

    _walk(tree)
    return brackets, words


# --- version 1.0: per-sentence counts ------------------------------------------
# What a v1 constituency stage records for each sentence, and what
# metrics.corpus_metric sums into the corpus F1. It counts brackets as EVALB
# does - a multiset, after the parameter set, with an error sentence kept out
# of the totals - so the corpus F1 is the EVALB program's figure.

# The parameter set the plan's constituency metric names. plan/v1.yaml states
# tasks.con.metric as `evalb: collins` and `nml: collapse_both_sides`, and this
# table is where those two words become an EvalbParams; plan.py accepts no other
# pair. COLLINS_NML keeps its name "collins_nml", which the CSVs and provenance
# records of the existing stages carry.
_PLAN_PARAMETER_SETS: Mapping[tuple[str, str], EvalbParams] = MappingProxyType({
    ("collins", "collapse_both_sides"): COLLINS_NML,
})


def plan_parameter_set(options: Mapping[str, str]) -> EvalbParams:
    """The EVALB parameter set a plan's constituency metric names.

    Args:
        options: The metric's options as the plan states them -
            `plan.tasks["con"].metric.options`, which holds `evalb` and `nml`.

    Raises:
        ValueError: If the pair names no parameter set defined here.
    """
    key = (options.get("evalb", ""), options.get("nml", ""))
    if key not in _PLAN_PARAMETER_SETS:
        known = [f"evalb={e}, nml={n}" for e, n in _PLAN_PARAMETER_SETS]
        raise ValueError(
            f"unknown EVALB parameter set evalb={key[0]}, nml={key[1]}; "
            f"expected one of {known}"
        )
    return _PLAN_PARAMETER_SETS[key]


@dataclass(frozen=True)
class ConCounts:
    """One sentence's bracket counts under one parameter set (or a sum of them).

    Attributes:
        matched:     Brackets the two trees share - label (after
                     ``equal_labels``), start and end - counted as a multiset.
        gold:        Brackets of the gold tree after the parameter set.
        test:        Brackets of the parser's tree after the parameter set.
        parsed:      1 if the parser produced a tree, 0 if it did not.
        evalb_error: 1 if the two trees no longer have the same words once the
                     deletions are applied - EVALB's error sentence ("Length
                     unmatch", "Words unmatch"). EVALB keeps such a sentence
                     out of its bracket totals, and so does this: matched,
                     gold and test are 0 and only this column says why.

    A sentence with no tree is matched 0, test 0 and its gold brackets: the
    parser is charged for it in recall. EVALB itself skips a sentence whose test
    line is empty (Status 2) and counts nothing for it; version 1.0 counts the
    gold brackets, so a parser cannot raise its score by returning nothing.

    A tree that keeps no word once the deletions are applied is EVALB's other
    skip (Status 2, the test side has no words): all four counts but `parsed`
    are 0 and it is not an error sentence, so `evalb_error` sums to EVALB's
    "Number of Error sentence".

    Every field is a count, so a sum of ConCounts is a ConCounts:
    `metrics.corpus_metric` pools them that way.
    """

    matched: int
    gold: int
    test: int
    parsed: int
    evalb_error: int = 0

    def __post_init__(self) -> None:
        """Refuse counts no tree pair can produce."""
        values = (self.matched, self.gold, self.test, self.parsed, self.evalb_error)
        if any(value < 0 for value in values):
            raise ValueError(f"negative bracket count: {self}")
        if self.matched > min(self.gold, self.test):
            raise ValueError(f"more matched brackets than gold or test ones: {self}")


def con_sentence_counts(
    predicted: str | None,
    gold: str,
    *,
    params: EvalbParams,
    expected_leaves: int | None = None,
) -> ConCounts:
    """Bracket counts of one sentence, as EVALB counts them under `params`.

    Args:
        predicted: The parser's tree in Penn Treebank notation; None or a blank
            string when it produced none.
        gold: The gold tree, function tags already stripped (see EvalbParams).
        params: The parameter set; for version 1.0 the one the plan names,
            `plan_parameter_set(plan.tasks["con"].metric.options)`. There is no
            default, so a stage cannot score under a parameter set the plan
            does not state.
        expected_leaves: Token count both trees must have before any deletion
            (the leaf-count refusal of `_parsed_tree`). Pass it whenever the
            caller knows the input, as a stage always does.

    What is counted, as evalb.c (https://nlp.cs.nyu.edu/evalb/) counts it
    under COLLINS.prm:

      * every nonterminal is a bracket, the outermost one included; a node
        whose label is in ``delete_labels`` is not, and its children stay;
      * a nonterminal label is cut at its first "-" or "=" before both
        (modify_label): NP-SBJ is NP, NML-1 is NML;
      * a preterminal whose label is in ``delete_labels`` takes its word out of
        the sentence;
      * a bracket left with no words is not a bracket;
      * labels are compared after ``equal_labels`` (EQ_LABEL ADVP PRT);
      * repeated brackets are counted as often as they occur, and a gold
        bracket matches at most one test bracket;
      * a test tree with no word left is skipped (Status 2): nothing counted;
      * if the surviving words differ, the sentence is an error sentence.

    One rule EVALB does not have, for a parser that writes no tags (SuPar
    writes "_" in every preterminal): when the parser's words are the gold
    words, each untagged preterminal first takes the gold tag of its word, as
    such a parser's trees are given to EVALB. Its punctuation is then deleted
    by tag on both sides and it can have no error sentence. Where the words
    differ, the word list `PUNCTUATION_LEAVES` decides instead, as
    `_brackets_with_params` describes.

    Raises:
        ValueError: If the gold tree is empty, either tree cannot be parsed or
            has a shape evalb.c cannot read (a node with no children, or a
            word beside a subtree), or a leaf count disagrees with
            `expected_leaves`.
    """
    gold_tree = _parsed_tree(gold, expected_leaves)
    if gold_tree is None:
        raise ValueError("gold tree is empty")
    _check_evalb_can_read(gold_tree)
    gold_brackets, gold_words = _brackets_with_params(gold_tree, params, evalb_labels=True)

    predicted_tree = _parsed_tree(predicted or "", expected_leaves)
    if predicted_tree is None:
        return ConCounts(matched=0, gold=len(gold_brackets), test=0, parsed=0)
    _check_evalb_can_read(predicted_tree)

    test_brackets, test_words = _brackets_with_params(
        _with_gold_tags(predicted_tree, gold_tree), params, evalb_labels=True
    )
    if not test_words:
        return ConCounts(matched=0, gold=0, test=0, parsed=1)
    if test_words != gold_words:
        return ConCounts(matched=0, gold=0, test=0, parsed=1, evalb_error=1)

    matched = sum((Counter(gold_brackets) & Counter(test_brackets)).values())
    return ConCounts(
        matched=matched, gold=len(gold_brackets), test=len(test_brackets), parsed=1
    )


def _check_evalb_can_read(tree) -> None:
    """Refuse a tree whose shape evalb.c reads as something else.

    NLTK reads both shapes below; evalb.c does not. ``(NP )`` is a node with no
    children to NLTK and a preterminal with an empty word to evalb.c's
    read_line, which then finds the lengths unequal. A word beside a subtree,
    ``(NP (DT the) cat)``, sends read_line into Error("Reading sentence")
    without advancing until MAX_ERROR stops the run. Such a tree is refused
    like an unparseable one rather than scored by rules EVALB never applied.

    Raises:
        ValueError: On the first such node.
    """
    for node in tree.subtrees():
        if len(node) == 0:
            raise ValueError(f"tree EVALB cannot read: ({node.label()} ) has no children")
        if len(node) > 1 and any(isinstance(child, str) for child in node):
            raise ValueError(
                f"tree EVALB cannot read: ({node.label()} ...) has a word beside "
                f"another child"
            )


def _with_gold_tags(predicted_tree, gold_tree):
    """The predicted tree with the gold tag on each preterminal that has none.

    Only where the two trees have the same words, position by position; the
    predicted tree is returned unchanged otherwise. A preterminal the parser
    did tag keeps its tag, so a tagging parser's mistagged punctuation still
    makes an error sentence, as it does in EVALB.
    """
    if predicted_tree.leaves() != gold_tree.leaves():
        return predicted_tree
    tree = predicted_tree.copy(deep=True)
    gold_tags = [tag for _, tag in gold_tree.pos()]
    for leaf, tag in zip(tree.treepositions("leaves"), gold_tags):
        node = tree[leaf[:-1]]
        if len(node) == 1 and node.label() in UNLABELLED_PRETERMINALS:
            node.set_label(tag)
    return tree
