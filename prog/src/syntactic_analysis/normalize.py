"""Output normalisation and the analyzer output contract.

Every wrapper's raw output passes through this module before anyone sees it.
The point is stated in the project's own documentation (dokumentacia.txt
§14.7: "meranie má merať kvalitu parsera, nie kompatibilitu implementácie").
The document before version 1.0 (tag exploracia-2026-10-17, §14.2) said it
as the fairness rationale for normalising the NLTK trees:

    "Bez normalizácie by sme merali kompatibilitu implementácie, nie kvalitu
     parsera."

Three concrete cases where that went wrong before this module existed:

  * Stanza emits ``(ROOT (S ...))`` and SuPar ``(TOP (S ...))``, while Benepar and
    NLTK emit a bare ``(S ...)``. The evaluation dropped only the outermost node,
    so the bare-rooted parsers silently lost one real constituent per sentence.
  * SuPar prints a literal ``(`` token as a bare bracket, which ``Tree.fromstring``
    happily misparses — no exception, just two vanished leaves and every span
    offset shifted.
  * Benepar re-segmented pre-tokenised input, returning two trees for one input
    sentence; callers kept ``trees[0]``, scoring a partial tree against a full
    reference.

None of these is a parser quality difference, and none of them raised. So the
checks here are deliberately loud: a wrapper that breaks the contract raises
``ContractViolation`` rather than producing a plausible wrong number.

The module imports only ``models``, so it can be unit-tested with fake analyzers
and without loading a single NLP model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from syntactic_analysis.models import (
    REASON_EMPTY_TREE,
    ConstituencyParse,
    DependencyParse,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from nltk import Tree


# PTB escapes for characters that are structural in bracketed notation. A literal
# "(" in a tree string is indistinguishable from an opening bracket, so it must be
# escaped before the tree is ever stringified.
BRACKET_ESCAPES: dict[str, str] = {
    "(": "-LRB-",
    ")": "-RRB-",
    "[": "-LSB-",
    "]": "-RSB-",
    "{": "-LCB-",
    "}": "-RCB-",
}

# Labels that parsers wrap a whole sentence in. These are not constituents; EVALB
# deletes TOP, and the project's own methodology says the root is excluded for
# every parser (dokumentacia.txt §7.2: the project's COLLINS is COLLINS.prm
# plus the ROOT and S1 wrappers; before version 1.0, §14.2 "Metodológia
# bracket F1", KORENE, at the tag exploracia-2026-10-17).
ROOT_WRAPPERS: tuple[str, ...] = ("TOP", "ROOT", "S1", "")


class ContractViolation(RuntimeError):
    """A wrapper returned output that breaks the pre-tokenised alignment contract.

    Attributes:
        analyzer: Name of the offending analyzer.
        method:   "analyze_dependency" or "analyze_constituency".
        reason:   Machine-readable cause. One of:
                  "sentence_count", "token_count", "token_text", "tree_count",
                  "leaf_count", "head_range".
        sentence_index: Index of the offending input sentence, when known.
    """

    def __init__(
        self,
        analyzer: str,
        method: str,
        reason: str,
        detail: str = "",
        *,
        sentence_index: int | None = None,
    ) -> None:
        self.analyzer = analyzer
        self.method = method
        self.reason = reason
        self.sentence_index = sentence_index
        where = "" if sentence_index is None else f" (sentence {sentence_index})"
        suffix = f": {detail}" if detail else ""
        super().__init__(f"[{analyzer}.{method}] {reason}{where}{suffix}")


def escape_token(word: str) -> str:
    """Replace every bracket CHARACTER in a token with its PTB escape.

    Character-wise, not whole-token: web text writes ":)", ":-)", "=)" and "(:"
    as single tokens, and there are 17 of them in the EWT test split. Escaping
    only tokens that are exactly a bracket left those unbalanced in the
    serialised tree, where one stray ")" closes a node early — a parse run died
    on exactly that after three hours. A token with no bracket character is
    returned unchanged and is not rebuilt.
    """
    if not any(character in BRACKET_ESCAPES for character in word):
        return word
    return "".join(BRACKET_ESCAPES.get(character, character) for character in word)


def escape_bracket_leaves_tree(tree: "Tree") -> "Tree":
    """Escape bracket characters in every leaf, in place, and return the tree.

    Must be called on the ``nltk.Tree`` object while the structure is still known.
    Doing it on the serialised string is impossible: at that point a literal "("
    and a structural "(" look identical — which is exactly the bug this prevents.
    """
    for position in tree.treepositions("leaves"):
        tree[position] = escape_token(tree[position])
    return tree


def collapse_to_single_line(tree_str: str) -> str:
    """Collapse a possibly multi-line bracketed tree onto one line.

    ``nltk.Tree.__str__`` pretty-prints with line breaks, so SuPar and NLTK emitted
    multi-line strings while Stanza and Benepar emitted single-line ones. The data
    model documents one tree per line.
    """
    return " ".join(tree_str.split())


def strip_unary_root_wrapper(tree: "Tree") -> "Tree":
    """Remove a unary TOP/ROOT/S1 wrapper so all parsers start from the same node.

    Only a *unary* wrapper is stripped: ``(ROOT (S ...))`` becomes ``(S ...)``, but
    a root with several children is a real constituent and is left alone.
    """
    while (
        tree.label() in ROOT_WRAPPERS
        and len(tree) == 1
        and not isinstance(tree[0], str)
    ):
        tree = tree[0]
    return tree


def _as_sentence_lists(text: Any) -> list[list[str]] | None:
    """Return the pre-tokenised sentences, or None if the input is raw text.

    The contract only applies to pre-tokenised input, where the caller supplied the
    tokens and can therefore demand them back unchanged.
    """
    if not isinstance(text, list) or not text:
        return None
    if all(isinstance(sent, list) for sent in text):
        return text
    return None


def normalize_dependency(
    parse: DependencyParse,
    text: Any,
    *,
    analyzer_name: str,
    declares_dependency: bool = True,
    strict: bool = True,
) -> DependencyParse:
    """Validate and normalise a dependency parse.

    For pre-tokenised input this enforces that the analyzer returned exactly one
    sentence per input with exactly the tokens it was given. Any mismatch used to
    be absorbed by ``zip()`` in the metrics, which silently shortened the
    denominator and *rewarded* an analyzer for emitting fewer tokens.

    Args:
        parse:               Raw output from the wrapper.
        text:                The input that produced it.
        analyzer_name:       For error messages.
        declares_dependency: Whether the analyzer lists DEPENDENCY_PARSING in
            ``supported_types``. If it does not, head/relation fields are blanked
            so a POS-only tool cannot be scored as a broken parser.
        strict:              Raise on violation. When False, violations are left
            for the caller to count.

    Returns the same object, modified in place.
    """
    if declares_dependency is False:
        for sentence in parse.sentences:
            for token in sentence.tokens:
                token.dep_rel = ""
                token.head_index = -1

    sentences = _as_sentence_lists(text)
    if sentences is None:
        return parse

    if strict and len(parse.sentences) != len(sentences):
        raise ContractViolation(
            analyzer_name,
            "analyze_dependency",
            "sentence_count",
            f"expected {len(sentences)}, got {len(parse.sentences)}",
        )

    for i, (sentence, words) in enumerate(zip(parse.sentences, sentences)):
        if strict and len(sentence.tokens) != len(words):
            raise ContractViolation(
                analyzer_name,
                "analyze_dependency",
                "token_count",
                f"expected {len(words)}, got {len(sentence.tokens)}",
                sentence_index=i,
            )
        if strict:
            got = [t.text for t in sentence.tokens]
            if got != list(words):
                first = next(
                    (k for k, (a, b) in enumerate(zip(got, words)) if a != b), 0
                )
                raise ContractViolation(
                    analyzer_name,
                    "analyze_dependency",
                    "token_text",
                    f"token {first}: expected {words[first]!r}, got {got[first]!r}",
                    sentence_index=i,
                )

        # Renumber defensively: indices are 1-based by contract, and head_index
        # values point into that numbering.
        n = len(sentence.tokens)
        for position, token in enumerate(sentence.tokens, start=1):
            token.index = position
            if token.head_index is None:
                token.head_index = -1
            if strict and not (-1 <= token.head_index <= n):
                raise ContractViolation(
                    analyzer_name,
                    "analyze_dependency",
                    "head_range",
                    f"token {position} has head {token.head_index}, "
                    f"outside [-1, {n}]",
                    sentence_index=i,
                )
    return parse


def normalize_constituency(
    parse: ConstituencyParse,
    text: Any,
    *,
    analyzer_name: str,
    strict: bool = True,
) -> ConstituencyParse:
    """Validate and normalise a constituency parse.

    Enforces one tree per input sentence, collapses multi-line trees, and checks
    that each tree covers exactly as many leaves as the sentence had tokens. The
    leaf-count check is what catches unescaped bracket tokens: ``Tree.fromstring``
    does not raise on them, it silently drops leaves and shifts every span.

    Leaf *text* is compared only to count mismatches, never to raise — parsers
    legitimately normalise quotes and dashes.

    Returns the same object, modified in place, with ``leaf_text_mismatches``
    recorded on it — a declared field of ``ConstituencyParse``, not an attribute
    attached on the way past, which is how it was set before.
    """
    parse.trees = [collapse_to_single_line(t) if t else "" for t in parse.trees]
    if len(parse.failures) != len(parse.trees):
        parse.failures = ["" if t else REASON_EMPTY_TREE for t in parse.trees]

    sentences = _as_sentence_lists(text)
    if sentences is None:
        parse.leaf_text_mismatches = 0
        return parse

    if strict and len(parse.trees) != len(sentences):
        raise ContractViolation(
            analyzer_name,
            "analyze_constituency",
            "tree_count",
            f"expected {len(sentences)}, got {len(parse.trees)}",
        )

    # The leaf-count check is not optional. It is what catches unescaped
    # bracket tokens, which `Tree.fromstring` does not raise on: it silently
    # drops leaves and shifts every span after them. There was a check_leaves
    # flag here to turn it off, which no caller ever passed.
    mismatches = 0
    from nltk import Tree  # local import: keeps this module importable bare

    for i, (tree_str, words) in enumerate(zip(parse.trees, sentences)):
        if not tree_str:
            continue  # a declared failure, not a contract breach
        try:
            tree = Tree.fromstring(tree_str)
        except (ValueError, IndexError) as exc:
            if strict:
                raise ContractViolation(
                    analyzer_name,
                    "analyze_constituency",
                    "leaf_count",
                    f"unparseable tree: {exc}",
                    sentence_index=i,
                ) from exc
            continue
        leaves = tree.leaves()
        if len(leaves) != len(words):
            if strict:
                raise ContractViolation(
                    analyzer_name,
                    "analyze_constituency",
                    "leaf_count",
                    f"expected {len(words)} leaves, got {len(leaves)} — "
                    "unescaped bracket tokens are the usual cause",
                    sentence_index=i,
                )
            continue
        expected = [escape_token(w) for w in words]
        mismatches += sum(1 for a, b in zip(leaves, expected) if a != b)

    parse.leaf_text_mismatches = mismatches
    return parse
