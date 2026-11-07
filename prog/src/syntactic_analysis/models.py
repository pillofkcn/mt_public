"""Data models — unified output containers for syntactic analysis.

These are simple data containers (dataclasses) that hold the results
of syntactic analysis. Every analyzer — regardless of the underlying tool
(spaCy, Stanza, NLTK, ...) — returns these same models. This is what makes
the framework unified: different tools produce the same output format.

How it works:
  1. Analyzers produce Token objects for each word in a sentence
  2. Tokens are grouped into Sentence objects (with original text)
  3. Sentences are wrapped in either DependencyParse or ConstituencyParse
  4. These result objects are returned by analyze_dependency() / analyze_constituency()

Key characteristics:
  - All index values are 1-based (following UD CoNLL-U convention, not 0-based)
  - head_index=0 means the token is the ROOT of the sentence
  - head_index=-1 means the head was not determined by the analyzer
  - POS tags use the Universal POS tagset (NOUN, VERB, ADJ, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Token:
    """Single token (word) with its syntactic attributes.

    Every word in a sentence becomes one Token. The Token holds all
    the syntactic information that the analyzer extracted for that word.

    Attributes:
        text:       The word as it appears in the text, e.g. "jumps"
        index:      Position in the sentence, 1-based (following CoNLL-U convention)
        pos:        Universal POS tag (language-independent), e.g. NOUN, VERB, ADJ
        xpos:       Language-specific POS tag, e.g. NN, VBZ (may be empty)
        lemma:      Base/dictionary form of the word, e.g. "jumps" -> "jump"
        dep_rel:    Dependency relation label describing how this token relates
                    to its head, e.g. "nsubj" (subject), "obj" (object), "det"
        head_index: Index of the token this word depends on.
                    0 means this token is the ROOT (no head above it).
                    -1 means the head was not determined by the analyzer.
    """

    text: str
    index: int
    pos: str = ""
    xpos: str = ""
    lemma: str = ""
    dep_rel: str = ""
    head_index: int = -1  # 0 = root, -1 = not determined

    def __repr__(self) -> str:
        """Compact representation showing index, text, and optional POS/dep info."""
        parts = [f"{self.index}:{self.text}"]
        if self.pos:
            parts.append(f"pos={self.pos}")
        if self.dep_rel:
            parts.append(f"dep={self.dep_rel}->{self.head_index}")
        return f"Token({', '.join(parts)})"


@dataclass
class Sentence:
    """One sentence — the original text plus all its annotated tokens.

    A sentence is the basic unit of analysis. Most analyzers split input
    text into sentences automatically during tokenization.

    Attributes:
        text:    The original sentence text as a single string.
        tokens:  List of Token objects with syntactic annotations.
        sent_id: Corpus identifier of the sentence when one is known (the
                 "# sent_id = ..." comment of a CoNLL-U file), else "". The EWT
                 ids encode the genre ("answers-20111108...", "email-...",
                 "newsgroup-...", "reviews-...", "weblog-..."), which is what
                 the per-genre error analysis keys on.
        doc_id:  The document the sentence belongs to, else "". In a CoNLL-U
                 file it is the last "# newdoc id = ..." before the sentence,
                 or, before any newdoc, the sent_id without its trailing
                 "-<digits>" (EWT ids are "<document>-<number>"). The document
                 is the unit the bootstrap resamples on the UD test sets
                 (plan/v1.yaml), because sentences of one document are not
                 independent of each other.
    """

    text: str
    tokens: list[Token] = field(default_factory=list)
    sent_id: str = ""
    doc_id: str = ""

    @property
    def pos_tags(self) -> list[tuple[str, str]]:
        """Extract POS tags as (word, tag) pairs.

        Returns a list of tuples, e.g. [("The", "DET"), ("cat", "NOUN"), ("sat", "VERB")]
        """
        return [(t.text, t.pos) for t in self.tokens]

    @property
    def dep_arcs(self) -> list[tuple[int, int, str]]:
        """Extract dependency arcs as (token_index, head_index, relation) triples.

        Only includes tokens that have a dependency relation set (non-empty dep_rel).
        Returns a list of tuples, e.g. [(1, 3, "det"), (2, 3, "nsubj"), (3, 0, "root")]
        """
        return [(t.index, t.head_index, t.dep_rel) for t in self.tokens if t.dep_rel]


@dataclass
class DependencyParse:
    """Result container for dependency parsing — may contain multiple sentences.

    This is the return type of analyze_dependency(). It holds a list of
    Sentence objects, each containing fully annotated tokens (POS tags,
    dependency heads, dependency relation labels).

    Attributes:
        sentences:     List of parsed Sentence objects.
        analyzer_name: Name of the analyzer that produced this result,
                       or "gold" for gold-standard data.
    """

    sentences: list[Sentence] = field(default_factory=list)
    analyzer_name: str = ""


# Why a sentence has no tree. The constituency stage writes REASON_OOV,
# REASON_NO_DERIVATION and REASON_TOO_LONG into its counts file under the names
# stage.FAILURE_CODES gives them, and any other reason as `error`.
REASON_OOV = "oov"
REASON_NO_DERIVATION = "no_derivation"
REASON_EMPTY_TREE = "empty_tree"
REASON_TOO_LONG = "too_long"


@dataclass
class ConstituencyParse:
    """Result container for constituency parsing — one tree string per sentence.

    This is the return type of analyze_constituency(). Each tree is a string
    in bracketed notation, e.g. "(S (NP (DT The) (NN cat)) (VP (VBZ sat)))".

    Attributes:
        trees:         List of bracketed tree strings, one per sentence.
                       An empty string means the parser failed on that sentence.
        analyzer_name: Name of the analyzer that produced this result.
        failures:      Parallel to trees — why each sentence failed, or "" on
                       success. Without this, an empty tree conflates three very
                       different situations: the word was outside the grammar's
                       vocabulary, no derivation existed, or our own wrapper
                       raised. The constituency stage reports a
                       failure-reason histogram so "coverage" and "quality" can
                       be told apart.
                       A wrapper writes one of REASON_OOV, REASON_NO_DERIVATION
                       and REASON_TOO_LONG here, and REASON_EMPTY_TREE stands
                       in for an empty tree that came with no reason.
        leaf_text_mismatches:
                       How many leaves came back with text that differs from the
                       token that was given, counted by `normalize_constituency`.
                       Parsers legitimately normalise quotes and dashes, so this
                       never raises — but it is the signal behind the fifth
                       measurement defect in dokumentacia.txt §14.7, where PTB-escaped
                       quotes went through a plain-text entry point and the
                       constituency ranking reversed. It used to be attached to
                       this object as an undeclared attribute and read by nobody.
    """

    trees: list[str] = field(default_factory=list)
    analyzer_name: str = ""
    failures: list[str] = field(default_factory=list)
    leaf_text_mismatches: int = 0

    def __post_init__(self) -> None:
        """Keep `failures` aligned with `trees` even for callers that omit it."""
        if not self.failures:
            self.failures = ["" if t else REASON_EMPTY_TREE for t in self.trees]
