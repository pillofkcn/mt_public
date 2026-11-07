"""NLTK — POS tagging (Averaged Perceptron) + constituency parsing (PCFG).

NLTK (Natural Language Toolkit) is the oldest and most well-known NLP library
for Python. In this framework it serves as the STATISTICAL baseline.

How it works:
  POS tagging:
    Uses an averaged perceptron tagger — a simple linear classifier that
    assigns POS tags based on word features (suffixes, prefixes, capitalization)
    and the context of surrounding words.

  Constituency parsing:
    Uses a Probabilistic Context-Free Grammar (PCFG) induced from the
    Penn Treebank corpus. Each grammar rule has a probability estimated
    via maximum likelihood, e.g. P(NP -> DT NN) = count(NP -> DT NN) / count(NP -> ...).
    ViterbiParser finds the single most probable parse tree using dynamic programming.

  Dependency parsing:
    NOT supported. analyze_dependency() returns POS-tagged tokens
    without any head or relation information.

Key characteristics:
  - NLTK outputs Penn Treebank POS tags (NN, VBZ, DT, ...) which must be
    converted to Universal POS tags (NOUN, VERB, DET, ...) for comparison
  - PCFG bracket F1 is well below the neural parsers'. On WSJ section 01 the
    confirmatory run of 17 Oct 2026 (results/evidence/confirmatory_2026-10-17)
    measured 0.73-0.81 over the sentences a PCFG row parsed, and less, by
    coverage, once a failed parse counts as 0
  - PCFG parsing has cubic time complexity — slow on longer sentences

What the PCFG measures, and what it used to measure by accident:
  On the 2077-sentence EWT test set this parser produced a tree for 5.3% of
  sentences. Reported as "constituency quality" that number is wrong: almost all
  of the loss was VOCABULARY, not grammar. The grammar's terminals were the word
  forms of the whole 3914-sentence WSJ sample it was then induced from (today
  section 00 alone, 1921 sentences, see below), so any sentence containing one
  unseen word was rejected before parsing started, as it still is. Three defects
  made it worse than it had to be (the counts are of that whole sample):

    1. The grammar was induced with start symbol S, but 456 of the training trees
       were rooted in something else (SINV 167, NP 104, FRAG 25, SBARQ 16, SQ 6...).
       Questions, inverted sentences and fragments were therefore not derivable at
       all. Fixed with a TOP start symbol and TOP -> <root> for every observed root.
    2. 6592 `-NONE-` subtrees (empty-category traces: *-1, *T*-2, 0, *U*) were
       induced into the grammar as lexical rules over terminals that can never
       occur in real input, taking MLE probability mass from usable rules.
       Fixed by strip_traces().
    3. The function-tag regex only stripped a single all-caps suffix, so NP-SBJ-1
       came out as NP-1 and never matched another parser's NP span.

  Coverage and quality are now separable two ways. `failures` records WHY each
  sentence produced no tree ("oov" vs "no_derivation"), and `pcfg_input="tags"`
  parses the POS-tag sequence against an unlexicalised copy of the same grammar,
  which removes the OOV effect entirely and leaves a quality measurement.

Which WSJ sections the PCFG is induced from:
  The NLTK sample is WSJ sections 00 (99 files, 1921 sentences) and 01 (100 files,
  1993 sentences). Section 01 is the held-out GOLD test set for the constituency
  comparison, so inducing from all 199 files would let this parser — and only this
  parser — train on its own test data. `train_sections` is therefore ("00",), the
  one value it accepts (plan/v1.yaml gives it): one grammar is induced and used on
  every constituency test set. Until 24 Oct 2026 it also accepted ("00", "01"), a
  full-sample ablation that induced the grammar from the test section as well;
  plan/v1.yaml never ran it, and it was removed. `name` does not change with the
  sections; training_corpus and grammar_info record them.

One algorithm, two training corpora — why there is an `nltk_ewt` row:
  tagger_model="ud_ewt_train" trains nltk's own PerceptronTagger on the
  UD_English-EWT train split with UPOS tags directly, instead of loading the
  shipped averaged_perceptron_tagger_eng. NLTK's documentation does not state
  what the shipped model was trained on; all it shows of its data is its
  tagset, Penn Treebank tags. Both rows tag with the same PerceptronTagger,
  whose feature templates the shipped weights are read into; the EWT row trains
  it with nltk's default of five iterations. So `nltk` vs `nltk_ewt` isolates
  the effect of the TRAINING DATA (the shipped model's unstated corpus -> EWT
  train) with the algorithm held fixed, while `nltk_ewt` against the other
  rows trained on that same EWT split isolates the effect of the ALGORITHM
  with the data held fixed.
  Without that pair the two effects are confounded in every POS comparison here:
  the pretrained tagger differs from every in-domain tagger it is ranked against
  in its algorithm AND its corpus AND its tagset at once. In this mode the
  analyzer is POS-only and the PTB->UPOS table (tagsets.PTB_TO_UPOS) is bypassed
  entirely — the tagger emits UPOS itself, so there is no conversion cost to
  charge to it.
  The perceptron is trained by scripts/train_nltk.py
  (NltkAnalyzer.train_ewt_tagger) before the lock, and only there: load_model
  reads the three files it writes and refuses a missing one, naming the
  script. Until 24 Oct 2026 load_model trained it when the files were missing.

The PTB->UPOS conversion is two tiers: the flat table (tagsets.PTB_TO_UPOS),
and the contextual rules `_refine_upos` adds on top of it. Both are applied.

What version 1.0 removed:
  The configurations plan/v1.yaml does not run went on 23 Oct 2026: the
  markovised grammar (grammar="markov", the rows nltk_markov and
  nltk_markov_tags), the PCFG parsed over the treebank's own gold PTB tags
  (pcfg_tags_source="gold" and set_gold_tags(), the *_gold rows) and the flat
  table alone (upos_rules=False, the row nltk_t1). The three options stay,
  because the plan states them, and accept the one value it gives them:
  grammar "treebank", pcfg_tags_source "nltk", upos_rules True.

Install: pip install nltk
"""

from __future__ import annotations

import hashlib
import random
import re
from functools import lru_cache, partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, NamedTuple

from syntactic_analysis.base import (
    NOT_LOADED,
    AnalysisType,
    AnnotationScheme,
    BaseAnalyzer,
    MethodLevel,
    MethodType,
    TrainedModel,
    TrainingDomain,
    TrainingSeed,
    require_trained,
)
from syntactic_analysis.data_loader import load_conllu, split_raw_text
from syntactic_analysis.downloads import DownloadedFiles, nltk_data_absent, nltk_data_files
from syntactic_analysis.offline import nltk_package
from syntactic_analysis.ptb_conventions import (
    SECTION_SENTENCES,
    section_fileids,
    section_files,
    strip_function_tags,
    strip_traces,
)
from syntactic_analysis.tagsets import PTB_TO_UPOS, UPOS_TAGS
from syntactic_analysis.models import (
    REASON_NO_DERIVATION,
    REASON_OOV,
    REASON_TOO_LONG,
    ConstituencyParse,
    DependencyParse,
    Sentence,
    Token,
)
from syntactic_analysis.normalize import BRACKET_ESCAPES
from syntactic_analysis.project import ewt_split_path, model_cache_dir

if TYPE_CHECKING:  # nltk stays a runtime-lazy import; see load_model()
    from nltk.tree import Tree


# The PTB -> UPOS table this tagger's output goes through is tagsets.PTB_TO_UPOS.
# It is the flat tier; `_refine_upos` below adds the contextual rules no flat
# table can express.

# Verb forms that UD tags AUX rather than VERB.
_BE_FORMS = frozenset({
    "be", "am", "is", "are", "was", "were", "been", "being",
    "'s", "'re", "'m", "ai",
})
# These are AUX only when they govern another verb ("have seen"), otherwise they
# are main verbs ("have a cat").
_HAVE_DO_FORMS = frozenset({
    "have", "has", "had", "having", "'ve", "'d", "do", "does", "did",
})
_NEGATORS = frozenset({"n't", "nt", "not"})
# UD splits PTB's "IN" into ADP and SCONJ. Two groups, because they need
# different treatment:
#
# Always SCONJ when tagged IN. These words can introduce a noun phrase, but the
# tagger uses a different tag when they do — demonstrative "that book" is DT, not
# IN — so an IN reading is a complementizer or subordinator.
_ALWAYS_SUBORDINATORS = frozenset({
    "that", "if", "because", "whether", "although", "though", "unless",
    "whereas", "while", "lest",
})
# Genuinely ambiguous: "before the war" (ADP) vs "before he came" (SCONJ). Decided
# by what follows — a nominal means a prepositional reading.
_AMBIGUOUS_SUBORDINATORS = frozenset({
    "as", "since", "before", "after", "until", "till", "once", "so",
})
_NOMINAL_TAGS = ("DT", "NN", "PRP", "JJ", "CD", "PDT", "WDT", "WP", "PRP$")


def _refine_upos(
    word: str,
    ptb_tag: str,
    upos: str,
    following: tuple[str, ...] = (),
) -> str:
    """Apply context rules the flat PTB->UPOS table cannot express.

    Every rule here corrects a systematic scheme difference, not a tagger error.
    Kept as a separate tier on top of the flat table; until version 1.0
    `upos_rules=False` switched it off, so that the share of the statistical
    taggers' error that is the conversion could be reported.

    The rules deliberately under-correct rather than over-correct: where the
    evidence is weak the original tag is kept, so the measured conversion cost is
    a lower bound rather than an estimate inflated by our own guessing.

    Args:
        word: Surface form, used for the lexical rules.
        ptb_tag: The tagger's Penn Treebank tag.
        upos: Result of the flat table lookup.
        following: The next few PTB tags, in order. Empty at end of sentence.
    """
    low = word.lower()
    nxt = following[0] if following else ""

    # "be" is AUX in UD in essentially every use, including as a copula.
    if ptb_tag in ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"):
        if low in _BE_FORMS:
            return "AUX"
        # "have"/"do" are AUX only when they govern another verb ("have seen",
        # "did not go"); otherwise they are main verbs ("have a cat").
        if low in _HAVE_DO_FORMS and any(
            t.startswith("VB") for t in following[:2]
        ):
            return "AUX"
        return upos

    # UD tags negation as PART, PTB as RB.
    if ptb_tag == "RB" and low in _NEGATORS:
        return "PART"

    # "to" is the infinitive marker (PART) before a verb, a preposition otherwise.
    if ptb_tag == "TO":
        return "PART" if nxt.startswith("VB") else "ADP"

    # PTB's IN covers both prepositions and subordinating conjunctions.
    if ptb_tag == "IN":
        if low in _ALWAYS_SUBORDINATORS:
            return "SCONJ"
        if low in _AMBIGUOUS_SUBORDINATORS:
            # A following verb means a clause ("before he came"), so SCONJ; a bare
            # nominal means a prepositional phrase ("before the war"), so ADP.
            # The immediate tag alone is not enough, because a clause usually
            # starts with its pronoun or determiner subject.
            if any(t.startswith("VB") or t == "MD" for t in following[:3]):
                return "SCONJ"
            if not nxt.startswith(_NOMINAL_TAGS):
                return "SCONJ"

    return upos


# --- POS: the locally trained, EWT-trained perceptron -------------------------

# Where the EWT-trained perceptron is kept. A DIRECTORY, not a file:
# PerceptronTagger.train(save_loc=...) calls save_to_json(), which writes three
# json files into the location it is given (weights, tagdict, classes), named
# after the tagger's `lang`. The matching read is PerceptronTagger.load_from_json
# (equivalently `PerceptronTagger(loc=...)`), which takes the same directory.
_PERCEPTRON_CACHE_NAME = "nltk_perceptron_ud_ewt_train"

# The script that trains it (NltkAnalyzer.train_ewt_tagger) and the transition
# parser of nltk_arc, which a missing model's error names (TrainedModel.script).
TRAINING_SCRIPT = "scripts/train_nltk.py"

# nltk's own default number of training passes. Stated rather than left implicit,
# because the point of this row is that only the DATA differs from the pretrained
# tagger — a different iteration count would make it a different configuration.
_PERCEPTRON_ITERATIONS = 5

# PerceptronTagger.train() reshuffles the sentences between iterations using the
# global `random`, so an unseeded run produces a slightly different model each
# time and a cached model could never be reproduced. Seeded (and the caller's RNG
# state restored afterwards) so the cached model is the model a retrain rebuilds.
# The provenance of a stage records it (training_seed_for, since 24 Oct 2026).
# What a retrain does not rebuild byte for byte is the order of the classes
# file: NLTK writes the set of classes in the order Python's string hashing
# gives it, different in every process unless PYTHONHASHSEED is fixed, as
# scripts/train_nltk.py fixes it since 24 Oct 2026 (dokumentacia.txt §11.4).
_PERCEPTRON_SEED = 1

# What NLTK's shipped tagger was trained on, as far as it can be stated: the
# tagger half of `training_corpus`. NLTK's documentation does not name the
# corpus, so the string says so, in the language of the results tables, and
# names the one thing the model shows of its data: it emits Penn Treebank tags.
# Until 23 Oct 2026 it read "WSJ PTB, full (averaged perceptron, pretrained)",
# a claim no NLTK document makes. The PCFG half is built from the configuration
# (`_pcfg_corpus`).
_PRETRAINED_TAGGER_CORPUS = "neuvedený v dokumentácii NLTK (značky PTB)"

# The values each closed-set option accepts. One outside its set is refused
# with the wording every wrapper gives an unknown name, "unknown <option> ...;
# expected one of [...]"; until 16 Oct 2026 these checks said "must be ...,
# got ..." instead. _TAGGER_PACKAGES holds the values of load_model's
# model_name: the one nltk data package nltk.pos_tag loads. Since version 1.0
# pcfg_tags_source, grammar and upos_rules accept one value each, the one
# plan/v1.yaml gives them (module docstring).
_TAGGER_MODELS = ("pretrained", "ud_ewt_train")
_TAGGER_PACKAGES = ("averaged_perceptron_tagger_eng",)
_PCFG_INPUTS = ("words", "tags")
_PCFG_TAGS_SOURCES = ("nltk",)
_GRAMMARS = ("treebank",)
_UPOS_RULES = (True,)


# --- constituency: grammar induction -----------------------------------------

# The start symbol used instead of "S". It is in normalize.ROOT_WRAPPERS, so a unary
# (TOP (X ...)) is stripped before span extraction and adds no spurious bracket.
_PCFG_START = "TOP"

# Marker chomsky_normal_form() puts in the labels it invents when it binarises a
# node. NLTK's default is "|", which collides with a real treebank label: PTB uses
# "X|Y" for an annotator's unresolved choice between two categories, and the sample
# contains exactly one, ADVP|PRT. un_chomsky_normal_form() then reads that label as
# one of its own artificial nodes and does `node[0], node[1]` on it — but it is
# unary, so any parse that used the rule died with IndexError from inside nltk
# rather than producing a tree. "@" appears in no treebank label, so the two
# alphabets no longer overlap. Both transforms must be given the same character.
_CNF_CHILD_CHAR = "@"

# Character chomsky_normal_form() appends for vertical (parent) annotation, e.g.
# "NP^<S>" for an NP whose parent is S. The treebank grammar adds none
# (vertMarkov=0, nltk's default), but this is nltk's own default value, named
# here rather than left implicit, so induction and un_chomsky_normal_form() are
# visibly passing the same character to each other and a future nltk default
# change cannot silently reopen the round-trip bug that the "|" collision above
# once caused for childChar.
_CNF_PARENT_CHAR = "^"

# The longest sentence, in tokens, a PCFG row attempts: the default of
# NltkAnalyzer(max_sentence_length=...), whose docstring gives the reason.
PCFG_MAX_SENTENCE_LENGTH = 40

# The WSJ sections a PCFG may be induced from: section 00 alone, the value
# plan/v1.yaml gives. Until 24 Oct 2026 any sections of the sample were
# accepted, ("00", "01") as the full-sample ablation (module docstring).
_TRAIN_SECTIONS = (("00",),)


def _validate_sections(sections) -> tuple[str, ...]:
    """Normalise and check a train_sections argument.

    Args:
        sections: An iterable of WSJ section names: ("00",), or ["00"] as
            plan/v1.yaml gives it.

    Returns the sections as a tuple.

    Raises:
        ValueError: If they are anything but section 00 alone: an empty
            iterable, a section the NLTK sample does not contain, or the
            held-out section 01, alone or beside 00. Caught at construction,
            not at induction time, so that no analyzer exists that would induce
            a grammar from the test section.
    """
    sections = tuple(sections)
    if sections not in _TRAIN_SECTIONS:
        raise ValueError(
            f"unknown train_sections {sections!r}; expected one of "
            f"{list(_TRAIN_SECTIONS)}: the PCFG is induced from WSJ section 00, "
            f"and section 01 is held out as the constituency test set"
        )
    return sections


class InducedGrammar(NamedTuple):
    """An induced PCFG plus the counts needed to report how it was built.

    Every count here is relative to the corpus files the grammar was actually
    induced from: WSJ section 00 (train_sections), not the whole sample (see
    n_fileids).

    Attributes:
        grammar:        The nltk PCFG object.
        root_labels:    Distinct root labels the start symbol can expand to, sorted.
        trees_used:     Training trees that contributed productions.
        trees_stripped: Training trees that lost at least one -NONE- subtree.
        trees_dropped:  Training trees that were nothing but traces and were skipped.
        n_productions:  Size of the raw production list handed to induce_pcfg, i.e.
            one entry per internal tree node across all training trees, counted
            BEFORE induce_pcfg merges repeated instances of the same rule into one
            probability-weighted production. This is training-data size, not
            grammar size — see n_distinct_productions.
        n_distinct_productions: len(grammar.productions()) — the number of
            distinct rules in the induced PCFG. THIS is "grammar size" in the
            sense that drives ViterbiParser's cost.
        n_fileids: How many corpus files the grammar was induced from: 99 for
            WSJ section 00, of the sample's 199, which is what shows in a log
            that the held-out section 01 was not read.
    """

    grammar: Any
    root_labels: tuple[str, ...]
    trees_used: int
    trees_stripped: int
    trees_dropped: int
    n_productions: int
    n_distinct_productions: int
    n_fileids: int


def _delexicalise(tree: "Tree") -> "Tree":
    """Replace every leaf with its preterminal's POS label, in place.

    Turns (NN dog) into (NN NN), so induce_pcfg yields a grammar whose terminals
    are POS tags rather than word forms. That is the unlexicalised grammar used by
    pcfg_input="tags": one tag per token, so leaf count — and therefore the base
    class's leaf-count contract — is preserved when the words are put back.
    """
    for position in tree.treepositions("leaves"):
        tree[position] = tree[position[:-1]].label()
    return tree


@lru_cache(maxsize=None)
def _induce_grammar(
    corpus: str,
    remove_traces: bool,
    start_symbol: str,
    terminals: str,
    grammar: str = "treebank",
    fileids: tuple[str, ...] | None = None,
) -> InducedGrammar:
    """Induce (and memoize) a PCFG from a bracketed treebank.

    Induction reads and binarises every tree in the selected files (~1900 for WSJ
    section 00, ~3900 for the whole sample) and builds tens of thousands of
    productions from them. It is deterministic given these six arguments, so
    analyzers that differ in nothing the grammar depends on share one
    induction.
    The cache is in-process only. Every stage runs as a process of its own
    (scripts/run_pipeline.py), so each stage that builds PCFG rows - POS and
    constituency - induces its grammars for itself. It is deliberately not on
    disk: a pickled grammar would go stale against edits to this module with no
    way to notice.

    `grammar` names the CNF binarisation. "treebank" is the only one since
    version 1.0 removed the markovised grammar; it stays an argument because
    it is part of what the cache key and the grammar's version name.

    The start symbol is not the raw "S" the original code used. 206 of the 1921
    trees of section 00, the default training set (456 of the whole sample's
    3914), are rooted in SINV / NP / FRAG / SBARQ / SQ / ..., so with start=S no
    question, inverted clause or fragment had a derivation — a parse failure the
    thesis was scoring as poor grammar quality. Instead every observed root label
    gets a `TOP -> <root>` production, one instance per tree, so induce_pcfg's MLE
    gives the root distribution its real shape (S is ~89% of section 00's roots)
    rather than pretending SQ and S are equally likely sentence types.

    Args:
        corpus: Name of an nltk bracketed-corpus reader, e.g. "treebank".
        remove_traces: Remove -NONE- empty categories before inducing.
        start_symbol: Label of the new unary root, normally "TOP".
        terminals: "words" for a lexicalised grammar, "tags" for one whose
            terminals are POS tags (see _delexicalise).
        grammar: "treebank" (default) for plain right-factored CNF binarisation,
            unchanged from the original behaviour; the only value.
        fileids: Restrict induction to these corpus files, e.g. only the WSJ
            section 00 files so that section 01 can be held out as gold test data
            (see NltkAnalyzer's train_sections). None (the default) uses the whole
            corpus, which is the original behaviour. A tuple rather than a list
            because it is part of the lru_cache key and must be hashable.

    Returns an InducedGrammar. Cached, so callers must treat it as read-only.

    The corpus is the NLTK data package of that name, downloaded when it is
    missing except during a run (offline.nltk_package).

    Raises:
        ValueError: If `terminals` is not "words"/"tags", or `grammar` is not
            "treebank".
        offline.NotCachedError: During a run, if the corpus is not installed.
    """
    if terminals not in _PCFG_INPUTS:
        raise ValueError(
            f"unknown terminals {terminals!r}; expected one of {sorted(_PCFG_INPUTS)}"
        )
    if grammar not in _GRAMMARS:
        raise ValueError(
            f"unknown grammar {grammar!r}; expected one of {sorted(_GRAMMARS)}"
        )

    import nltk
    import nltk.corpus
    from nltk import Nonterminal
    from nltk.grammar import Production, induce_pcfg

    nltk_package(corpus, f"corpora/{corpus}")
    reader = getattr(nltk.corpus, corpus)
    # parsed_sents() takes a list of fileids or None for "all of them"; the tuple
    # is what makes this argument cacheable, so it is converted only here.
    trees_in = (
        reader.parsed_sents() if fileids is None else reader.parsed_sents(list(fileids))
    )
    n_fileids = len(reader.fileids()) if fileids is None else len(fileids)

    start = Nonterminal(start_symbol)
    productions: list = []
    root_productions: list = []
    root_labels: set[str] = set()
    used = stripped = dropped = 0

    for raw_tree in trees_in:
        # Copy first: collapse_unary/chomsky_normal_form mutate in place, and the
        # corpus reader hands out views that can be served again from its block
        # cache — the original code was quietly editing the corpus under itself.
        tree = raw_tree.copy(deep=True)

        if remove_traces:
            n_before = len(tree.leaves())
            tree = strip_traces(tree)
            if tree is None:
                dropped += 1
                continue
            if len(tree.leaves()) != n_before:
                stripped += 1

        if terminals == "tags":
            # Before collapse_unary: collapsePOS=True merges preterminals into their
            # parents, after which a leaf's parent is no longer its POS tag.
            _delexicalise(tree)

        tree.collapse_unary(collapsePOS=True)
        # ViterbiParser needs at most binary branching. parentChar is passed
        # explicitly even though vertMarkov=0, its default, adds no "^"
        # annotation, so that induction and un_chomsky_normal_form() always
        # agree on both characters.
        tree.chomsky_normal_form(
            childChar=_CNF_CHILD_CHAR, parentChar=_CNF_PARENT_CHAR
        )

        root = tree.label()
        root_labels.add(root)
        root_productions.append(Production(start, [Nonterminal(root)]))
        productions += tree.productions()
        used += 1

    productions += root_productions
    # Named "pcfg" rather than "grammar" so it cannot shadow the `grammar: str`
    # parameter above ("treebank") — the shadowing would be harmless
    # here (the string is not read again after this point) but confusing to
    # read, since InducedGrammar.grammar is this object, not that string.
    pcfg = induce_pcfg(start, productions)
    return InducedGrammar(
        grammar=pcfg,
        root_labels=tuple(sorted(root_labels)),
        trees_used=used,
        trees_stripped=stripped,
        trees_dropped=dropped,
        n_productions=len(productions),
        n_distinct_productions=len(pcfg.productions()),
        n_fileids=n_fileids,
    )


# --- model versions ------------------------------------------------------------
#
# model_version names every model this wrapper loads the way the UDPipe and
# RDRPOSTagger wrappers name theirs, "<name>@<first twelve hex digits of a
# SHA-256>", computed when the model is loaded (since 23 Oct 2026; until then
# it was empty, so the provenance of nltk_pre, nltk_ewt, pcfg_words and
# pcfg_tags could not say which model ran):
#
#   * a perceptron tagger - the one NLTK ships in its data package, which a new
#     download can replace while the NLTK version stays the same, or the one
#     trained here on EWT train: the SHA-256 of the bytes of its three json
#     files (weights, tagdict, classes, PerceptronTagger.param_files' order)
#     one after another, which `cat` of the three files into sha256sum gives;
#   * a PCFG, which is induced at load and has no file: the SHA-256 of its
#     productions, one per line as "<production> <probability (repr)>",
#     sorted, joined by newlines, in UTF-8.


def _stream_digest(openers: Iterable[Callable[[], Any]]) -> str:
    """SHA-256 of the bytes of the binary streams the openers open, one after another."""
    digest = hashlib.sha256()
    for open_stream in openers:
        with open_stream() as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _shipped_tagger_version(package: str) -> str:
    """`<package>@<digest>` of the NLTK data package nltk.pos_tag loads.

    The files are found as PerceptronTagger.load_from_json finds them:
    nltk.data.find("taggers/averaged_perceptron_tagger_<lang>/").
    """
    import nltk.data
    from nltk.tag.perceptron import PerceptronTagger

    lang = package.removeprefix("averaged_perceptron_tagger_")
    location = nltk.data.find(f"taggers/{package}/")
    names = PerceptronTagger(load=False).param_files(lang)
    return f"{package}@{_stream_digest(location.join(name).open for name in names)[:12]}"


def _shipped_tagger_files(package: str) -> list[Path]:
    """The files of the NLTK data package nltk.pos_tag loads, as NLTK finds them.

    As PerceptronTagger.load_from_json finds them (and _shipped_tagger_version
    reads them): the three json files in nltk.data.find("taggers/<package>/"),
    or the package's zip file when NLTK finds it zipped. Nothing is
    downloaded.

    Raises:
        LookupError: If NLTK does not find the package.
    """
    import nltk.data
    from nltk.tag.perceptron import PerceptronTagger

    lang = package.removeprefix("averaged_perceptron_tagger_")
    location = nltk.data.find(f"taggers/{package}/")
    zipped = getattr(location, "zipfile", None)
    if zipped is not None:
        return [Path(zipped.filename)]
    names = PerceptronTagger(load=False).param_files(lang)
    return [Path(location.join(name).path) for name in names]


def _tagger_files_version(paths: Iterable[Path]) -> str:
    """`<directory>@<digest>` of a tagger saved as json files in one directory."""
    paths = list(paths)
    return f"{paths[0].parent.name}@{_stream_digest(partial(open, path, 'rb') for path in paths)[:12]}"


def ewt_tagger_files(cache_dir: str | Path | None = None) -> tuple[Path, ...]:
    """The three json files of the perceptron trained on EWT, in NLTK's order.

    In <cache_dir>/nltk_perceptron_ud_ewt_train/, <project_root>/model_cache
    when cache_dir is None. NLTK writes the model as three files inside the
    directory it is given (PerceptronTagger.save_to_json) - weights, tagdict
    and classes, named by param_files() after the tagger's `lang` - rather
    than as one pickle, so the model is a directory and "is it there?" means
    "are all three". The names are taken from NLTK itself, which keeps them
    correct if it renames them.
    """
    from nltk.tag.perceptron import PerceptronTagger

    directory = (
        Path(cache_dir) if cache_dir is not None else model_cache_dir()
    ) / _PERCEPTRON_CACHE_NAME
    tagger = PerceptronTagger(load=False)
    return tuple(directory / name for name in tagger.param_files(tagger.lang))


def _ewt_training_corpus(train_path: str | Path | None) -> tuple[list, str]:
    """The UPOS-tagged sentences of a CoNLL-U corpus, and how `training_corpus` names it.

    Read and checked by the training and by every load: it costs a fraction of
    a second, and it is what `training_corpus` reports.

    Raises:
        RuntimeError: If the corpus is empty, or carries a tag outside the 17
            UPOS (a PTB/UD mix-up, which would score as near-zero rather than
            as an error).
    """
    path = Path(train_path) if train_path is not None else ewt_split_path("train")
    train = load_conllu(path)
    # UPOS straight out of the treebank: in this mode no PTB tag is ever
    # produced, so PTB_TO_UPOS and _refine_upos play no part and the
    # conversion tier is not on this row's scorecard at all.
    tagged = [
        [(tok.text, tok.pos) for tok in sent.tokens] for sent in train.sentences
    ]
    if not tagged:
        raise RuntimeError(f"training corpus {path} loaded zero sentences")

    unexpected = {t for sent in tagged for _w, t in sent} - UPOS_TAGS
    if unexpected:
        raise RuntimeError(
            f"training data contains non-UPOS tags: {sorted(unexpected)}"
        )

    # The counts and the release named the way every other in-domain row
    # names its own: "current split" is the file on disk, which no UD
    # release tag identifies (data/DATASET.md). The trailing "averaged
    # perceptron trained locally" this cell carried until 16 Oct 2026 was
    # architecture, not corpus.
    n_tok = sum(len(sent) for sent in tagged)
    corpus = (
        f"UD_English-EWT train (current split), {len(tagged)} sent / "
        f"{n_tok} tok (UPOS)"
    )
    return tagged, corpus


def _grammar_version(
    info: InducedGrammar, grammar: str, terminals: str, sections: tuple[str, ...],
    max_sentence_length: int | None,
) -> str:
    """`PCFG <grammar> over <terminals> from WSJ <sections>@<digest> (...)` of an induced PCFG."""
    from nltk.grammar import Production

    lines = sorted(
        f"{Production.__str__(production)} {production.prob()!r}"
        for production in info.grammar.productions()
    )
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    cap = (
        f"max_sentence_length {max_sentence_length}"
        if max_sentence_length is not None else "no length cap"
    )
    return (
        f"PCFG {grammar} over {terminals} from WSJ {'+'.join(sections)}@{digest[:12]} "
        f"({info.n_distinct_productions} productions, {cap})"
    )


# --- constituency: output normalisation --------------------------------------

# A node label immediately after an opening bracket. Terminals are never preceded
# by "(", so this cannot match a word.
_LABEL_AT_OPEN_RE = re.compile(r"\(([^\s()]+)")

class NltkAnalyzer(BaseAnalyzer):

    def __init__(
        self,
        *,
        key: str | None = None,
        upos_rules: bool = True,
        pcfg_input: str = "words",
        pcfg_strip_traces: bool = True,
        grammar: str = "treebank",
        max_sentence_length: int | None = PCFG_MAX_SENTENCE_LENGTH,
        train_sections: tuple[str, ...] = ("00",),
        tagger_model: str = "pretrained",
        pcfg_tags_source: str = "nltk",
    ) -> None:
        """Create the analyzer.

        Args:
            key: Key for this instance, overriding the one `name`
                computes. The computed key names the SYSTEM — input mode, tagger
                corpus. plan/v1.yaml gives every row its key.
            upos_rules: Apply the contextual PTB->UPOS rules (_refine_upos) on top
                of the flat table. True is the only value: the flat table alone
                (False, the row nltk_t1) went with version 1.0.
            pcfg_input: What the PCFG parses. "words" (default) is the real system:
                the grammar's terminals are WSJ word forms, so an unseen word means
                no parse, and the resulting success rate is a vocabulary-coverage
                measurement. "tags" parses this analyzer's own perceptron tag
                sequence against an unlexicalised copy of the same grammar and then
                puts the words back as leaves; that removes the OOV effect and
                leaves a structural-quality measurement. Both rows are worth
                reporting, which is why the default is not changed.
            pcfg_strip_traces: Drop -NONE- empty categories from the training trees
                before inducing. On by default; switchable to quantify what the
                traces were costing.
            max_sentence_length: Skip sentences longer than this rather than
                parsing them, recording them as "too_long". ViterbiParser is cubic
                in sentence length and linear in grammar size, so a single long
                sentence can cost more than a whole run: an 81-token sentence took
                812 seconds and ~5 GB. 40 tokens covers the large majority of the
                EWT test set; None disables the cap. Skipped sentences are counted
                separately from parse failures, because declining to attempt a
                sentence is not the same as being unable to analyse it.
            grammar: Which CNF binarisation the PCFG is induced with. "treebank"
                (default), the unmarkovised right-factored grammar, is the only
                value: the markovised grammar went with version 1.0.
            train_sections: Which WSJ sections of the NLTK Penn Treebank sample the
                PCFG is induced from: ("00",), the only value (plan/v1.yaml gives
                ["00"]), because section 01 is the held-out gold test set for
                constituency parsing — inducing from it would train this parser
                on its own test data. The full-sample ablation ("00", "01") was
                removed on 24 Oct 2026. Does not change `name`; provenance
                records the section.
            tagger_model: Which POS model this analyzer tags with. "pretrained"
                (default) is NLTK's shipped averaged_perceptron_tagger_eng, whose
                training corpus NLTK's documentation does not state; it emits
                PTB tags that this wrapper converts to UPOS.
                "ud_ewt_train" trains the same perceptron on the EWT train split
                with UPOS tags directly, which makes this analyzer POS-only (there
                are no PTB tags left for the PCFG) and turns the `nltk`/`nltk_ewt`
                pair into a controlled comparison of training domains — see the
                module docstring. In that mode every grammar option must keep
                its default.
            pcfg_tags_source: Where the tags parsed in pcfg_input="tags" mode come
                from. "nltk" (default) uses this analyzer's own perceptron tags, so
                the result is the grammar's quality behind a real tagger. It is
                the only value: the treebank's own gold PTB tags ("gold") went
                with version 1.0.

        Raises:
            ValueError: If any option is outside its allowed values, if
                `train_sections` is not section 00 alone, or if the combination
                is meaningless: "ud_ewt_train" with pcfg_input="tags" (that
                tagger emits no PTB tags to parse over), or with a
                pcfg_strip_traces or max_sentence_length other than the default
                (that mode reads neither).
        """
        for option, value, allowed in (
            ("pcfg_input", pcfg_input, _PCFG_INPUTS),
            ("grammar", grammar, _GRAMMARS),
            ("tagger_model", tagger_model, _TAGGER_MODELS),
            ("pcfg_tags_source", pcfg_tags_source, _PCFG_TAGS_SOURCES),
            ("upos_rules", upos_rules, _UPOS_RULES),
        ):
            if value not in allowed:
                raise ValueError(
                    f"unknown {option} {value!r}; expected one of {sorted(allowed)}"
                )
        train_sections = _validate_sections(train_sections)
        # Refused rather than quietly ignored: each of these combinations would
        # produce a plausible-looking analyzer that measures something other than
        # what its configuration says.
        if tagger_model == "ud_ewt_train":
            # The EWT-trained tagger builds no grammar and converts no PTB tag,
            # so these options would be stored and never read, while `name` and
            # `display_name` said "nltk_ewt" / "NLTK (EWT)" whatever they held.
            # Until 16 Oct 2026 they were accepted.
            # (train_sections has one value since 24 Oct 2026, so it cannot
            # differ from its default here.)
            defaults = (
                ("pcfg_strip_traces", pcfg_strip_traces, True),
                ("max_sentence_length", max_sentence_length, PCFG_MAX_SENTENCE_LENGTH),
            )
            ignored = [option for option, value, default in defaults if value != default]
            if ignored:
                raise ValueError(
                    f"tagger_model='ud_ewt_train' is a UPOS tagger with no grammar "
                    f"and no PTB->UPOS conversion; {', '.join(ignored)} would be "
                    f"ignored in that mode"
                )
        if tagger_model == "ud_ewt_train" and pcfg_input == "tags":
            raise ValueError(
                "tagger_model='ud_ewt_train' emits UPOS, not the PTB tags the "
                "unlexicalised grammar is built over; pcfg_input='tags' is "
                "unavailable in that mode"
            )
        self._key = key
        self._max_sentence_length = max_sentence_length
        self._pcfg_input = pcfg_input
        self._pcfg_strip_traces = pcfg_strip_traces
        self._grammar = grammar
        self._train_sections = train_sections
        self._tagger_model = tagger_model
        self._tagger = None  # averaged perceptron POS tagger
        self._parser = None  # ViterbiParser with induced PCFG grammar
        self._grammar_info: InducedGrammar | None = None  # induction counts, for logs
        # Counted from the corpus by load_model() in "ud_ewt_train" mode; unused
        # by the pretrained tagger, whose corpus is a fixed fact rather than
        # something this run measured.
        self._training_corpus = NOT_LOADED
        # Set by load_model(); see model_version.
        self._tagger_version = ""
        self._grammar_version = ""

    @property
    def name(self) -> str:
        # Tag mode is a different system, not a setting: it answers a different
        # question and must occupy its own key and CSV column, so the two input
        # modes have distinct keys, nltk and nltk_tags. train_sections
        # deliberately does NOT appear here: it configures one system rather
        # than defining another, and training_corpus records it. An explicitly
        # supplied key wins over all of it.
        if self._key is not None:
            return self._key
        if self._tagger_model == "ud_ewt_train":
            # A separate tool as far as the results tables are concerned: same
            # algorithm, different training corpus, and POS-only.
            return "nltk_ewt"
        key = "nltk"
        if self._pcfg_input == "tags":
            key += "_tags"
        return key

    @property
    def grammar_info(self) -> InducedGrammar | None:
        """Counts describing the induced PCFG, or None before load_model().

        Also None in tagger_model="ud_ewt_train" mode, which never builds one.
        """
        return self._grammar_info

    @property
    def train_sections(self) -> tuple[str, ...]:
        """WSJ sections of the treebank sample the PCFG is induced from."""
        return self._train_sections

    @property
    def supported_types(self) -> set[AnalysisType]:
        if self._tagger_model == "ud_ewt_train":
            # The EWT-trained perceptron emits UPOS, so there are no PTB
            # preterminals for the WSJ grammar to stand on: this configuration is
            # a tagger and nothing else.
            return {AnalysisType.POS_TAGGING}
        return {AnalysisType.POS_TAGGING, AnalysisType.CONSTITUENCY_PARSING}

    @property
    def method_type(self) -> MethodType:
        return MethodType.STATISTICAL

    @property
    def method_detail(self) -> str:
        if self._tagger_model == "ud_ewt_train":
            # The tool-wide string, which the analyzer's repr prints: it says
            # that this is the perceptron of the `nltk` row trained on another
            # corpus. A stage's provenance records `method_detail_for`, the
            # same "Averaged Perceptron" for both rows, and leaves the corpus
            # to `training_corpus`.
            return "Averaged Perceptron (trained on EWT)"
        if self._pcfg_input == "tags":
            return "Averaged Perceptron / PCFG, unlexicalised, parsed over tags"
        return "Averaged Perceptron / PCFG"

    def method_detail_for(self, analysis_type: AnalysisType) -> str:
        """The architecture of one task: the perceptron for POS, the grammar for
        constituency.

        The rule of `training_corpus_for`, for the same reason: the tool-wide
        `method_detail` names both components, so until 16 Oct 2026 a POS row
        read "/ PCFG", and a constituency row named the perceptron while its
        Trained on cell, rightly, left the perceptron's corpus out. A row that
        parses over this tagger's own tags keeps both, since its trees depend
        on the perceptron's tags as much as on the grammar.
        """
        if (
            analysis_type is not AnalysisType.CONSTITUENCY_PARSING
            or self._tagger_model == "ud_ewt_train"
        ):
            return "Averaged Perceptron"
        if self._pcfg_input == "tags":
            return self.method_detail
        return "PCFG"

    @property
    def display_name(self) -> str:
        if self._tagger_model == "ud_ewt_train":
            return "NLTK (EWT)"
        if self._pcfg_input == "tags":
            return "NLTK (tags)"
        return "NLTK"

    @property
    def annotation_scheme(self) -> dict:
        if self._tagger_model == "ud_ewt_train":
            # Trained on UPOS directly, so the output IS Universal — no conversion
            # table stands between this tagger and the gold standard.
            return {AnalysisType.POS_TAGGING: AnnotationScheme.UD}
        # PTB on both axes: pos_tag emits Penn Treebank tags that this wrapper
        # converts to UPOS, and the PCFG is induced from PTB trees.
        return {
            AnalysisType.POS_TAGGING: AnnotationScheme.PTB,
            AnalysisType.CONSTITUENCY_PARSING: AnnotationScheme.PTB,
        }

    @property
    def method_level(self) -> dict:
        # Averaged perceptron over discrete feature templates, and a PCFG whose
        # parameters are relative-frequency counts: L1 on both tasks, whichever
        # grammar variant, input mode or training corpus is selected. Retraining
        # the perceptron on EWT changes the data, not the amount of the model that
        # is learned, so the level is the same — that is the whole point of the
        # nltk / nltk_ewt pair.
        if self._tagger_model == "ud_ewt_train":
            return {AnalysisType.POS_TAGGING: MethodLevel.STATISTICAL}
        return {
            AnalysisType.POS_TAGGING: MethodLevel.STATISTICAL,
            AnalysisType.CONSTITUENCY_PARSING: MethodLevel.STATISTICAL,
        }

    @property
    def training_domain(self) -> dict:
        if self._tagger_model == "ud_ewt_train":
            # Trained on the EWT test set's own training split: in-domain, like
            # the other rows trained on that split, which is what makes them
            # comparable.
            return {AnalysisType.POS_TAGGING: TrainingDomain.IN}
        # Out-of-domain for the EWT test: the PCFG is induced from WSJ newswire,
        # and the shipped tagger was not trained on this project's EWT train
        # split (NLTK's documentation does not state its corpus at all). (The
        # WSJ-01 gold run treats every constituency parser as in-domain and says
        # so itself.)
        return {
            AnalysisType.POS_TAGGING: TrainingDomain.OUT,
            AnalysisType.CONSTITUENCY_PARSING: TrainingDomain.OUT,
        }

    def _pcfg_corpus(self) -> str:
        """The PCFG half of `training_corpus`: sections, size, grammar options."""
        label = "section" if len(self._train_sections) == 1 else "sections"
        n_sent = sum(SECTION_SENTENCES[s] for s in self._train_sections)
        pcfg = (
            f"nltk treebank sample, {label} {', '.join(self._train_sections)}, "
            f"{n_sent} sent (PCFG"
        )
        if self._pcfg_strip_traces:
            pcfg += ", traces stripped"
        if self._pcfg_input == "tags":
            pcfg += ", unlexicalised"
        return pcfg + ")"

    @property
    def training_corpus(self) -> str:
        if self._tagger_model == "ud_ewt_train":
            return self._training_corpus
        return f"{_PRETRAINED_TAGGER_CORPUS} / {self._pcfg_corpus()}"

    def training_corpus_for(self, analysis_type: AnalysisType) -> str:
        """The corpus of one task: the tagger's for POS, the grammar's for
        constituency.

        The tool-wide `training_corpus` names both, so a per-task cell built
        from it would carry the other task's corpus as well. One case keeps
        both on purpose: a row that parses over this tagger's own tags
        (pcfg_input="tags") owes its trees to the tagger's corpus as much as to
        the grammar's.
        """
        if self._tagger_model == "ud_ewt_train":
            return self._training_corpus
        if analysis_type is not AnalysisType.CONSTITUENCY_PARSING:
            return _PRETRAINED_TAGGER_CORPUS
        if self._pcfg_input == "tags":
            return f"{_PRETRAINED_TAGGER_CORPUS} / {self._pcfg_corpus()}"
        return self._pcfg_corpus()

    @property
    def model_version(self) -> str:
        """Every model load_model() loaded, by name and digest; "" before it.

        The tagger: averaged_perceptron_tagger_eng@<digest> (the shipped one)
        or nltk_perceptron_ud_ewt_train@<digest> (the one trained on EWT, from
        its model_cache files). In the pretrained mode, which also induces a
        PCFG, the grammar after it: "PCFG treebank over words from WSJ
        00@<digest> (<n> productions, max_sentence_length 40)". The digests are
        described above `_stream_digest`. `model_version_for` names the model
        of one task.
        """
        return "; ".join(part for part in (self._tagger_version, self._grammar_version) if part)

    def model_version_for(self, analysis_type: AnalysisType) -> str:
        """The model one task runs: the tagger for POS, the grammar for
        constituency.

        A grammar parsed over this analyzer's own tags (pcfg_input="tags")
        owes its trees to the tagger as well, so its version names both, as
        `training_corpus_for` names both corpora.
        """
        if analysis_type is not AnalysisType.CONSTITUENCY_PARSING or not self._grammar_version:
            return self._tagger_version
        if self._pcfg_input == "tags":
            return f"{self._grammar_version}; tags {self._tagger_version}"
        return self._grammar_version

    def training_seed_for(self, analysis_type: AnalysisType) -> TrainingSeed | None:
        """_PERCEPTRON_SEED for the perceptron trained on EWT; None for anything else.

        In tagger_model="ud_ewt_train" mode `train_ewt_tagger` trains the tagger
        with Python's global random generator seeded with _PERCEPTRON_SEED,
        which PerceptronTagger.train() reshuffles the training sentences with
        after every iteration; nothing else in that training is random. The
        value is read from the constant. The shipped tagger is a published
        model and the PCFG is induced at load, so neither states a seed.
        None before load_model().
        """
        if (self._tagger_model != "ud_ewt_train" or self._tagger is None
                or analysis_type is not AnalysisType.POS_TAGGING):
            return None
        return TrainingSeed(
            _PERCEPTRON_SEED,
            "random.seed(_PERCEPTRON_SEED) in wrappers/nltk.py before nltk's "
            "PerceptronTagger.train(), which reshuffles the training sentences with "
            "Python's global generator after every iteration; nothing else in the "
            "training is random",
        )

    def trained_model(
        self,
        model_name: str = "averaged_perceptron_tagger_eng",
        *,
        cache_dir: str | Path | None = None,
        **load,
    ) -> TrainedModel | None:
        """The perceptron trained on EWT, which load_model reads; None in any other mode.

        Takes load_model's keywords; only cache_dir says where the model is.
        In tagger_model="ud_ewt_train" mode: the three json files of
        `ewt_tagger_files`, which scripts/train_nltk.py writes. The shipped
        tagger is a published model and the PCFG is induced at load from the
        treebank sample, so in the other modes this project trains no file
        and the answer is None. Since 24 Oct 2026.
        """
        if self._tagger_model != "ud_ewt_train":
            return None
        return TrainedModel(TRAINING_SCRIPT, files=ewt_tagger_files(cache_dir))

    def downloaded_files(
        self, model_name: str = "averaged_perceptron_tagger_eng", **load,
    ) -> tuple[DownloadedFiles, ...]:
        """The NLTK data load_model reads: the shipped tagger and the treebank sample.

        Takes load_model's keywords. In every mode but tagger_model=
        "ud_ewt_train" load_model reads the tagger NLTK ships (its three json
        files, _shipped_tagger_files) and induces the PCFG from the files of
        the treebank sample's train_sections (ptb_conventions.section_files:
        the 99 files of WSJ 00) - nltk_pre too, which is scored on POS only.
        In "ud_ewt_train" mode it reads the perceptron trained here
        (trained_model) and the EWT train split, which is data, not a model,
        and nothing downloaded: (). Each file is declared under the entry of
        nltk.data.path NLTK finds it in. A package NLTK does not find is
        declared where NLTK would look first (downloads.nltk_data_absent), so
        a lock check reports it missing: the tagger by its three files, the
        treebank by its directory. Nothing is downloaded or loaded. Since
        24 Oct 2026.

        Raises:
            ValueError: If model_name is not a package nltk.pos_tag loads.
        """
        if self._tagger_model == "ud_ewt_train":
            return ()
        if model_name not in _TAGGER_PACKAGES:
            raise ValueError(
                f"unknown model_name {model_name!r}; expected one of {sorted(_TAGGER_PACKAGES)}"
            )
        found: list[Path] = []
        absent_files: list[str] = []
        absent_directories: list[str] = []
        try:
            found += _shipped_tagger_files(model_name)
        except LookupError:
            from nltk.tag.perceptron import PerceptronTagger

            lang = model_name.removeprefix("averaged_perceptron_tagger_")
            absent_files += [
                f"taggers/{model_name}/{name}" for name in PerceptronTagger(load=False).param_files(lang)
            ]
        try:
            found += section_files(self._train_sections)
        except LookupError:
            absent_directories.append("corpora/treebank/combined")
        declared = nltk_data_files(found)
        if absent_files or absent_directories:
            declared += (nltk_data_absent(absent_files, absent_directories),)
        return declared

    @classmethod
    def train_ewt_tagger(
        cls,
        *,
        train_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> tuple[Path, ...]:
        """Train the averaged perceptron on EWT with UPOS, and save it: the model of nltk_ewt.

        The one place the model of tagger_model="ud_ewt_train" is trained.
        scripts/train_nltk.py calls it before the lock; load_model never does
        (since 24 Oct 2026). Until then load_model trained the model when its
        files were missing, with this same code.

        Args:
            train_path: CoNLL-U corpus to train on. None uses the EWT train split.
            cache_dir:  Parent of the model's directory. None uses
                        <project_root>/model_cache.

        Returns the three files written (`ewt_tagger_files`).

        Raises:
            RuntimeError: If the corpus is empty or carries a tag outside the
                17 UPOS (_ewt_training_corpus).
        """
        from nltk.tag.perceptron import PerceptronTagger

        tagged, _corpus = _ewt_training_corpus(train_path)
        files = ewt_tagger_files(cache_dir)
        directory = files[0].parent
        tagger = PerceptronTagger(load=False)
        # train() reshuffles the training sentences between iterations from the
        # global RNG, so an unseeded run produces a different model every time and
        # the cached model could never be rebuilt. The caller's RNG state is put
        # back afterwards: seeding here must not reach into anything else.
        state = random.getstate()
        random.seed(_PERCEPTRON_SEED)
        try:
            # save_to_json() calls mkdir(), not makedirs(), so the parent has
            # to exist before it runs.
            directory.mkdir(parents=True, exist_ok=True)
            tagger.train(tagged, save_loc=str(directory), nr_iter=_PERCEPTRON_ITERATIONS)
        finally:
            random.setstate(state)
        return files

    def load_model(
        self,
        model_name: str = "averaged_perceptron_tagger_eng",
        *,
        train_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        **kwargs,
    ) -> None:
        """Load the POS tagger and, for constituency, build a PCFG grammar from Penn Treebank.

        Args:
            model_name: Name of the NLTK POS tagger data package to download.
                        "averaged_perceptron_tagger_eng", the default, is the
                        only one: nltk.pos_tag loads that package whatever is
                        named here, so any other name is refused rather than
                        downloaded and ignored. Unused when
                        tagger_model="ud_ewt_train", which reads its own.
            train_path: The CoNLL-U corpus the EWT perceptron was trained on,
                        counted for `training_corpus` in
                        tagger_model="ud_ewt_train" mode; None is the EWT train
                        split. Ignored otherwise.
            cache_dir:  Override the directory the EWT-trained perceptron is read
                        from. Defaults to <project_root>/model_cache — an ABSOLUTE
                        path (project.model_cache_dir), so the cache is the same
                        directory no matter where a script is launched from.

        This method performs two setup steps:
          1. Downloads the POS tagger model (averaged perceptron weights), or
             reads the one scripts/train_nltk.py trained on the EWT train split
             when tagger_model="ud_ewt_train"
          2. Downloads Penn Treebank and induces a PCFG grammar from WSJ
             section 00 (train_sections) for constituency parsing
        The NLTK data packages are downloaded only when they are missing and
        never during a run, which refuses instead (offline.nltk_package). The
        perceptron trained on EWT is never trained here: a missing file is
        refused, naming scripts/train_nltk.py (since 24 Oct 2026; until then
        this method trained it, and `allow_cache=False` trained it in memory
        for the out-of-repository test suite).

        Step 2 is memoized process-wide by _induce_grammar(); calling load_model()
        again, or on a second NltkAnalyzer with the same PCFG settings, is cheap.
        It is skipped entirely in tagger_model="ud_ewt_train" mode, which supports
        POS tagging only. The tokenizer models raw text needs (punkt) are
        downloaded by data_loader.split_raw_text, as for every wrapper that
        uses it, not here.

        Raises:
            TypeError:  If an unknown keyword argument is passed.
            ValueError: If `model_name` is not "averaged_perceptron_tagger_eng".
            base.ModelNotTrainedError: In tagger_model="ud_ewt_train" mode, if
                          a file of the perceptron is missing.
            RuntimeError: In tagger_model="ud_ewt_train" mode, as
                          _load_ewt_tagger raises it.
            offline.NotCachedError: During a run, if the tagger package or
                          the treebank sample is not installed.
        """
        self._reject_unknown_options(kwargs)
        if model_name not in _TAGGER_PACKAGES:
            raise ValueError(
                f"unknown model_name {model_name!r}; "
                f"expected one of {sorted(_TAGGER_PACKAGES)}"
            )
        import nltk
        from nltk.parse import ViterbiParser

        if self._tagger_model == "ud_ewt_train":
            self._tagger = self._load_ewt_tagger(train_path=train_path, cache_dir=cache_dir)
            return

        nltk_package(model_name, f"taggers/{model_name}")
        self._tagger = nltk.pos_tag
        self._tagger_version = _shipped_tagger_version(model_name)

        self._grammar_info = _induce_grammar(
            "treebank",
            self._pcfg_strip_traces,
            _PCFG_START,
            self._pcfg_input,
            self._grammar,
            section_fileids(self._train_sections),
        )
        self._grammar_version = _grammar_version(
            self._grammar_info, self._grammar, self._pcfg_input, self._train_sections,
            self._max_sentence_length,
        )
        # ViterbiParser finds the single most probable parse via dynamic programming
        self._parser = ViterbiParser(self._grammar_info.grammar)

    def _load_ewt_tagger(
        self,
        *,
        train_path: str | Path | None,
        cache_dir: str | Path | None,
    ) -> Any:
        """Return the tag() of the averaged perceptron scripts/train_nltk.py trained on EWT.

        The files are checked first: a missing one is refused, naming the
        script, and nothing is trained (require_trained). The training corpus
        is read and checked on every load even though the model comes from
        its files: it costs a fraction of a second, it is what
        `training_corpus` reports, and a corpus that is not UPOS is refused
        here as in the training.

        Args:
            train_path: The CoNLL-U corpus the model was trained on. None uses
                        the EWT train split.
            cache_dir:  Parent of the model's directory. None uses <project_root>/model_cache.

        Returns the bound `tag` method, so it is call-compatible with nltk.pos_tag
        and the rest of this wrapper needs no second code path.

        Raises:
            base.ModelNotTrainedError: If a file of the model is missing.
            RuntimeError: If the corpus is empty, if it carries a tag outside the
                17 UPOS, or if the model's files hold tags outside them.
        """
        from nltk.tag.perceptron import PerceptronTagger

        trained = self.trained_model(cache_dir=cache_dir)
        require_trained(self.name, trained)
        _tagged, self._training_corpus = _ewt_training_corpus(train_path)

        directory = trained.files[0].parent
        tagger = PerceptronTagger(load=False)
        # load_from_json() accepts a Path directly (nltk >= 3.9); the
        # equivalent PerceptronTagger(loc=directory) constructor does the same
        # thing, but this way the untrained instance above is reused.
        tagger.load_from_json(lang=tagger.lang, loc=directory)
        stale = set(tagger.classes) - UPOS_TAGS
        if stale:
            raise RuntimeError(
                f"cached tagger in {directory} was trained on non-UPOS tags "
                f"{sorted(stale)}; train it again with {TRAINING_SCRIPT} --force"
            )
        self._tagger_version = _tagger_files_version(trained.files)
        return tagger.tag

    @property
    def max_sentence_length(self) -> int | None:
        """Above this many symbols a sentence is declined, not attempted.

        Readable from outside because which sentences were declined must be
        reconstructible: it is a function of this cap and the sample,
        not something the trees can say. Exact Viterbi inference is cubic, so the
        cap is a statement about our compute budget and never about the grammar.
        """
        return self._max_sentence_length

    def _analyze_dependency(self, text: str | list[list[str]]) -> DependencyParse:
        """Run POS tagging only — returns tokens with POS tags but NO dependency info.

        Args:
            text: Either a raw string (NLTK tokenizes automatically) or
                  a list of pre-tokenized sentences (words already split).

                  Example raw:          "The cat sat."
                  Example pretokenized: [["The", "cat", "sat", "."]]

        Returns a DependencyParse where each token has a POS tag (Universal)
        and xpos tag (Penn Treebank), but head_index and dep_rel are not set.
        NLTK does not support dependency parsing.

        In tagger_model="ud_ewt_train" mode there is no Penn Treebank tag to carry:
        the tagger was trained on UPOS, so its output is already the Universal tag
        and xpos stays empty.
        """
        self._ensure_loaded()
        sentences = []

        if isinstance(text, list):
            word_lists = text  # already pre-tokenized by caller
        else:
            # Tokenize raw text: first into sentences, then into words
            word_lists = split_raw_text(text)

        for words in word_lists:
            # (word, tag) pairs — PTB tags from the pretrained tagger, UPOS from
            # the EWT-trained one; both are called exactly the same way.
            tagged = self._tagger(words)

            if self._tagger_model == "ud_ewt_train":
                # The tag IS the UPOS. Running it through PTB_TO_UPOS as well
                # would look up an already-converted tag and fall through to "X"
                # for every token.
                sentences.append(
                    Sentence(
                        text=" ".join(words),
                        tokens=[
                            Token(
                                text=word,
                                index=i + 1,
                                pos=tag,
                                xpos="",  # no language-specific tagset here
                                lemma=word,  # no lemmatizer; use surface form
                            )
                            for i, (word, tag) in enumerate(tagged)
                        ],
                    )
                )
                continue

            tags = [tag for _word, tag in tagged]
            tokens = []
            for i, (word, tag) in enumerate(tagged):
                upos = PTB_TO_UPOS.get(tag, "X")
                upos = _refine_upos(word, tag, upos, following=tuple(tags[i + 1 : i + 4]))
                tokens.append(
                    Token(
                        text=word,
                        index=i + 1,  # 1-based indexing (CoNLL convention)
                        pos=upos,
                        xpos=tag,  # keep original Penn Treebank tag
                        lemma=word,  # NLTK has no lemmatizer here; use surface form
                    )
                )
            sentences.append(Sentence(text=" ".join(words), tokens=tokens))
        return DependencyParse(sentences=sentences, analyzer_name=self.name)

    def _pcfg_terminals(self, words: list[str]) -> list[str]:
        """Map one sentence's tokens to the symbols the grammar expects.

        In "words" mode the terminals are the tokens themselves. In "tags" mode
        they are this analyzer's own perceptron tags, with the tagger's literal
        "("/")" rewritten to the treebank's -LRB-/-RRB-. Exactly one symbol per
        token either way, so the words can be substituted back without
        disturbing the leaf-count contract.

        Args:
            words: Tokens of a single sentence.
        """
        if self._pcfg_input == "words":
            return list(words)
        # nltk.pos_tag emits the literal "(" / ")" for parentheses, but the treebank
        # spells them -LRB-/-RRB-. Without this bridge every parenthesis is an OOV
        # *tag* in pcfg_input="tags" mode — i.e. the lexical failure this mode
        # exists to remove. The table is normalize.BRACKET_ESCAPES: the PTB spelling
        # of a bracket is one convention, for a token and for its tag.
        return [
            BRACKET_ESCAPES.get(tag, tag)
            for _word, tag in self._tagger(list(words))
        ]

    def _analyze_constituency(self, text: str | list[list[str]]) -> ConstituencyParse:
        """Run PCFG constituency parsing using ViterbiParser.

        Args:
            text: Either a raw string or a list of pre-tokenized sentences.

                  Example raw:          "The cat sat."
                  Example pretokenized: [["The", "cat", "sat", "."]]

        Returns a ConstituencyParse with one bracketed tree string per sentence,
        and a parallel `failures` list naming the reason for each empty tree. The
        reasons are not interchangeable and the original code conflated them:

          "oov"           the grammar has no lexical rule for some input symbol, so
                          parsing never starts. A VOCABULARY result — it says
                          nothing about the grammar's structural quality.
          "no_derivation" every symbol was covered but no analysis spanned the
                          sentence. A GRAMMAR result.
          "too_long"      the sentence is past `max_sentence_length`. A COST
                          decision of this wrapper: nothing was attempted. The
                          constituency stage records it as `over_length`
                          (stage.FAILURE_CODES).

        Those are the only three ways a sentence ends without a tree. Anything
        else — a bug in this wrapper, a corrupt grammar — propagates, because
        scoring our own defects as parse failures is how a 5.3% success rate went
        unexplained for so long.

        Limitations:
          - "words" mode only parses sentences whose every word occurs in WSJ
            section 00, which the grammar was induced from (1921 sentences);
            that is the dominant failure mode
          - Bracket F1 well below the neural parsers' (figures in the module
            docstring)
          - Cubic time complexity — slow on longer sentences

        Raises:
            NotImplementedError: In tagger_model="ud_ewt_train" mode, which is a
                POS tagger only — no grammar is built for it.
            RuntimeError: If the PCFG parser was not loaded.
        """
        if self._tagger_model == "ud_ewt_train":
            raise NotImplementedError(
                "tagger_model='ud_ewt_train' supports POS tagging only — it emits "
                "UPOS, and the PCFG is built over Penn Treebank tags."
            )
        self._ensure_loaded()
        if self._parser is None:
            raise RuntimeError("PCFG parser not loaded.")

        if isinstance(text, list):
            word_lists = text  # already pre-tokenized by caller
        else:
            # Tokenize raw text: first into sentences, then into words
            word_lists = split_raw_text(text)

        grammar = self._parser.grammar()
        trees: list[str] = []
        failures: list[str] = []

        for i, words in enumerate(word_lists):
            symbols = self._pcfg_terminals(list(words))

            # ViterbiParser is O(n^3 * |productions|) and the grammar has tens of
            # thousands of productions, so cost explodes on long sentences: a
            # single 81-token sentence took 812 seconds and about 5 GB of RSS,
            # which is more than the rest of a 30-sentence run put together. The
            # effect is worst in "tags" mode, where nothing is filtered out by OOV
            # first and every sentence reaches the chart.
            #
            # Skipped sentences are recorded as their own failure reason rather
            # than folded into "no_derivation": a sentence we declined to attempt
            # is not a sentence the grammar could not analyse, and reporting them
            # together would overstate the grammar's weakness.
            if self._max_sentence_length and len(symbols) > self._max_sentence_length:
                trees.append("")
                failures.append(REASON_TOO_LONG)
                continue

            # Checked up front rather than inferred from the exception ViterbiParser
            # raises internally, so an OOV word is distinguishable from every other
            # ValueError that could come out of the parser.
            try:
                grammar.check_coverage(symbols)
            except ValueError:
                trees.append("")
                failures.append(REASON_OOV)
                continue

            try:
                parse = next(self._parser.parse(symbols))
            except StopIteration:
                trees.append("")
                failures.append(REASON_NO_DERIVATION)
                continue

            # parentChar matters only for a markovised grammar (vertMarkov>0), but
            # is passed unconditionally: it is nltk's own default, harmless
            # against a treebank parse that never added "^" annotation, and
            # passing it explicitly means both transforms always agree.
            parse.un_chomsky_normal_form(
                childChar=_CNF_CHILD_CHAR, parentChar=_CNF_PARENT_CHAR
            )
            self._set_word_leaves(parse, list(words))
            trees.append(self._normalize_tree(parse))
            failures.append("")

        return ConstituencyParse(
            trees=trees, analyzer_name=self.name, failures=failures
        )

    @staticmethod
    def _set_word_leaves(tree: "Tree", words: list[str]) -> "Tree":
        """Write the sentence's own tokens onto the tree's leaves, in place.

        Two jobs, one place. In "tags" mode the parse came back with POS tags as
        leaves and this is what relexicalises it; the tag sequence has exactly one
        entry per token and the parse spans all of it, so leaf i belongs to token i
        and the base class's leaf-count contract holds by construction. In "words"
        mode the leaves already are the tokens and only the escaping matters.

        Bracket tokens are escaped to their PTB forms either way: a literal "("
        written into the serialised tree would be read back as structure, and
        Tree.fromstring() does not raise on that — it silently drops leaves and
        shifts every span, producing a plausible-looking wrong score.

        Args:
            tree: The parse, still holding whatever symbols were parsed.
            words: The original tokens, in order.

        Raises:
            ValueError: If the leaf count and the token count disagree, which would
                mean the parsed symbols and the tokens were not parallel.
        """
        from syntactic_analysis.normalize import escape_token

        positions = tree.treepositions("leaves")
        if len(positions) != len(words):
            raise ValueError(
                f"parse has {len(positions)} leaves for {len(words)} tokens"
            )
        for position, word in zip(positions, words):
            tree[position] = escape_token(word)
        return tree

    @staticmethod
    def _normalize_tree(tree) -> str:
        """Normalize a PCFG parse tree for bracket F1 evaluation.

        Two normalizations (both standard in EVALB evaluation):
          1. Strip probability suffix — ViterbiParser appends "(p=...)" to
             the string representation of ProbabilisticTree, which makes the
             tree unparseable by Tree.fromstring() in the scorer.
          2. Strip function tags — PCFG grammar retains Penn Treebank function
             tags (NP-SBJ, PP-DIR, ADVP-DIR) that neural parsers do not produce.
             EVALB strips these before comparison. Without stripping, labeled
             bracket spans like ("NP-SBJ", 0, 2) would never match ("NP", 0, 2).

        The earlier pattern for (2) was r"\\(([A-Z]+)-[A-Z]+" and only removed ONE
        all-caps group, so it left "NP-SBJ-1" as "NP-1", "PP-LOC-PRD" as "PP-PRD",
        and did not touch "VP-2" or "NP=2" at all. Every surviving suffix is a span
        label that cannot match any other parser's — a silent precision loss on the
        constituents the parser got right. See strip_function_tags for the cases.

        Both are output normalization (same category as PTB→UPOS tag mapping
        and SD→UD dependency label mapping) — the parser's structural output
        is unchanged, only annotation-scheme artifacts are removed.
        """
        tree_str = str(tree)
        # 1. Remove trailing probability annotation "(p=1.23e-20)"
        tree_str = re.sub(r"\s*\(p=[^)]+\)\s*$", "", tree_str)
        # 2. Remove function tags. Only labels are rewritten: terminals are never
        #    preceded by "(", so a word is never mistaken for a node label.
        return _LABEL_AT_OPEN_RE.sub(
            lambda m: "(" + strip_function_tags(m.group(1)), tree_str
        )

    def _ensure_loaded(self) -> None:
        """Verify that load_model() has been called before analysis.

        Raises:
            RuntimeError: If the tagger has not been loaded.
        """
        self._require_loaded(self._tagger is not None)
