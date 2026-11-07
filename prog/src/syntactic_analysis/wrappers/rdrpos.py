"""RDRPOSTagger — POS tagging by a Single Classification Ripple-Down Rules cascade.

RDRPOSTagger (Nguyen, Nguyen, Pham & Pham) tags by running a token through a
tree of IF-THEN rules. The tree is a Single Classification Ripple-Down Rules
(SCRDR) structure: every node carries a condition over a fixed 13-slot context
window (the two preceding words and their tags, the token and its current tag,
the two following words and their tags, and three suffixes), plus a conclusion.
A node that fires passes control to its EXCEPT child — a more specific rule that
may override it — and a node that does not fire passes control to its ELSE
sibling. The deepest node that fired supplies the final tag, so later, more
specific rules "ripple down" over earlier, more general ones.

How it works here:
  1. An initial tag is assigned from a lexicon (.DICT): exact form, then
     lowercased form, then a digit test, then a 5/4/3/2-character suffix entry,
     then capitalisation, then a global unknown-word default.
  2. That initial tagging is rippled through the rule tree (.RDR).
  3. If only the always-true root fires, the initial lexicon tag stands.

Where its rules come from — this matters for the thesis' RULE_BASED row
----------------------------------------------------------------------
The rules are INDUCED FROM A CORPUS, not written by hand. The SCRDR learner
tags the training corpus with the lexicon, compares against the gold tags, and
grows an exception rule for each remaining error class until a frequency
threshold is hit. So this tagger belongs in the same family as Brill's
transformation-based learning: the MECHANISM at run time is a deterministic,
inspectable, non-probabilistic rule cascade, but the CONTENT of those rules is
learned from annotated data. It is not a hand-written grammar, and it must not
be read as one. Until version 1.0 `RegexpUposAnalyzer` (wrappers/regexp.py,
removed) was the hand-written member of this row and this one the
corpus-induced member.

Key characteristics:
  - Supports ONLY POS tagging — no dependency or constituency parsing
  - Deterministic: no probabilities, no beam, no randomness
  - Native UPOS — the model is trained on the UPOS column of UD_English-EWT
    train, which IS the target tagset, so nothing is converted
  - Trained on the EWT train split, the same treebank the gold test split comes
    from — the in-domain advantage UDPipe and Stanza share, which is why
    `training_corpus` is reported beside every score

Install and licensing — read before citing
------------------------------------------
RDRPOSTagger is NOT on PyPI and its GitHub repository carries no `setup.py` or
`pyproject.toml`, so `pip install <github archive>` fails outright with
"does not appear to be a Python project". There is nothing to install.

Its `License.txt` states GPL-3.0-or-later (GitHub reports NOASSERTION only
because the file paraphrases the licence instead of reproducing its text). This
project is MIT, so upstream's GPL sources are neither vendored here nor
downloaded and imported at run time. Instead, the SCRDR *inference* — tree
construction from the .RDR text format, the 13-slot context window, and the
ripple-down traversal, roughly 120 lines in total — is implemented directly in
this module from the published algorithm. Its output was checked against
upstream's own tagger token for token: on the full EWT test split with the
authors' published model (below), and on the EWT dev split with the model
trained here (its training record).

The model
---------
  "en_ewt_train"  (TRAINED_MODEL) the model scripts/train_rdrpos.py trains on
                  this project's EWT train file (data/en_ewt-ud-train.conllu)
                  with the authors' own learner, at a pinned commit of their
                  repository, and with a fixed hash seed, which makes its rule
                  tree the same on every run (the script's docstring says why
                  it is needed). The learner is GPL as well: the script
                  downloads it outside the repository and runs it there as a
                  separate program, and this module reads the .RDR and .DICT
                  files it writes. They lie in model_cache/rdrpos_ewt/ beside
                  the training record, and load_model refuses them unless both
                  are the files the record names. The script also records that
                  this module's reader tags the dev split exactly as the
                  learner's own tagger does.

Nothing is downloaded: load_model reads what the learner wrote, and only that.
Until 24 Oct 2026 this module also loaded the authors' published model,
"en_ewt_upos" (trained by them on UD_English-EWT r2.4 train; its two files
were downloaded from a pinned commit of their repository), which was the
default of load_model. plan/v1.yaml never used it; it was removed on that
date, and its download with it.

`key` (constructor) names the row, "rdrpos" unless given.

Cite: Nguyen, Nguyen, Pham & Pham (2016), "A Robust Transformation-Based
Learning Approach Using Ripple Down Rules for Part-of-Speech Tagging", AI
Communications 29(3); and the same authors (2014), "RDRPOSTagger: A Ripple Down
Rules-based Part-Of-Speech Tagger", EACL 2014 demonstrations. NOTICE credits
the software itself, GPL-3.0-or-later, by its own years (2013-2019).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from syntactic_analysis.base import (
    NOT_LOADED,
    AnalysisType,
    AnnotationScheme,
    BaseAnalyzer,
    MethodLevel,
    MethodType,
    TrainedModel,
    TrainingDomain,
    require_trained,
)
from syntactic_analysis.checksum import sha256_of
from syntactic_analysis.data_loader import split_raw_text
from syntactic_analysis.models import (
    ConstituencyParse,
    DependencyParse,
    Sentence,
    Token,
)
from syntactic_analysis.project import model_cache_dir

__all__ = [
    "RdrPosAnalyzer",
    "model_files_available",
    "TRAINED_MODEL",
    "build_tree",
    "read_lexicon",
    "initial_tag",
]

# The model scripts/train_rdrpos.py trains on the EWT train file (see the
# module docstring), and where it lies: the training file the learner is given,
# the two files it writes beside it (it names them after it), and the record.
# Under model_cache/, which git ignores. The only model of this wrapper.
TRAINED_MODEL = "en_ewt_train"
EWT_WORK_DIR = model_cache_dir() / "rdrpos_ewt"
EWT_TRAIN_FILE = "en_ewt-ud-train.UPOS"
EWT_RULES = EWT_WORK_DIR / f"{EWT_TRAIN_FILE}.RDR"
EWT_LEXICON = EWT_WORK_DIR / f"{EWT_TRAIN_FILE}.DICT"
EWT_TRAINING_RECORD = EWT_WORK_DIR / "training_record.json"

# The training corpus of the model, as `training_corpus` reports it.
_TRAINED_CORPUS = (
    "UD_English-EWT train (current split, retrained locally), UPOS rules induced by SCRDR"
)

# The 13 context slots, in the order the .RDR file's condition keys use them.
# Mapping them by name keeps the rule parser readable; the traversal itself
# works on plain indices because it runs once per token per tree level.
_SLOT_INDEX: dict[str, int] = {
    "prevWord2": 0,
    "prevTag2": 1,
    "prevWord1": 2,
    "prevTag1": 3,
    "word": 4,
    "tag": 5,
    "nextWord1": 6,
    "nextTag1": 7,
    "nextWord2": 8,
    "nextTag2": 9,
    "suffixL2": 10,
    "suffixL3": 11,
    "suffixL4": 12,
}

# Placeholders for "this slot falls outside the sentence" / "this token is too
# short to have this suffix". They are ordinary values that a rule condition can
# match against, which is why they are concrete strings and not None.
_NO_WORD = "<W>"
_NO_TAG = "<T>"
_NO_SUFFIX = "<SFX>"

# A context window with every slot at its out-of-sentence placeholder. The
# window is built by copying this and overwriting what is in range.
_EMPTY_CONTEXT: tuple[str, ...] = (
    _NO_WORD, _NO_TAG, _NO_WORD, _NO_TAG, _NO_WORD, _NO_TAG,
    _NO_WORD, _NO_TAG, _NO_WORD, _NO_TAG,
    _NO_SUFFIX, _NO_SUFFIX, _NO_SUFFIX,
)

# The three lexicon keys that are not word forms but fallbacks.
_UNKNOWN_WORD_KEY = "TAG4UNKN-WORD"
_UNKNOWN_CAPITAL_KEY = "TAG4UNKN-CAPITAL"
_UNKNOWN_NUM_KEY = "TAG4UNKN-NUM"

# Upstream rewrites these three characters to a doubled apostrophe before
# looking them up, because its training corpora normalise quotes that way. The
# substitution is replicated so the lexicon hits the same entries; the ORIGINAL
# token text is still what this wrapper emits, because the output contract
# requires pre-tokenised input back verbatim.
_QUOTE_FORMS = ("“", "”", '"')
_QUOTE_LEXICON_KEYS = ("''", ".", ",")

_HAS_DIGIT = re.compile(r"[0-9]")


@dataclass
class _RdrNode:
    """One node of the SCRDR tree.

    Attributes:
        condition:  The 13-slot context pattern this node tests. Slots the rule
                    does not constrain are None.
        check_slots: Indices of the non-None slots, precomputed. The traversal
                    is the hot loop — one comparison per constrained slot rather
                    than 13 per node is the difference between seconds and
                    minutes over a full treebank.
        conclusion: The tag this node assigns when it fires.
        depth:      Exception depth. 0 is the always-true root.
        except_child: More specific rule tried when this node FIRES.
        else_child:   Alternative rule tried when this node does NOT fire.
        father:      Parent node, needed only while building the tree.
    """

    condition: tuple[str | None, ...]
    check_slots: tuple[int, ...]
    conclusion: str
    depth: int
    except_child: "_RdrNode | None" = None
    else_child: "_RdrNode | None" = None
    father: "_RdrNode | None" = field(default=None, repr=False)


def _concrete_value(fragment: str) -> str:
    """Extract the literal value from one condition or conclusion fragment.

    A fragment looks like ``object.prevWord1 == "the"`` or
    ``object.conclusion = "VERB"``. An EMPTY string literal is not a literal at
    all — it is upstream's spelling of the out-of-sentence placeholder, and which
    placeholder it means depends on the slot's kind.

    Args:
        fragment: One ``object.<slot> == "<value>"`` (or ``= "<value>"``) piece.

    Returns the literal string, or the matching placeholder.
    """
    if fragment.find('""') > 0:
        if fragment.find("Word") > 0:
            return _NO_WORD
        if fragment.find("suffixL") > 0:
            return _NO_SUFFIX
        return _NO_TAG
    # Everything between the first quote and the closing quote at the very end.
    return fragment[fragment.find('"') + 1: len(fragment) - 1]


def _parse_condition(text: str) -> tuple[tuple[str | None, ...], tuple[int, ...]]:
    """Parse a rule's condition into a 13-slot pattern plus its constrained slots.

    Args:
        text: The left-hand side of a rule line, e.g.
              ``object.prevTag1 == "DET" and object.word == "Gulf"``.

    Returns ``(pattern, check_slots)``, where `pattern` has None in every slot
    the rule leaves free.
    """
    slots: list[str | None] = [None] * 13
    for rule in text.split(" and "):
        rule = rule.strip()
        # "object.prevTag1 == \"DET\"" -> key is between the "." and the space.
        key = rule[rule.find(".") + 1: rule.find(" ")]
        index = _SLOT_INDEX.get(key)
        if index is not None:
            slots[index] = _concrete_value(rule)
    return tuple(slots), tuple(i for i, v in enumerate(slots) if v is not None)


def build_tree(rdr_text: str) -> _RdrNode:
    """Build the SCRDR tree from the text of a .RDR model file.

    The file format is one rule per line, ``<condition> : <conclusion>``, with
    exception depth encoded as leading TAB characters. A line one level deeper
    than the previous one is that rule's EXCEPT child; a line at the same depth
    is its ELSE sibling; a shallower line is the ELSE sibling of the ancestor at
    that depth.

    Args:
        rdr_text: Full contents of the .RDR file.

    Returns the root node, whose condition is unconstrained and therefore always
    fires. Its conclusion is never used: a token whose deepest firing node is the
    root keeps its lexicon tag instead (see `tag_tokens`).

    Raises:
        ValueError: If the file contains no usable rules, which means the wrong
            file was cached or the download was truncated past its checksum.
    """
    root = _RdrNode(condition=(None,) * 13, check_slots=(), conclusion="", depth=0)
    current = root
    current_depth = 0
    n_rules = 0

    lines = rdr_text.splitlines()
    # Line 0 is the always-true root rule; the root node above already stands
    # in for it, so parsing starts at line 1.
    for line in lines[1:]:
        depth = 0
        for char in line:
            if char == "\t":
                depth += 1
            else:
                break

        line = line.strip()
        if not line:
            continue
        if line.startswith("cc"):
            continue  # a "cornerstone case" annotation, not a rule

        head, _, tail = line.partition(" : ")
        condition, check_slots = _parse_condition(head.strip())
        node = _RdrNode(
            condition=condition,
            check_slots=check_slots,
            conclusion=_concrete_value(tail.strip()),
            depth=depth,
        )

        if depth > current_depth:
            current.except_child = node
        elif depth == current_depth:
            current.else_child = node
        else:
            while current.depth != depth:
                current = current.father  # type: ignore[assignment]
            current.else_child = node

        node.father = current
        current = node
        current_depth = depth
        n_rules += 1

    if n_rules == 0:
        raise ValueError(
            "the .RDR model file contained no rules — the cached file is not a "
            "Ripple-Down Rules model"
        )
    return root


def read_lexicon(dict_text: str) -> dict[str, str]:
    """Parse a .DICT lexicon into a form/suffix -> tag mapping.

    The file is whitespace-separated ``<key> <tag>`` lines. Keys are word forms,
    suffix patterns spelled ``.*ing``, and the three TAG4UNKN-* fallbacks.

    Args:
        dict_text: Full contents of the .DICT file.

    Returns the mapping. Malformed short lines are skipped rather than raising:
        a lexicon is data, and one bad line should not take down the tagger.
    """
    lexicon: dict[str, str] = {}
    for line in dict_text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            lexicon[parts[0]] = parts[1]
    return lexicon


def initial_tag(word: str, lexicon: dict[str, str]) -> tuple[str, str]:
    """Assign the pre-ripple tag to one token from the lexicon alone.

    The lookup order is upstream's, and it is itself a small rule cascade:
    exact form, lowercased form, "contains a digit", then progressively shorter
    suffix entries (5, 4, 3, 2 characters), then capitalisation, then the global
    unknown-word default.

    Args:
        word: The token as it appears in the text.
        lexicon: Mapping from `read_lexicon`.

    Returns ``(lookup_form, tag)``. The lookup form differs from `word` only for
    the three quote characters upstream normalises to ``''``; the CONTEXT window
    must use that normalised form so the rules see what they were learned on,
    while the emitted token keeps its original text.
    """
    if word in _QUOTE_FORMS:
        for key in _QUOTE_LEXICON_KEYS:
            if key in lexicon:
                return "''", lexicon[key]
        return "''", lexicon.get(_UNKNOWN_WORD_KEY, "X")

    if not word:  # defensive: an empty token has no shape to test
        return word, lexicon.get(_UNKNOWN_WORD_KEY, "X")

    if word in lexicon:
        return word, lexicon[word]
    lowered = word.lower()
    if lowered in lexicon:
        return word, lexicon[lowered]

    if _HAS_DIGIT.search(word) is not None:
        return word, lexicon.get(_UNKNOWN_NUM_KEY, "X")

    # Suffix entries are stored with a ".*" prefix. The length guards are
    # upstream's: a 2-character suffix is only consulted for words of 4
    # characters or more, so short words are not tagged by their whole selves.
    length = len(word)
    if length >= 6 and ".*" + word[-5:] in lexicon:
        return word, lexicon[".*" + word[-5:]]
    if length >= 5 and ".*" + word[-4:] in lexicon:
        return word, lexicon[".*" + word[-4:]]
    if length >= 4:
        if ".*" + word[-3:] in lexicon:
            return word, lexicon[".*" + word[-3:]]
        if ".*" + word[-2:] in lexicon:
            return word, lexicon[".*" + word[-2:]]

    if word[0].isupper():
        return word, lexicon.get(_UNKNOWN_CAPITAL_KEY, "X")
    return word, lexicon.get(_UNKNOWN_WORD_KEY, "X")


def _context_window(forms: list[str], tags: list[str], i: int) -> tuple[str, ...]:
    """Build the 13-slot context window for token `i`.

    Args:
        forms: Lookup forms of the whole sentence (see `initial_tag`).
        tags:  Initial tags of the whole sentence, parallel to `forms`.
        i:     Index of the token being tagged.

    Returns the window as a tuple, with placeholders where the window runs off
    either end of the sentence or the token is too short for a suffix slot.
    """
    context = list(_EMPTY_CONTEXT)
    word = forms[i]
    context[4] = word
    context[5] = tags[i]

    # Suffix slots. The thresholds are upstream's and are deliberately one
    # character longer than the suffix: a 4-letter word gets its 2- and
    # 3-character suffixes but not its 4-character one, which would be almost
    # the whole word.
    if len(word) >= 4:
        context[10] = word[-2:]
        context[11] = word[-3:]
    if len(word) >= 5:
        context[12] = word[-4:]

    if i > 0:
        context[2] = forms[i - 1]
        context[3] = tags[i - 1]
    if i > 1:
        context[0] = forms[i - 2]
        context[1] = tags[i - 2]
    last = len(forms) - 1
    if i < last:
        context[6] = forms[i + 1]
        context[7] = tags[i + 1]
    if i < last - 1:
        context[8] = forms[i + 2]
        context[9] = tags[i + 2]
    return tuple(context)


def _find_fired_node(root: _RdrNode, context: tuple[str, ...]) -> _RdrNode:
    """Walk the SCRDR tree and return the deepest node that fired.

    A node that fires hands control to its EXCEPT child (a more specific rule
    that may override it); a node that does not fire hands control to its ELSE
    sibling. That is the whole ripple-down mechanism.

    Args:
        root:    Tree root from `build_tree`.
        context: 13-slot window from `_context_window`.

    Returns the last node whose condition was satisfied. The root's condition is
    unconstrained, so this never returns None.
    """
    current: _RdrNode | None = root
    fired = root
    while current is not None:
        condition = current.condition
        satisfied = True
        for slot in current.check_slots:
            if condition[slot] != context[slot]:
                satisfied = False
                break
        if satisfied:
            fired = current
            current = current.except_child
        else:
            current = current.else_child
    return fired


def model_files_available(
    model_name: str = TRAINED_MODEL, cache_dir: str | Path | None = None
) -> bool:
    """Report whether the model's two files and its training record exist.

    Exists so the out-of-repository test suite can skip cleanly on a machine
    where scripts/train_rdrpos.py has not run, and so callers can tell whether
    load_model() will find the model before they call it.

    Args:
        model_name: TRAINED_MODEL; any other name is no model of this wrapper.
        cache_dir:  Override for the cache directory. Defaults to
                    <project_root>/model_cache; the files are in its
                    rdrpos_ewt folder.

    Returns True only if the rules, the lexicon and the training record are
    all present.
    """
    if model_name != TRAINED_MODEL:
        return False
    return all(path.is_file() for path in _trained_paths(cache_dir).values())


# The script that writes the model, which a missing model's error names
# (TrainedModel.script).
TRAINING_SCRIPT = "scripts/train_rdrpos.py"


def _trained_paths(cache_dir: str | Path | None = None) -> dict[str, Path]:
    """The rules, the lexicon and the record of TRAINED_MODEL.

    In model_cache/rdrpos_ewt, or in the folder of that name under `cache_dir`.
    """
    directory = (
        Path(cache_dir) / EWT_WORK_DIR.name if cache_dir is not None else EWT_WORK_DIR
    )
    return {
        "rules": directory / EWT_RULES.name,
        "lexicon": directory / EWT_LEXICON.name,
        "record": directory / EWT_TRAINING_RECORD.name,
    }


class RdrPosAnalyzer(BaseAnalyzer):

    def __init__(self, *, key: str | None = None) -> None:
        """Create the analyzer.

        Args:
            key: Key and CSV column name; "rdrpos" unless given. The
                 model is chosen by load_model's `model_name`.
        """
        self._key = key or "rdrpos"
        self._root = None      # SCRDR tree root (_RdrNode)
        self._lexicon = None   # form/suffix -> tag mapping
        self._model_name = ""  # TRAINED_MODEL once load_model() has run
        self._model_version = ""  # set by load_model(); see model_version

    @property
    def name(self) -> str:
        return self._key

    @property
    def supported_types(self) -> set[AnalysisType]:
        return {AnalysisType.POS_TAGGING}  # POS tagging only

    @property
    def method_type(self) -> MethodType:
        # RULE_BASED describes the run-time mechanism: a deterministic cascade
        # of explicit IF-THEN rules with no probabilities anywhere. The rules
        # themselves were induced from a corpus — see the module docstring, and
        # `training_corpus` below, which is what keeps that visible in results
        # tables.
        return MethodType.RULE_BASED

    @property
    def method_detail(self) -> str:
        return "SCRDR (Ripple-Down Rules, corpus-induced)"

    @property
    def display_name(self) -> str:
        return "RDRPOSTagger"

    @property
    def annotation_scheme(self) -> dict:
        # The model's tags are the treebank's own UPOS column. No conversion.
        return {AnalysisType.POS_TAGGING: AnnotationScheme.UD}

    @property
    def method_level(self) -> dict:
        # L0: the SCRDR induction chooses rules, not weights; at run time the
        # cascade is a deterministic IF-THEN tree with no numeric parameters.
        return {AnalysisType.POS_TAGGING: MethodLevel.RULE}

    @property
    def training_domain(self) -> dict:
        # The rules were induced from UD_English-EWT train (the current split),
        # i.e. from the test treebank's own training split — in-domain.
        return {AnalysisType.POS_TAGGING: TrainingDomain.IN}

    @property
    def training_corpus(self) -> str:
        return _TRAINED_CORPUS if self._model_name == TRAINED_MODEL else NOT_LOADED

    @property
    def model_version(self) -> str:
        """The two loaded model files, each preceded by its folder and followed
        by the first twelve hex digits of its SHA-256:
        `rdrpos_ewt/<rules>@<digest>, rdrpos_ewt/<lexicon>@<digest>`.

        In the form the UDPipe wrapper names its model in: the file, and the
        digest of the file actually loaded, computed at load. Until 16 Oct 2026
        this wrapper had no version, and a provenance record named the row by
        its training corpus. Empty until load_model() has run.
        """
        return self._model_version

    def trained_model(
        self,
        model_name: str = TRAINED_MODEL,
        *,
        cache_dir: str | Path | None = None,
        **load,
    ) -> TrainedModel:
        """The rules and the lexicon scripts/train_rdrpos.py writes, which load_model reads.

        Takes load_model's keywords; model_name and cache_dir say where the
        model is. The model is the two files the tagger runs on, the .RDR rule
        tree and the .DICT lexicon; the training record beside them, which
        load_model checks them against, describes the training and is not a
        file of the model. Never None: the one model of this wrapper is
        trained here. Since 24 Oct 2026.

        Raises:
            ValueError: If model_name is unknown.
        """
        if model_name != TRAINED_MODEL:
            raise ValueError(
                f"unknown model_name {model_name!r}; expected one of {[TRAINED_MODEL]}"
            )
        paths = _trained_paths(cache_dir)
        return TrainedModel(TRAINING_SCRIPT, files=(paths["rules"], paths["lexicon"]))

    def load_model(
        self,
        model_name: str = TRAINED_MODEL,
        *,
        cache_dir: str | Path | None = None,
        **kwargs,
    ) -> None:
        """Load the .RDR rule tree and the .DICT lexicon scripts/train_rdrpos.py wrote.

        Nothing is downloaded (module docstring). Until 24 Oct 2026 the default
        was the authors' published model, "en_ewt_upos", which this method
        downloaded when it was missing; it was removed then, because
        plan/v1.yaml does not use it, and with it the `allow_download` option.

        Args:
            model_name: "en_ewt_train" (TRAINED_MODEL), the model
                        scripts/train_rdrpos.py trains, the only one; see the
                        module docstring.
            cache_dir:  Override for the model cache. Defaults to
                        <project_root>/model_cache — an ABSOLUTE path
                        (project.model_cache_dir), so the cache is the same file
                        no matter what the process's working directory is.
                        The model is read from its rdrpos_ewt folder.

        Raises:
            TypeError:    If an unknown keyword argument is passed.
            ValueError:   If model_name is unknown.
            base.ModelNotTrainedError: If a file of the model is missing
                          (trained_model, require_trained).
            RuntimeError: If the training record is missing, or a file is
                          not the one the record names.
        """
        self._reject_unknown_options(kwargs)
        require_trained(self.name, self.trained_model(model_name, cache_dir=cache_dir))
        self._load_trained(cache_dir)

    def _load_trained(self, cache_dir: str | Path | None) -> None:
        """Load TRAINED_MODEL, the files scripts/train_rdrpos.py wrote and recorded.

        Named by its files, each preceded by its folder and followed by the
        first twelve hex digits of its SHA-256:
        `rdrpos_ewt/en_ewt-ud-train.UPOS.RDR@<12 hex>, rdrpos_ewt/...DICT@<12 hex>`.

        Raises:
            RuntimeError: If a file or the record is missing, or a file is not
                the one the record names.
        """
        paths = _trained_paths(cache_dir)
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise RuntimeError(
                f"RDRPOSTagger model {TRAINED_MODEL!r} is not trained: missing "
                f"{missing}. Train it with scripts/train_rdrpos.py."
            )
        record = json.loads(paths["record"].read_text(encoding="utf-8"))
        digests = {}
        for role in ("rules", "lexicon"):
            digests[role] = sha256_of(paths[role])
            recorded = record.get("files", {}).get(role, {}).get("sha256", "")
            if digests[role] != recorded:
                raise RuntimeError(
                    f"{paths[role]} is not the file {paths['record'].name} names "
                    f"(sha256 {digests[role][:12]}, recorded {recorded[:12] or 'none'}). "
                    f"Train it again with scripts/train_rdrpos.py --force."
                )
        # Both files are UTF-8 and may contain non-ASCII bytes, so the encoding
        # is pinned explicitly rather than left to the platform default (which
        # is what upstream's own loaders use). On a Windows cp1250 locale a
        # .DICT raises UnicodeDecodeError outright, while a .RDR decodes but
        # silently mojibakes its non-ASCII rule literals — the quieter and
        # therefore worse of the two failures.
        self._root = build_tree(paths["rules"].read_text(encoding="utf-8"))
        self._lexicon = read_lexicon(paths["lexicon"].read_text(encoding="utf-8"))
        if not self._lexicon:
            raise RuntimeError(f"the lexicon {paths['lexicon']} parsed to zero entries")
        self._model_name = TRAINED_MODEL
        self._model_version = ", ".join(
            f"{paths[role].parent.name}/{paths[role].name}@{digests[role][:12]}"
            for role in ("rules", "lexicon")
        )

    def tag_tokens(self, words: list[str]) -> list[str]:
        """Tag one pre-tokenized sentence, returning one UPOS tag per word.

        Args:
            words: The sentence's tokens, in order.

        Returns a list of UPOS tags, exactly as long as `words`.

        The sentence is tagged in two passes because the rule conditions read
        the NEIGHBOURS' initial tags: every token must have its lexicon tag
        before any token is rippled, or a rule's `prevTag1` would see an already
        corrected tag and fire differently.

        Upstream joins the tokens into a string and re-splits on whitespace;
        this works on the list directly, which is identical for whitespace-free
        tokens and strictly safer for the alignment contract if a token ever
        contains a space.
        """
        self._ensure_loaded()
        if not words:
            return []

        forms: list[str] = []
        tags: list[str] = []
        for word in words:
            form, tag = initial_tag(word, self._lexicon)
            forms.append(form)
            tags.append(tag)

        out: list[str] = []
        for i in range(len(words)):
            node = _find_fired_node(self._root, _context_window(forms, tags, i))
            # depth 0 is the always-true root: no real rule matched, so the
            # lexicon's initial tag stands.
            out.append(node.conclusion if node.depth > 0 else tags[i])
        return out

    def _analyze_dependency(self, text: str | list[list[str]]) -> DependencyParse:
        """Run POS tagging only — no dependency information produced.

        Args:
            text: Either a raw string or a list of pre-tokenized sentences.

                  Example raw:          "The cat sat."
                  Example pretokenized: [["The", "cat", "sat", "."]]

        Returns a DependencyParse whose tokens carry UPOS tags; head_index and
        dep_rel are left unset, and the base class blanks them anyway because
        DEPENDENCY_PARSING is not in supported_types.
        """
        self._ensure_loaded()

        if isinstance(text, list):
            word_lists = text  # already pre-tokenized; returned verbatim
        else:
            # RDRPOSTagger ships no tokenizer of its own.
            word_lists = split_raw_text(text)

        sentences = []
        for words in word_lists:
            words = list(words)
            tags = self.tag_tokens(words)
            tokens = [
                Token(
                    text=word,   # the ORIGINAL token, never the lexicon lookup form
                    index=i + 1,
                    pos=tag,
                    xpos="",     # the UPOS model produces no language-specific tag
                    lemma=word,  # RDRPOSTagger has no lemmatizer
                )
                for i, (word, tag) in enumerate(zip(words, tags))
            ]
            sentences.append(Sentence(text=" ".join(words), tokens=tokens))
        return DependencyParse(sentences=sentences, analyzer_name=self.name)

    def _analyze_constituency(self, text: str | list[list[str]]) -> ConstituencyParse:
        """Not supported — RDRPOSTagger is a POS tagger only.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError(
            "RDRPOSTagger does not support constituency parsing — "
            "it is a POS tagger only."
        )

    def _ensure_loaded(self) -> None:
        """Guard against calling analyze methods before load_model().

        Raises:
            RuntimeError: If load_model() has not been called.
        """
        self._require_loaded(self._root is not None and self._lexicon is not None)
