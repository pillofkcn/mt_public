"""Abstract base classes — the contract every analyzer must implement.

This module defines the BaseAnalyzer abstract class and five supporting enums:
AnalysisType, MethodLevel (with MethodType as its coarse reading),
AnnotationScheme and TrainingDomain. Between them they carry the three
categorisation axes listed below. Any new analyzer added to the framework must
subclass BaseAnalyzer and implement its abstract members.

How it works:
  1. BaseAnalyzer defines a unified interface: load_model(), analyze_dependency(),
     analyze_constituency(), analyze_pos()
  2. Every concrete analyzer (spaCy, Stanza, NLTK, ...) inherits from it
  3. The user interacts with all analyzers through the same API

Three categorisation axes:
  - AnalysisType      — WHAT the analyzer produces (POS, dependency, constituency)
  - MethodLevel       — HOW MUCH of the model is learned (an ordinal scale L0..L4;
                        MethodType is its coarse rule-based / statistical / neural
                        reading)
  - AnnotationScheme + TrainingDomain
                      — WHICH resources it stands on: the scheme its output follows
                        (UD, SD, PTB, none) and whether its training data overlaps
                        the test domain (in / in_plus / out / none)

The resource axis exists because it turned out to matter more than the method
axis. A Stanford-Dependencies parser scored against a Universal-Dependencies gold
loses around a quarter of its attachments to a scheme difference, not to parser
quality, and no amount of label mapping fixes that — the structures genuinely
differ. And on POS tagging the tools trained on the test treebank's own training
split outrank the tools trained elsewhere regardless of method: a perceptron
tagger trained on EWT beats a CNN tagger trained on OntoNotes. A comparison that
mixes schemes or domains therefore has to be reported as resource cost rather than
as a quality ranking, and that is only possible if scheme and domain are data the
framework carries rather than facts buried in comments.

The method axis is ORDINAL and PER COMPONENT, not one nominal label per tool. UDPipe
1 is the case that forced this: its tagger (MorphoDiTa) is an averaged perceptron,
its parser (Parsito) is a transition-based parser with a feed-forward neural-network
classifier — one tool, two levels. A single `method_type` for the whole tool would
have to be wrong for one of its components.

Output contract (enforced, not documented-and-hoped-for):
  analyze_dependency() and analyze_constituency() are CONCRETE here. They call the
  wrapper's _analyze_* method and then run the result through normalize.py, which
  verifies that pre-tokenised input came back with exactly the sentences and tokens
  it was given. Subclasses must override the underscore-prefixed methods; trying to
  override the public ones raises TypeError at class-creation time, because a
  wrapper that bypasses normalisation can silently produce plausible wrong numbers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from pathlib import Path
from typing import Mapping, NamedTuple

from syntactic_analysis import normalize
from syntactic_analysis.downloads import DownloadedFiles
from syntactic_analysis.models import ConstituencyParse, DependencyParse
from syntactic_analysis.normalize import ContractViolation
from syntactic_analysis.project import project_root

__all__ = [
    "AnalysisType",
    "MethodType",
    "MethodLevel",
    "AnnotationScheme",
    "TrainingDomain",
    "TrainingSeed",
    "TrainedModel",
    "ModelNotTrainedError",
    "require_trained",
    "BaseAnalyzer",
    "ContractViolation",
]


class AnalysisType(Enum):
    """Enumeration of syntactic analysis output types.

    An analyzer can support one or more of these types. For example,
    spaCy supports POS_TAGGING and DEPENDENCY_PARSING but not
    CONSTITUENCY_PARSING.
    """

    POS_TAGGING = auto()           # Assign a part-of-speech tag to each token (NOUN, VERB, ...)
    DEPENDENCY_PARSING = auto()    # Identify head-dependent relations between tokens
    CONSTITUENCY_PARSING = auto()  # Build a phrase-structure tree (nested NP, VP, PP, ...)


class MethodType(Enum):
    """Coarse method category: the three-way reading of MethodLevel.

    Kept as the thesis's headline vocabulary (rule-based / statistical / neural).
    It is derived from MethodLevel via `MethodLevel.coarse`; the level, declared
    per analysis type, is the authoritative classification.
    """

    RULE_BASED = auto()    # MethodLevel.RULE
    STATISTICAL = auto()   # MethodLevel.STATISTICAL
    NEURAL = auto()        # MethodLevel.NEURAL_SHALLOW .. NEURAL_PRETRAINED


class MethodLevel(IntEnum):
    """Ordinal scale of how much of a model is learned from data.

    Each level adds one kind of learning on top of the previous one, which is what
    makes the scale ordered, so an H1a chain steps from a lower level to a higher
    one (plan.py checks that each chain rises):

      L0 RULE              symbolic rules decide; no numeric parameter is optimised
                           on data. Hand-written cascades and corpus-induced
                           ones (RDRPOSTagger's SCRDR) both belong here — the
                           induction picks rules, not weights.
      L1 STATISTICAL       parameters learned from data over hand-defined discrete
                           features or n-grams; a linear or count-based model.
                           Averaged perceptron (NLTK, UDPipe's MorphoDiTa
                           tagger), logistic regression (nltk_transition), PCFG.
      L2 NEURAL_SHALLOW    learned dense representations (embeddings) plus a
                           feed-forward or convolutional network, with NO recurrent
                           or pretrained contextual encoder. UDPipe's Parsito
                           parser (one hidden layer), spaCy en_core_web_sm
                           (HashEmbed + CNN tok2vec, no pretrained vectors).
      L3 NEURAL_RECURRENT  a BiLSTM encoder over pretrained static word vectors
                           and/or a character model. Stanza (tagger, parser,
                           constituency + charlm), SuPar crf-con-en.
      L4 NEURAL_PRETRAINED a contextual encoder pretrained on unlabelled text,
                           fine-tuned or frozen under a trained head. Benepar
                           benepar_en3 (T5-small); Stanza combined_electra-large
                           and ptb3-revised_electra-large (frozen ELECTRA-large).

    The L2/L3 boundary is defined by two features — a recurrent encoder AND
    pretrained vectors — and spaCy sm has neither, which is why it is L2. That is a
    criterion applied consistently, not a verdict on quality. Nor can the placement
    move a trend: spaCy is trained on OntoNotes, domain OUT, so it enters no trend
    stratum, and placing it at L2 or at L3 changes no conclusion about a trend.

    Declared PER ANALYSIS TYPE on every analyzer (`BaseAnalyzer.method_level`),
    because one tool can sit at two levels: UDPipe 1 is L1 for POS and L2 for
    dependencies.
    """

    RULE = 0
    STATISTICAL = 1
    NEURAL_SHALLOW = 2
    NEURAL_RECURRENT = 3
    NEURAL_PRETRAINED = 4

    @property
    def coarse(self) -> MethodType:
        """The three-way category this level reads as."""
        if self is MethodLevel.RULE:
            return MethodType.RULE_BASED
        if self is MethodLevel.STATISTICAL:
            return MethodType.STATISTICAL
        return MethodType.NEURAL

    @property
    def code(self) -> str:
        """Short label for tables, e.g. 'L2'."""
        return f"L{int(self)}"

    @property
    def label(self) -> str:
        """Human-readable label, e.g. 'L2 neural, shallow'."""
        names = {
            MethodLevel.RULE: "rule-based",
            MethodLevel.STATISTICAL: "statistical (discrete features)",
            MethodLevel.NEURAL_SHALLOW: "neural, shallow (no contextual encoder)",
            MethodLevel.NEURAL_RECURRENT: "neural, recurrent (BiLSTM)",
            MethodLevel.NEURAL_PRETRAINED: "neural, pretrained transformer",
        }
        return f"{self.code} {names[self]}"


class TrainingDomain(Enum):
    """Whether an analyzer's training data overlaps the test domain.

    Relative to the thesis's primary gold, the UD English Web Treebank test split:

      IN      — the EWT training split (any release) and NOTHING ELSE. Stanza's
                `ewt` package (r2.15), and everything trained here on
                data/en_ewt-ud-train.conllu. (Before version 1.0 also UDPipe's
                published EWT r2.5 model, and until 24 Oct 2026 RDRPOSTagger's
                published EWT r2.4 model, which the wrappers no longer load.)
      IN_PLUS — the EWT training split PLUS other treebanks. Stanza's default
                `combined` package is EWT + GUM + GUMReddit + PUD + Pronouns.
      OUT     — trained on other data (WSJ Penn Treebank, OntoNotes 5).
      NONE    — nothing was trained (hand-written rules).

    IN and IN_PLUS are separate because the difference between them is a
    resource difference, and folding them together silently reintroduces the
    confound the whole axis exists to expose: with `combined` counted as IN, the
    only L3 member of the in-domain POS stratum was a model trained on several
    treebanks and a newer UD release than every L0 and L1 member, so an ordered
    trend over that stratum could be produced by training data rather than by
    architecture. A level trend therefore runs over IN only; IN_PLUS versus IN
    for the same architecture is itself one of the resource contrasts (RQ2).

    The value is the string used in CSV columns and tables. The constituency
    stage's WSJ run inverts the reading (every parser is then in-domain); that
    script says so explicitly rather than reading this property.

    The members are declared in READING ORDER, and that is a contract: a table
    that lists the domains takes its column order from this enum, so the order
    is stated once, here.
    """

    IN = "in"
    IN_PLUS = "in_plus"
    OUT = "out"
    NONE = "none"


class AnnotationScheme(Enum):
    """Annotation scheme an analyzer's output follows, per analysis type.

    UD  — Universal Dependencies labels AND structure. Directly comparable to the
          UD gold standard.
    SD  — Stanford Dependencies (including its ClearNLP/OntoNotes descendants).
          Labels can be mapped to UD; the structure cannot, because prepositions,
          copulas, auxiliaries and coordination attach differently.
    PTB — Penn Treebank constituency labels, or the PTB POS tagset.
    NONE— No committed scheme; what an undeclared scheme reads as.
    """

    UD = auto()
    SD = auto()
    PTB = auto()
    NONE = auto()


# What a wrapper reports for a fact only load_model() can establish, and the
# error it raises when analysis comes first. One spelling each: until
# 16 Oct 2026 the placeholder was written three ways and the guard with its
# message was copied into every wrapper. dokumentacia.txt quotes the message
# as the one the code prints.
NOT_LOADED = "unspecified (load_model not called)"
MODEL_NOT_LOADED = "Model not loaded. Call load_model() before analysis."


class TrainingSeed(NamedTuple):
    """The seed a wrapper states for the training of the model it loaded.

    `seed` is the value; `note` says where it is fixed and what it seeds, in
    the words the provenance of a stage records (stage.training_seed).
    """

    seed: int
    note: str


@dataclass(frozen=True)
class TrainedModel:
    """A model this project trains: the files load_model reads, and the script that writes them.

    The wrapper declares it (BaseAnalyzer.trained_model), and the declaration
    is the one place these paths are stated for a configuration: load_model
    refuses a model with a file missing (require_trained), plan/LOCK.json
    records the SHA-256 of every file (seal.model_digests), and a confirmatory
    run checks the files against the lock before the model is loaded
    (seal.verify_model). Since 24 Oct 2026, when the owner decided that a
    missing model trained here fails a build instead of being trained there.

    Attributes:
        script:      The training script that writes the files, relative to
                     the project folder: "scripts/train_udpipe.py".
        files:       The files load_model reads, as absolute paths.
        directories: The directories load_model reads as a whole (a spaCy
                     pipeline), as absolute paths: every file under one, at
                     any depth, is a file of the model.
    """

    script: str
    files: tuple[Path, ...] = ()
    directories: tuple[Path, ...] = ()

    def missing(self) -> list[Path]:
        """The files that are not on disk, and the directories that are not or hold no file."""
        absent = [path for path in self.files if not path.is_file()]
        absent += [
            path for path in self.directories
            if not path.is_dir() or not any(item.is_file() for item in path.rglob("*"))
        ]
        return absent


class ModelNotTrainedError(RuntimeError):
    """A model this project trains is not on disk.

    It is trained by its training script (TrainedModel.script), before the
    plan is locked; a build or a run never trains one. A RuntimeError, as
    every failure of load_model, so a stage records it as the configuration's
    failure and goes on with the others.
    """


def _shown(path: Path) -> str:
    """A path as an error message names it: relative to the project folder if inside it."""
    try:
        return path.resolve().relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return str(path)


def require_trained(name: str, model: TrainedModel) -> None:
    """Refuse to load a model this project trains while a file of it is missing.

    Called by load_model of every wrapper whose model this project trains,
    with its own declaration (BaseAnalyzer.trained_model), before anything is
    read. The message names the configuration, what is missing and the
    script that trains it.

    Raises:
        ModelNotTrainedError: If a declared file or directory is missing.
    """
    absent = model.missing()
    if absent:
        raise ModelNotTrainedError(
            f"{name}: the model is not trained: missing {', '.join(_shown(path) for path in absent)}. "
            f"Train it with {model.script}; a build or a run trains no model (since 24 Oct 2026)"
        )


class BaseAnalyzer(ABC):
    """Abstract base class that all analyzer wrappers must implement.

    Subclasses implement:
      - name, supported_types, method_type, method_detail   (identity)
      - method_level, annotation_scheme, training_domain,
        training_corpus                                     (comparability metadata)
      - load_model()                                        (setup)
      - _analyze_dependency(), _analyze_constituency()      (the actual work)

    and must NOT implement analyze_dependency() / analyze_constituency(), which are
    provided here and enforce the output contract.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        """Refuse subclasses that bypass the normalisation hook.

        Overriding the public methods would skip the alignment checks, which is how
        misaligned output stayed invisible in the first place. Fail at import time
        rather than at results time.
        """
        super().__init_subclass__(**kwargs)
        for method in ("analyze_dependency", "analyze_constituency"):
            if method in cls.__dict__:
                raise TypeError(
                    f"{cls.__name__} must not override {method}(); implement "
                    f"_{method}() instead. The public method applies the output "
                    f"contract (see normalize.py) and must not be bypassed."
                )

    # --- identity ---------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique lowercase key for this analyzer instance, e.g. 'spacy'.

        Used as the configuration key and as the CSV column name. Instances that
        wrap different models of the same tool need different keys, e.g.
        'stanza_ewt' and 'stanza_comb'.
        """

    @property
    def display_name(self) -> str:
        """Human-readable label for tables and figures, e.g. 'spaCy', 'SuPar'.

        Defaults to `name`. Kept separate because the canonical key must stay
        lowercase and stable while the label is free to be pretty.
        """
        return self.name

    @property
    @abstractmethod
    def supported_types(self) -> set[AnalysisType]:
        """Set of analysis types this analyzer can perform."""

    @property
    @abstractmethod
    def method_type(self) -> MethodType:
        """Coarse category of the tool as a whole: RULE_BASED, STATISTICAL or NEURAL.

        Shown by `__repr__`. It is NOT the authoritative
        classification: that is `method_level`, declared per analysis type,
        because a tool can sit at different levels for different tasks (UDPipe 1:
        perceptron tagger, neural parser). Use `method_type_for()` when the task
        is known.
        """

    @property
    @abstractmethod
    def method_detail(self) -> str:
        """Specific architecture, e.g. 'BiLSTM', 'CRF (CharLSTM encoder)'."""

    # --- comparability metadata -------------------------------------------

    @property
    def method_level(self) -> Mapping[AnalysisType, MethodLevel]:
        """Ordinal method level per analysis type — the authoritative classification.

        Override in every wrapper, for every supported type. The empty default
        means "undeclared", and `method_level_for()` refuses to guess: a level
        silently derived from `method_type` would put UDPipe's neural parser in the
        statistical row, which is exactly the mistake this axis exists to prevent.
        """
        return {}

    @property
    def annotation_scheme(self) -> Mapping[AnalysisType, AnnotationScheme]:
        """Scheme per analysis type.

        A mapping rather than a single value because it genuinely varies per
        capability: Stanza is UD for dependencies but PTB for constituency, and
        Benepar is UD for POS, SD for dependencies and PTB for constituency.

        Override in every wrapper. The empty default means "unspecified", and
        unlike an undeclared level or domain it does NOT show in the results
        tables: `scheme_for` reads it as NONE, so every table prints an
        undeclared scheme as NONE, where an undeclared level or domain is
        written as the "?" hole.
        """
        return {}

    @property
    def training_corpus(self) -> str:
        """What this analyzer's model was trained on, in human-readable form.

        Reported as a column next to every score. Without it, "statistical is worse
        than neural" is confounded with "trained on 100k tokens versus 1M", and the
        reader cannot tell which claim the numbers support.

        Version 1.0 records the per-task string `training_corpus_for` returns in
        the provenance of every configuration, and nothing reads it back. Until
        23 Oct 2026 evaluation/design.py also compared the strings of two rows
        to pair them, so a wrapper's corpus string could be reworded only
        together with that reader.
        """
        return "unspecified"

    @property
    def model_version(self) -> str:
        """Which release of the model is loaded, when the tool can say.

        Read into the provenance of every configuration. It was an
        undeclared contract until 9 Oct 2026: the table writer read it with
        `getattr(analyzer, "model_version", "")`, four of the eleven wrappers
        happened to provide it, and for the other seven the column was silently
        empty with nothing saying whether that meant "no version" or "nobody
        implemented it".

        Declaring it here did not settle that by itself. As of 16 Oct 2026 the
        wrappers of spaCy, Stanza, SuPar, the NLTK transition parser, UDPipe,
        Benepar and RDRPOSTagger implement it, and each returns "" until its
        model is loaded; since 23 Oct 2026 the NLTK wrapper does as well. The
        others return this default, and for them an empty string means "this
        wrapper records no version" - not that the model has none. A provenance
        record of the old stages names such a row by its training corpus
        instead (`collect_provenance`); the stages of version 1.0
        (syntactic_analysis.stage) refuse to score a configuration whose
        `model_version_for` its task is empty.
        """
        return ""

    def model_version_for(self, analysis_type: AnalysisType) -> str:
        """model_version of the model one analysis type runs. Override when it
        differs per task.

        A wrapper that loads one model for every task answers with
        `model_version`. One that loads a model per task (wrappers/nltk.py: the
        tagger for POS, the grammar for constituency) names only the model of
        the task asked about, as `training_corpus_for` names only its corpus.
        """
        return self.model_version

    @property
    def training_domain(self) -> Mapping[AnalysisType, TrainingDomain]:
        """Training-data domain relative to the EWT test set, per analysis type.

        The machine-readable companion of `training_corpus`: `training_corpus` is
        prose for the reader, this is what build.check_declarations compares with
        the domain the plan implies, and what a stage's provenance records.
        Override in every wrapper; the empty default means
        "undeclared" and `training_domain_for()` refuses to guess.
        """
        return {}

    def scheme_for(self, analysis_type: AnalysisType) -> AnnotationScheme:
        """Scheme for one analysis type, or NONE if the analyzer did not declare it."""
        return self.annotation_scheme.get(analysis_type, AnnotationScheme.NONE)

    def training_corpus_for(self, analysis_type: AnalysisType) -> str:
        """Training corpus for one analysis type. Override when it differs per task."""
        return self.training_corpus

    def training_seed_for(self, analysis_type: AnalysisType) -> TrainingSeed | None:
        """The seed the loaded model of one analysis type was trained with.

        None (the default) means this wrapper states no seed: its model is
        downloaded, or its seed is in a training record the stage reads
        (stage.training_seed). A wrapper whose model this project trains with
        a seed fixed in code - its own or its library's - overrides it. It
        reads the value from its own constant or from the loaded model; a
        seed that only a compiled library's source holds is a constant beside
        the citation of that source. Since 24 Oct 2026: the NLTK perceptron
        of nltk_ewt, the classifier of nltk_arc and UDPipe 1.
        """
        return None

    def trained_model(self, **load) -> TrainedModel | None:
        """The model this project trains that load_model(**load) would read, or None.

        None (the default) means the wrapper loads a published model, or one
        it builds at load from data it is given (the PCFG). A wrapper whose
        model a training script of this project writes overrides it: it takes
        the keywords load_model takes, ignores those that do not say where
        the model is, and names every file load_model reads and the script
        that writes them (TrainedModel). load_model calls it itself and
        refuses a missing file (require_trained), so what is declared and what
        is read cannot drift apart. Since 24 Oct 2026: udpipe_ewt, nltk_ewt,
        nltk_arc, spacy_ewt and rdr_ewt.
        """
        return None

    def downloaded_files(self, **load) -> tuple[DownloadedFiles, ...]:
        """The downloaded files load_model(**load) would read, root by root.

        Every file the load reads that was downloaded rather than trained
        here - a published model, the transformer it reads, the treebank a
        grammar is induced from, and since 25 Oct 2026 the data files a
        library its load imports reads from its own package directory
        (downloads.library_data) - as DownloadedFiles (downloads.py): the
        root it lies under and its path relative to it. Empty (the default)
        for a wrapper whose load reads nothing downloaded: one that reads
        only a model this project trains (trained_model). It takes the
        keywords load_model takes and reads no model; plan/LOCK.json records the
        SHA-256 of every file (seal.downloaded_digests), and a confirmatory
        run checks them right before the load (seal.verify_model), as it
        checks the models trained here. A file that is not on disk is
        declared all the same, where its library would look for it, so that
        the check reports it missing rather than the declaration failing.
        build.downloaded_files adds to every configuration the data files of
        the libraries every stage imports before it builds one
        (build.STAGE_LIBRARIES, since 25 Oct 2026). Since 24 Oct 2026: the
        Stanza, spaCy, Benepar, SuPar and NLTK wrappers.
        """
        return ()

    def method_detail_for(self, analysis_type: AnalysisType) -> str:
        """Architecture of one analysis type. Override when it differs per task.

        What a per-task `Architecture` cell prints. The tool-wide
        `method_detail` names every component, so for UDPipe and NLTK such a
        cell named the other task's model as well: until 16 Oct 2026 the UDPipe
        POS row read "... / Parsito FFNN transition parser" and the NLTK POS
        rows "Averaged Perceptron / PCFG".
        """
        return self.method_detail

    def method_level_for(self, analysis_type: AnalysisType) -> MethodLevel:
        """Method level for one analysis type.

        Raises:
            LookupError: If the wrapper did not declare a level for this type. This
                is deliberate — see `method_level`.
        """
        try:
            return self.method_level[analysis_type]
        except KeyError:
            raise LookupError(
                f"{self.name} declares no method level for {analysis_type.name}; "
                f"add it to `method_level` (see MethodLevel for the criteria)"
            ) from None

    def method_type_for(self, analysis_type: AnalysisType) -> MethodType:
        """Coarse category for one analysis type, derived from its level."""
        return self.method_level_for(analysis_type).coarse

    def training_domain_for(self, analysis_type: AnalysisType) -> TrainingDomain:
        """Training domain for one analysis type.

        Raises:
            LookupError: If the wrapper did not declare a domain for this type.
        """
        try:
            return self.training_domain[analysis_type]
        except KeyError:
            raise LookupError(
                f"{self.name} declares no training domain for {analysis_type.name}; "
                f"add it to `training_domain` (see TrainingDomain)"
            ) from None

    # --- setup ------------------------------------------------------------

    def _require_loaded(self, loaded: bool) -> None:
        """Raise RuntimeError(MODEL_NOT_LOADED) unless `loaded`.

        Each wrapper's `_ensure_loaded` says what loaded means for it - a
        tagger, a pipeline, a parser together with its POS source - and leaves
        the error to this method.
        """
        if not loaded:
            raise RuntimeError(MODEL_NOT_LOADED)

    @abstractmethod
    def load_model(self, model_name: str, **kwargs) -> None:
        """Download (if needed) and initialize a language model.

        Must be called before any analysis method; otherwise those raise RuntimeError.

        A wrapper that does not forward its keywords to the tool refuses the ones
        it does not know (_reject_unknown_options).
        """

    # --- analysis (concrete; do not override) -----------------------------

    def analyze_dependency(self, text: str | list[list[str]]) -> DependencyParse:
        """Run dependency parsing, then enforce the output contract.

        Args:
            text: Either a raw string (the analyzer tokenizes it itself), or a list
                  of pre-tokenized sentences, each a list of words.

                  Example raw:          "The cat sat."
                  Example pretokenized: [["The", "cat", "sat", "."]]

        Returns a DependencyParse whose tokens are 1-based, with head_index 0 for
        the root and -1 for "not determined". An analyzer that does not declare
        DEPENDENCY_PARSING does not raise here: a POS-only tagger returns its
        tags with every head and relation blanked, which is how `analyze_pos`
        reaches it.

        Raises:
            ContractViolation: If pre-tokenized input did not come back with the
                               same sentences and tokens.
            RuntimeError:      If load_model() was not called first, or did not
                               load the model this capability needs.
        """
        parse = self._analyze_dependency(text)
        return normalize.normalize_dependency(
            parse,
            text,
            analyzer_name=self.name,
            declares_dependency=self.supports(AnalysisType.DEPENDENCY_PARSING),
        )

    def analyze_constituency(self, text: str | list[list[str]]) -> ConstituencyParse:
        """Run constituency parsing, then enforce the output contract.

        Returns a ConstituencyParse with exactly one single-line bracketed tree per
        input sentence; an empty string marks a declared failure, with the reason in
        `failures`.

        Raises:
            ContractViolation:   On tree-count or leaf-count mismatch.
            NotImplementedError: If this analyzer does not declare constituency
                                 parsing at all.
            RuntimeError:        If load_model() was not called first, or did not
                                 load the model this capability needs.
        """
        parse = self._analyze_constituency(text)
        return normalize.normalize_constituency(
            parse, text, analyzer_name=self.name
        )

    def analyze_pos(self, text: str | list[list[str]]) -> DependencyParse:
        """Run POS tagging.

        Always routes to analyze_dependency(), whose tokens already carry POS tags
        — that is how every dependency-capable wrapper works. Nothing overrides
        this method; a POS-only analyzer implements `_analyze_dependency` and fills
        in the POS fields only, leaving head and relation blank (RDRPOSTagger,
        and NLTK configured with an EWT tagger). `normalize` blanks those fields
        for a wrapper that does not declare DEPENDENCY_PARSING, so a POS-only
        tool cannot be scored as a broken parser.

        Deliberately NOT in `__init_subclass__`'s guarded tuple: that guard tells
        an author to implement `_{method}` instead, and there is no `_analyze_pos`
        hook for anything to call.
        """
        return self.analyze_dependency(text)

    # --- to be implemented by wrappers ------------------------------------

    @abstractmethod
    def _analyze_dependency(self, text: str | list[list[str]]) -> DependencyParse:
        """Wrapper-specific dependency parsing. Called by analyze_dependency()."""

    @abstractmethod
    def _analyze_constituency(self, text: str | list[list[str]]) -> ConstituencyParse:
        """Wrapper-specific constituency parsing. Called by analyze_constituency()."""

    # --- helpers ----------------------------------------------------------

    def supports(self, analysis_type: AnalysisType) -> bool:
        """Check whether this analyzer supports a given analysis type."""
        return analysis_type in self.supported_types

    @staticmethod
    def _reject_unknown_options(options: Mapping[str, object]) -> None:
        """Refuse a load_model keyword this wrapper does not know.

        A mistyped option must not be ignored: a misspelt `train_path` would
        train on the default file and say nothing. The rule is one for every
        wrapper - a load_model that forwards its keywords to the tool keeps
        them, every other one calls this first - and until 12 Oct 2026 one
        wrapper refused such a keyword while the others swallowed it.

        Raises:
            TypeError: If `options` is not empty.
        """
        if options:
            raise TypeError(f"Unknown load_model options: {sorted(options)}")

    def __repr__(self) -> str:
        """Example: 'spacy_sm [POS_TAGGING, DEPENDENCY_PARSING] (NEURAL: CNN)'."""
        types = ", ".join(t.name for t in sorted(self.supported_types, key=lambda t: t.value))
        return f"{self.name} [{types}] ({self.method_type.name}: {self.method_detail})"
