"""NLTK TransitionParser — statistical transition-based dependency parsing.

Why this wrapper exists
-----------------------
The method axis is an ordinal level, declared per task (base.MethodLevel), and
the dependency comparison had no parser at L1, statistical over discrete
features. UDPipe 1 was first taken for one; its parser, Parsito, has a
neural-network classifier and is L2 (see wrappers/udpipe.py). Mixing other
languages into the project was ruled out, which eliminates the obvious
candidates (MaltParser and Stanford's parser are both Java).
``nltk.parse.TransitionParser`` is the in-stack alternative: Nivre's arc-eager /
arc-standard transition system driven by a classifier over sparse feature
templates — the family MaltParser belongs to. It is the L1 step of the H1a
dependency chain of plan/v1.yaml (nltk_arc -> udpipe_ewt -> stanza_ewt) and
the only feature-based statistical dependency parser in the comparison.

How it works
------------
  1. Training reads a CoNLL-U treebank, converts each sentence to an
     ``nltk.parse.DependencyGraph``, and runs the static oracle to turn every
     sentence into a sequence of (configuration -> transition) examples.
  2. A linear classifier is fitted over those examples (see SUBSTITUTION below).
  3. Parsing is greedy and deterministic: at each configuration the classifier
     ranks transitions and the highest-ranked *legal* one is applied.

SUBSTITUTION — this is NOT NLTK's default classifier
----------------------------------------------------
``TransitionParser.train()`` hardcodes ``svm.SVC(kernel="poly", degree=2,
probability=True)``. On the ~368k transitions the UD_English-EWT training split
produces that is computationally infeasible — a quadratic-kernel SVC with
Platt-scaling cross-validation over that many examples runs for days. This
wrapper subclasses ``TransitionParser`` and overrides ``train()`` to fit
``sklearn.linear_model.LogisticRegression`` instead. Everything else — the
feature set, the oracle, the transition system and the greedy decoding loop in
``TransitionParser.parse()`` — is upstream NLTK, unmodified.

LogisticRegression specifically (rather than, say, ``LinearSVC``) because the
untouched ``parse()`` loop ranks transitions with ``model.predict_proba(...)``,
which ``LinearSVC`` does not provide; swapping that in would have meant forking
the decoding loop as well.

The substituted technique is still "transition-based parser with a linear
classifier", which is what MaltParser with liblinear is — so the L1 placement
holds — but it IS a deviation from NLTK's shipped configuration and must be
reported, not hidden. ``method_detail`` therefore says "linear classifier" and
``model_version`` names LogisticRegression, never "SVM".

POS tags: gold at training time, predicted at test time
-------------------------------------------------------
The feature set is POS-heavy (``STK_0_POS``, ``STK_1_POS``, ``BUF_0..3_POS``),
so the parser needs a tag for every token it touches.

  * At TRAINING time it reads the treebank's own gold UPOS. That is normal and
    correct: the training split's annotation IS the supervision signal, and
    every trained parser in this project (UDPipe included) was built that way.
  * At TEST time it must NOT see gold tags — that would leak the gold standard
    into the prediction and inflate the score. Predicted UPOS therefore comes
    from another in-stack analyzer (``pos_source``, which plan/v1.yaml names
    for the row as its input: udpipe_ewt), and every row must be given one.
    ``training_corpus``, ``method_detail`` and ``model_version`` all record
    which tagger supplied them, because the score is only interpretable as
    "this parser on top of that tagger".

Expected quality
----------------
Below UDPipe, and that is the finding rather than a defect. A greedy arc-eager
parser with NLTK's deliberately basic feature set ("Please note that these
features are very basic" — upstream's own comment) and a linear classifier has
no beam, no dynamic oracle, no averaged perceptron over rich templates, and no
non-projective handling: ``TransitionParser._is_projective`` discards
non-projective training sentences outright (~2.2% of EWT train), and
non-projective test structures are simply unreachable for the decoder.

Install: nltk and scikit-learn (both already required by the project).
Model:   Trained by scripts/train_nltk.py (``NltkTransitionAnalyzer.train_model``)
         into <project_root>/model_cache/ — an ABSOLUTE path, so the cache is the
         same file regardless of the process's working directory — before the
         lock. ``load_model()`` reads the two files it writes and refuses a missing
         one, naming the script; it trains nothing. Until 24 Oct 2026 the first
         ``load_model()`` call trained the parser when the files were missing.
"""

from __future__ import annotations

import hashlib
import io
import os
import pickle
import tempfile
import time
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from typing import Iterable, Sequence

# Absolute import: this resolves to the top-level `nltk` package, not the sibling
# wrappers/nltk.py. Unlike the other wrappers, nltk cannot be deferred here —
# _LinearTransitionParser subclasses TransitionParser, which must exist at class
# definition time. sklearn stays deferred (it is only needed to train).
from nltk.parse import DependencyGraph
from nltk.parse.transitionparser import TransitionParser

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
from syntactic_analysis.checksum import sha256_of
from syntactic_analysis.data_loader import parse_conllu
from syntactic_analysis.models import (
    ConstituencyParse,
    DependencyParse,
    Sentence,
    Token,
)
from syntactic_analysis.project import ewt_split_path, model_cache_dir

ARC_EAGER = "arc-eager"
ARC_STANDARD = "arc-standard"
ALGORITHMS = (ARC_EAGER, ARC_STANDARD)

# The script that trains the parser (NltkTransitionAnalyzer.train_model), which
# a missing model's error names (TrainedModel.script). It trains the EWT
# perceptron of wrappers/nltk.py as well; no wrapper imports another, so the
# name is written in both.
TRAINING_SCRIPT = "scripts/train_nltk.py"

# UD spells the top relation "root"; DependencyGraph defaults to Malt's "ROOT"
# and warns once per sentence when it finds no node under that label.
_UD_TOP_RELATION = "root"

# Placeholders for the HEAD/DEPREL columns when a graph is built for PARSING
# rather than training. DependencyGraph skips any line whose HEAD is "_", and
# rstrip() would eat an empty trailing DEPREL cell, so both need a concrete
# non-empty value that the decoder overwrites anyway.
_DUMMY_HEAD = 0
_DUMMY_REL = "_"

# One row of the 4-column Malt-TAB form DependencyGraph parses:
# (FORM, UPOS, HEAD, DEPREL).
MaltRow = tuple[str, str, int, str]


# --- CoNLL-U -> DependencyGraph conversion (pure; no model, no nltk state) ----


def conllu_to_malt_rows(conllu_text: str) -> list[list[MaltRow]]:
    """Convert CoNLL-U text into per-sentence 4-column Malt-TAB rows.

    ``DependencyGraph`` does accept a 10-column CoNLL line, but its 10-column
    extractor maps column 4 (XPOS) onto the ``tag`` key that
    ``Configuration.extract_features()`` actually reads, and column 5 (FEATS)
    onto ``feats``, which explodes into one feature per morphological attribute.
    Neither is usable here: the tags available at test time are UPOS (from the
    ``pos_source`` analyzer), and a model trained on gold FEATS would face empty
    FEATS at test time. It also leaves the root under a "root" label that
    DependencyGraph's Malt default ("ROOT") cannot find, so it warns on every
    sentence. The CoNLL-U is therefore projected down to the 4-column Malt-TAB
    form, where ``tag`` is unambiguously the UPOS.

    Args:
        conllu_text: Contents of a .conllu file.

    Returns one list of (FORM, UPOS, HEAD, DEPREL) rows per sentence, in file
    order. Multi-word-token range lines ("2-3") and empty nodes ("8.1") are
    skipped, exactly as ``data_loader.load_conllu`` does, so token numbering
    stays contiguous and 1-based and HEAD values keep pointing at the right rows.
    The lines are read by ``data_loader.parse_conllu``, the reader the gold is
    loaded with; this function only projects its tokens onto the four columns.
    A token with no annotated head is left out, and so is a sentence that is
    then left with no row.
    """
    sentences: list[list[MaltRow]] = []
    for sentence in parse_conllu(conllu_text).sentences:
        rows = [
            (token.text, token.pos, token.head_index, token.dep_rel)
            for token in sentence.tokens
            if token.head_index != -1  # no annotated head — nothing to supervise on
        ]
        if rows:
            sentences.append(rows)
    return sentences


def _malt_cell_extractor(cells: Sequence[str], index: int):
    """Map a 4-column Malt-TAB row onto DependencyGraph's 8-tuple.

    Returns ``(index, word, lemma, ctag, tag, feats, head, rel)``.

    ``lemma`` is deliberately None. NLTK's own 4-column extractor copies the word
    form into the lemma slot, which makes ``extract_features()`` emit a
    ``STK_0_LEMMA_<form>`` feature that is a byte-for-byte duplicate of
    ``STK_0_FORM_<form>`` — pure redundancy that inflates the feature dictionary
    by roughly 40% for no signal. ``_check_informative(None)`` is False, so a
    None lemma is skipped instead.

    ``feats`` is "" for the same reason FEATS are dropped above: they are
    gold-only, so a model trained on them would find nothing there at test time.
    """
    word, tag, head, rel = cells
    return index, word, None, tag, tag, "", head, rel


def _sanitize_cell(value: str) -> str:
    """Make a cell safe for the tab-separated text DependencyGraph parses.

    A tab or newline inside a FORM would split one token into two cells or two
    lines and silently shift every HEAD in the sentence. CoNLL-U forbids both,
    but pre-tokenised input reaches this module from callers, not only from the
    treebank. Cells are only ever used to build feature STRINGS; the Token text
    the wrapper returns is the caller's original word, untouched.
    """
    return value.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def depgraph_from_rows(
    rows: Iterable[MaltRow], *, top_relation_label: str = _UD_TOP_RELATION
) -> DependencyGraph:
    """Build an ``nltk.parse.DependencyGraph`` from Malt-TAB rows.

    Args:
        rows: (FORM, UPOS, HEAD, DEPREL) tuples for ONE sentence, in token order.
        top_relation_label: The DEPREL the row attached to node 0 carries.
            DependencyGraph warns "doesn't contain a node that depends on the
            root element" unless it finds one, so this has to match the data:
            "root" for real UD rows, the dummy relation for the head-less graphs
            built by ``depgraph_for_tokens``. It is metadata only — nothing in
            TransitionParser reads ``graph.root``.

    Returns a DependencyGraph with node 0 as the artificial TOP and nodes 1..n as
    the tokens, so ``len(graph.nodes) == n + 1`` — which is what
    ``Configuration.__init__`` assumes when it fills its buffer with
    ``range(1, len(dep_graph.nodes))``.

    The rows are serialised to tab-separated text rather than poked into
    ``graph.nodes`` directly, so DependencyGraph still builds its own ``deps``
    index and root pointer; ``_is_projective()`` reads those.
    """
    text = "\n".join(
        f"{_sanitize_cell(form)}\t{_sanitize_cell(tag)}\t{head}\t{_sanitize_cell(rel)}"
        for form, tag, head, rel in rows
    )
    # cell_separator="\t" rather than the default whitespace split, because a UD
    # FORM may legitimately contain a space.
    return DependencyGraph(
        text,
        cell_extractor=_malt_cell_extractor,
        cell_separator="\t",
        top_relation_label=top_relation_label,
    )


def depgraph_for_tokens(words: Sequence[str], tags: Sequence[str]) -> DependencyGraph:
    """Build a head-less DependencyGraph for parsing.

    The decoder reads only FORM and POS off the input graph; HEAD and DEPREL are
    placeholders it overwrites. Keeping them present but dummy is what lets one
    4-column reader serve both training and parsing.

    Args:
        words: Tokens of one sentence.
        tags:  Predicted UPOS, one per token. These must NOT be gold tags — see
               the module docstring.

    Raises:
        ValueError: If words and tags have different lengths.
    """
    if len(words) != len(tags):
        raise ValueError(
            f"words/tags length mismatch: {len(words)} tokens, {len(tags)} tags"
        )
    # "X" (the UD "other" tag) rather than "" for a missing tag: _check_informative
    # rejects "", so an untagged token would contribute no POS feature at all and
    # the configuration would look nothing like anything seen in training.
    return depgraph_from_rows(
        ((word, tag or "X", _DUMMY_HEAD, _DUMMY_REL) for word, tag in zip(words, tags)),
        # Every dummy row points at node 0 under _DUMMY_REL, so naming that as the
        # top relation is what stops DependencyGraph warning once per sentence that
        # it found no root — these graphs have no real heads yet, by design.
        top_relation_label=_DUMMY_REL,
    )


def heads_and_rels(depgraph: DependencyGraph, n_tokens: int) -> list[tuple[int, str]]:
    """Read (head, relation) back off a parsed DependencyGraph, for tokens 1..n.

    Args:
        depgraph: A graph returned by ``TransitionParser.parse()``.
        n_tokens: Number of real tokens, so the artificial TOP node (0) is
                  excluded and the underlying defaultdict is never grown past the
                  end of the sentence.

    Returns a list of (head_index, dep_rel) in token order; head 0 means root.

    On tokens the decoder never attached: ``TransitionParser.parse()`` initialises
    every node to ``head=0, rel=""`` and then writes only the arcs it actually
    produced, so an unattached token surfaces as an extra root carrying an EMPTY
    relation label. That is upstream NLTK's own behaviour and it is left alone
    rather than repaired — inventing a "root" label for those tokens would hand
    the parser free LAS credit it did not earn. Callers that care can count
    ``head == 0`` per sentence.
    """
    out: list[tuple[int, str]] = []
    for index in range(1, n_tokens + 1):
        node = depgraph.nodes[index]
        head = node.get("head")
        out.append((0 if head is None else int(head), node.get("rel") or ""))
    return out


# --- the classifier substitution ---------------------------------------------


class _PreloadedClassifier:
    """A ``pickle`` stand-in whose ``load`` returns an already-fitted model.

    Scoped to one ``_LinearTransitionParser.parse()`` call. Every other attribute
    is delegated, so upstream code that reaches for ``pickle.dump`` or
    ``pickle.HIGHEST_PROTOCOL`` inside that window behaves normally.
    """

    def __init__(self, model, original) -> None:
        self._model = model
        self._original = original

    def load(self, handle):
        # The handle upstream opened is simply not read. Closing it here rather
        # than leaving it to the garbage collector keeps the file descriptor
        # count flat across 2077 calls.
        try:
            handle.close()
        except Exception:  # noqa: BLE001 - a closed or unclosable handle is fine
            pass
        return self._model

    def __getattr__(self, name):
        return getattr(self._original, name)


class _LinearTransitionParser(TransitionParser):
    """NLTK's TransitionParser with a linear classifier instead of a poly SVM.

    Only ``train()`` is overridden. The oracle, the feature extractor, the
    transition system and ``parse()`` are upstream code, unmodified — so what is
    measured here is NLTK's parsing behaviour with a different, and for this data
    size a feasible, classifier plugged into the same slot. See the module
    docstring for the full rationale.

    State beyond upstream's:
        _model:       The fitted sklearn classifier. Upstream keeps it only on
                      disk; keeping it on the instance is what makes pickling the
                      WHOLE parser a self-sufficient artifact.
        _train_stats: Counts recorded at fit time, surfaced by the wrapper's
                      ``training_corpus`` and ``model_version``.
    """

    def __init__(
        self,
        algorithm: str,
        *,
        max_iter: int = 300,
        C: float = 1.0,
        random_state: int = 0,
    ) -> None:
        """Args mirror LogisticRegression's.

        ``C`` is sklearn's own default — deliberately untuned, because a
        hyperparameter search against the test gold is exactly the kind of
        thing that makes a reported number meaningless. ``random_state`` is
        fixed at 0 (sklearn's default is None). The lbfgs solver ``train()``
        uses does not read it: scikit-learn uses it only for the sag, saga and
        liblinear solvers. The fitted classifier keeps it among its
        parameters, and the stage's provenance records it from there
        (``NltkTransitionAnalyzer.training_seed_for``). Until 24 Oct 2026 this
        docstring called both values sklearn's defaults.

        ``max_iter`` is the one exception, raised from sklearn's 100 because on
        the full EWT training split lbfgs hits the limit and stops with a
        ConvergenceWarning. Letting the optimizer finish is a correctness
        property of the fit, not a score tweak; ``model_version`` reports
        ``converged`` and ``n_iter`` either way, so a run that still hit the
        ceiling is visible rather than assumed away.
        """
        super().__init__(algorithm)
        self._model = None
        self._max_iter = max_iter
        self._C = C
        self._random_state = random_state
        self._train_stats: dict[str, object] = {}

    def parse(self, depgraphs, modelFile=None):
        """Decode with the fitted model this instance already holds.

        Upstream's ``parse()`` begins with ``pickle.load(open(modelFile, "rb"))``,
        so the 37.4 MB classifier was deserialised once per CALL — and the measuring
        loop calls ``_analyze_dependency`` once per sentence, which is 2077 times in a
        dependency pass. Measured with a warm cache: 15.9 ms of the 20.5 ms a
        one-sentence ``parse()`` took was the unpickle. It understated this row's
        published throughput by roughly a third and made most of its batch-32 gain
        an amortised reload rather than anything about the parser.

        The decoding loop is NOT forked: this calls ``super().parse()`` and only
        intercepts its model-loading line, for the duration of that one call, with
        the identical object. ``train()`` pickles exactly ``self._model`` into
        that file, so what upstream would read back and what it is handed here
        are the same fitted classifier, and the predictions are unchanged —
        which the per-sentence CSVs are checked against byte for byte.

        Not thread-safe, because the interception is a module attribute on
        ``nltk.parse.transitionparser``. Nothing in this project parses
        concurrently, and the alternative — copying upstream's seventy-line decode
        loop into this file — would change what the measurement is OF.
        """
        if self._model is None:
            # Fall through to upstream's own loading, so a parser restored from a
            # pickle without this attribute still works.
            return super().parse(depgraphs, modelFile)

        from nltk.parse import transitionparser as _upstream

        original = _upstream.pickle
        _upstream.pickle = _PreloadedClassifier(self._model, original)
        try:
            return super().parse(depgraphs, modelFile)
        finally:
            _upstream.pickle = original

    def train(self, depgraphs, modelfile, verbose: bool = True) -> None:
        """Fit a linear classifier over the oracle's transitions.

        Identical to ``TransitionParser.train()`` up to and including the
        libsvm-format example file; from there the poly-kernel
        ``svm.SVC(..., probability=True)`` is replaced by ``LogisticRegression``.

        Args:
            depgraphs: Training DependencyGraphs, with gold heads and relations.
            modelfile: Path the fitted classifier is pickled to, in exactly the
                       layout ``TransitionParser.parse()`` expects to read back.
            verbose:   Print the oracle's per-corpus progress lines. Upstream's
                       ``_create_training_examples_*`` print unconditionally, so
                       they are silenced here rather than asked nicely.

        Raises:
            ValueError: If the oracle produced no usable training examples (e.g.
                every input sentence was non-projective or empty).
        """
        import numpy as np
        from sklearn.datasets import load_svmlight_file
        from sklearn.linear_model import LogisticRegression

        started = time.perf_counter()
        input_file = tempfile.NamedTemporaryFile(
            prefix="transition_parse.train", dir=tempfile.gettempdir(), delete=False
        )
        try:
            quiet = nullcontext() if verbose else redirect_stdout(io.StringIO())
            with quiet:
                if self._algorithm == self.ARC_STANDARD:
                    self._create_training_examples_arc_std(depgraphs, input_file)
                else:
                    self._create_training_examples_arc_eager(depgraphs, input_file)
            input_file.close()

            if not self._dictionary:
                raise ValueError(
                    "no training examples were produced — every input sentence "
                    "was rejected (non-projective or empty)"
                )

            # n_features is pinned to the dictionary size rather than inferred from
            # the file. parse() builds its test vectors with
            # shape=(1, len(self._dictionary)), so the two must agree; relying on
            # the highest feature id happening to occur in the file works today but
            # is exactly the kind of coincidence that breaks quietly later.
            x_train, y_train = load_svmlight_file(
                input_file.name,
                n_features=len(self._dictionary),
                zero_based=True,  # _convert_to_binary_features numbers from 0
            )

            model = LogisticRegression(
                C=self._C,
                max_iter=self._max_iter,
                solver="lbfgs",  # the only solver here that fits a true multinomial
                random_state=self._random_state,
            )
            model.fit(x_train, y_train)

            # Store coef_ column-major. parse() calls predict_proba() once per
            # transition, and sklearn's decision_function computes
            # safe_sparse_dot(X, coef_.T); with a C-ordered coef_ that transpose is
            # F-ordered, so scipy copies the whole ~38 MB coefficient matrix into C
            # order on EVERY transition. Measured on this model: 7.6 ms/call
            # C-ordered vs 0.034 ms/call F-ordered — 220x, or ~6 minutes of pure
            # memcpy over the test set. The probabilities are bit-identical.
            model.coef_ = np.asfortranarray(model.coef_)

            self._model = model
            self._train_stats = {
                "n_transitions": int(x_train.shape[0]),
                "n_features": int(x_train.shape[1]),
                "n_classes": int(len(model.classes_)),
                "n_iter": int(np.max(model.n_iter_)),
                "max_iter": int(self._max_iter),
                "converged": bool(np.max(model.n_iter_) < self._max_iter),
                "fit_seconds": round(time.perf_counter() - started, 1),
            }

            with open(modelfile, "wb") as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            os.remove(input_file.name)


# --- the wrapper --------------------------------------------------------------


class NltkTransitionAnalyzer(BaseAnalyzer):
    """Transition-based dependency parser (NLTK) driven by a linear classifier."""

    def __init__(
        self,
        pos_source: BaseAnalyzer,
        *,
        key: str | None = None,
    ) -> None:
        """Create an unloaded analyzer.

        Args:
            pos_source: Analyzer supplying PREDICTED UPOS at parse time; must
                already be loaded when this one analyzes. Required, so that the
                tags the parser reads are the ones its configuration names (in
                plan/v1.yaml, `inputs.pos_source`) and never a tagger chosen by
                omission. Until 23 Oct 2026 a row without a key defaulted to
                UDPipe with the published UD 2.5 model, the tagger the stage
                scripts before version 1.0 gave this parser.

                There is deliberately no option to feed gold tags here: doing so
                at test time would leak the gold standard into the prediction.
            key: Key and CSV column name, overriding "nltk_transition".

        Raises:
            ValueError: If `pos_source` is None.
        """
        if pos_source is None:
            raise ValueError(
                f"NltkTransitionAnalyzer {key or 'nltk_transition'!r} needs its "
                f"pos_source: the analyzer whose UPOS it parses over"
            )
        self._key = key
        self._parser: _LinearTransitionParser | None = None
        self._classifier_path: Path | None = None  # what TransitionParser.parse() reads
        self._cache_path: Path | None = None       # the whole-parser pickle
        self._model_files = ""                     # "<pickle>@<digest of both files>", load_model
        self._algorithm = ARC_EAGER
        self._pos_source = pos_source
        self._train_meta: dict[str, object] = {}

    # --- identity ---------------------------------------------------------

    @property
    def name(self) -> str:
        return self._key or "nltk_transition"

    @property
    def pos_source(self) -> BaseAnalyzer:
        """The analyzer whose UPOS the parser reads at test time."""
        return self._pos_source

    @property
    def display_name(self) -> str:
        return "NLTK-TransitionParser"

    @property
    def supported_types(self) -> set[AnalysisType]:
        # Dependencies only. The UPOS on the output tokens is BORROWED from
        # pos_source, not produced here, so declaring POS_TAGGING would credit
        # this analyzer with another tool's tagging accuracy.
        return {AnalysisType.DEPENDENCY_PARSING}

    @property
    def method_type(self) -> MethodType:
        return MethodType.STATISTICAL

    @property
    def method_detail(self) -> str:
        """e.g. 'Transition-based (arc-eager, linear classifier)'.

        Says "linear classifier", never "SVM": NLTK ships a degree-2 polynomial
        SVC and this wrapper substitutes LogisticRegression (module docstring).
        Once loaded it also names the tagger the POS features came from, because
        the score is only interpretable as "this parser on top of that tagger".
        """
        detail = f"Transition-based ({self._algorithm}, linear classifier)"
        if self._pos_source is not None:
            detail += f", POS from {self._pos_source.name}"
        return detail

    # --- comparability metadata -------------------------------------------

    @property
    def annotation_scheme(self) -> dict:
        # Trained on UD_English-EWT, so both the labels AND the structure are
        # already Universal Dependencies. No SD->UD mapping must be applied to
        # this analyzer's output — that would corrupt labels that are correct.
        return {AnalysisType.DEPENDENCY_PARSING: AnnotationScheme.UD}

    @property
    def method_level(self) -> dict:
        # Logistic regression over NLTK's sparse discrete feature templates —
        # no embeddings, no hidden layer. L1, the only feature-based statistical
        # dependency parser in the comparison (UDPipe's parser is L2).
        return {AnalysisType.DEPENDENCY_PARSING: MethodLevel.STATISTICAL}

    @property
    def training_domain(self) -> dict:
        return {AnalysisType.DEPENDENCY_PARSING: TrainingDomain.IN}

    @property
    def training_corpus(self) -> str:
        """Split, size, and the tagger that supplies test-time POS.

        The split and its size are given in the form the NLTK and UDPipe
        wrappers give the same split, release included: "current split" is the
        file on disk, which no UD release tag identifies (data/DATASET.md).
        Until 16 Oct 2026 this cell named no release and counted
        "(N sentences, ... tokens)".
        """
        if not self._train_meta:
            return NOT_LOADED

        meta = self._train_meta
        parts = [
            f"UD_English-EWT train (current split), "
            f"{meta.get('n_sentences', '?')} sent / {meta.get('n_tokens', '?')} tok "
            f"({meta.get('n_projective', '?')} projective; "
            f"gold UPOS as training features)"
        ]
        if meta.get("max_train_sentences"):
            parts.append(f"CAPPED at {meta['max_train_sentences']} sentences")
        pos_name = self._pos_source.name if self._pos_source is not None else "?"
        parts.append(f"test-time UPOS predicted by {pos_name}")
        return "; ".join(parts)

    @property
    def model_version(self) -> str:
        """Concrete trained-model identity, for provenance logging.

        There is no published model to name — this one is trained in-place — so
        the identity is the files it was loaded from and the configuration that
        produced them. First the parser pickle in model_cache/ by name, with one
        SHA-256 prefix of the two files parse() depends on (the pickle, which
        holds the feature dictionary, then the classifier file it reads), as
        the other models trained here are named (since 24 Oct 2026); then the
        algorithm, classifier and its convergence, the treebank file by SHA-256
        prefix, and the fitted model's shape. The tagger the test-time UPOS
        comes from is named after it, with that tagger's own model_version
        where it records one: the score is this parser's on top of that model.

        Returns:
            Empty string if load_model() has not been called yet.
        """
        if self._parser is None:
            return ""
        import sklearn

        meta = dict(self._train_meta)
        meta.update(self._parser._train_stats)
        tagger = ""
        if self._pos_source is not None:
            tagger = f"; POS from {self._pos_source.name}"
            if self._pos_source.model_version:
                tagger += f" [{self._pos_source.model_version}]"
        files = f"{self._model_files}; " if self._model_files else ""
        return (
            f"{files}{self._algorithm} + LogisticRegression(lbfgs, C={self._parser._C}, "
            f"max_iter={meta.get('max_iter')}, n_iter={meta.get('n_iter')}, "
            f"converged={meta.get('converged')}) "
            f"[{meta.get('n_features', '?')} features, "
            f"{meta.get('n_classes', '?')} transitions, "
            f"{meta.get('n_transitions', '?')} training examples] "
            f"train={meta.get('train_file', '?')}@{meta.get('train_sha256', '?')} "
            f"(nltk {_nltk_version()}, sklearn {sklearn.__version__}){tagger}"
        )

    def training_seed_for(self, analysis_type: AnalysisType) -> TrainingSeed | None:
        """The random_state the loaded classifier was fitted with; None before load.

        Read from the fitted LogisticRegression itself (its parameters travel
        with it in the cached pickle), not from the default in
        ``_LinearTransitionParser``, so it is the value of the model that runs.
        The note says why it decides nothing: the solver is lbfgs, which does
        not read random_state, and the rest of the training draws no random
        number - the oracle turns the treebank's projective sentences into
        examples in file order, the feature ids are assigned in order of first
        appearance, and load_svmlight_file reads them back as written.
        """
        model = getattr(self._parser, "_model", None)
        if model is None or analysis_type is not AnalysisType.DEPENDENCY_PARSING:
            return None
        params = model.get_params()
        return TrainingSeed(
            params["random_state"],
            f"random_state of the fitted LogisticRegression (solver {params['solver']}), "
            f"read from the loaded classifier; scikit-learn reads random_state only for "
            f"the sag, saga and liblinear solvers, and nothing else in the training is "
            f"random (NLTK's oracle writes the examples in treebank order)",
        )

    # --- setup ------------------------------------------------------------

    def trained_model(
        self,
        model_name: str = ARC_EAGER,
        *,
        algorithm: str | None = None,
        cache_dir: str | Path | None = None,
        max_train_sentences: int | None = None,
        **load,
    ) -> TrainedModel:
        """The two files scripts/train_nltk.py writes, which load_model reads.

        Takes load_model's keywords; model_name or algorithm, cache_dir and
        max_train_sentences say where the model is (`model_paths`). Both files
        are the model: the whole-parser pickle, with the feature dictionary,
        and the classifier file parse() reads. Never None: the parser of this
        wrapper is trained here. Since 24 Oct 2026.

        Raises:
            ValueError: If the algorithm is unknown.
        """
        algorithm = _algorithm_of(model_name, algorithm)
        return TrainedModel(
            TRAINING_SCRIPT, files=model_paths(algorithm, cache_dir, max_train_sentences),
        )

    @classmethod
    def train_model(
        cls,
        model_name: str = ARC_EAGER,
        *,
        train_path: str | Path | None = None,
        algorithm: str | None = None,
        cache_dir: str | Path | None = None,
        max_train_sentences: int | None = None,
        verbose: bool = True,
    ) -> tuple[Path, Path]:
        """Train the parser on a CoNLL-U file and write its two files: the model of nltk_arc.

        The one place this parser is trained. scripts/train_nltk.py calls it
        before the lock; load_model never does (since 24 Oct 2026). Until then
        load_model trained the parser, with this same code, when its pickle
        was missing (or with force_retrain), and wrote the classifier file
        again from the pickle when only that was missing.

        Args:
            model_name: "arc-eager" (default) or "arc-standard"; it also
                supplies the transition algorithm unless ``algorithm``
                overrides it, and names the files (`model_paths`).
            train_path: CoNLL-U training file. Defaults to
                <project_root>/data/en_ewt-ud-train.conllu.
            algorithm: Transition system, "arc-eager" or "arc-standard".
            cache_dir: Where the files are written. Defaults to
                <project_root>/model_cache — an ABSOLUTE path, so the cache is the same
                file regardless of the process's working directory. A relative
                default here previously cost this repo a duplicated 16 MB model
                download (see udpipe.py).
            max_train_sentences: Train on only the first N sentences. For the
                out-of-repository test suite; it gets its own file names so
                a 500-sentence model can never be mistaken for the full one,
                and ``training_corpus`` reports the cap.
            verbose: Print training progress. Training takes minutes, so a silent
                run is worse than a noisy one.

        Returns the two files written: the whole-parser pickle and the
        classifier file.

        Raises:
            ValueError: If the algorithm is unknown, or the oracle produced no
                training example.
            FileNotFoundError: If the training file does not exist.
        """
        algorithm = _algorithm_of(model_name, algorithm)
        pickle_path, classifier_path = model_paths(algorithm, cache_dir, max_train_sentences)
        pickle_path.parent.mkdir(parents=True, exist_ok=True)
        _train_parser(
            algorithm,
            Path(train_path) if train_path else ewt_split_path("train"),
            pickle_path,
            classifier_path,
            max_train_sentences=max_train_sentences,
            verbose=verbose,
        )
        return pickle_path, classifier_path

    def load_model(
        self,
        model_name: str = ARC_EAGER,
        *,
        algorithm: str | None = None,
        cache_dir: str | Path | None = None,
        max_train_sentences: int | None = None,
        **kwargs,
    ) -> None:
        """Load the trained parser from the two files scripts/train_nltk.py wrote.

        Nothing is trained here: a missing file is refused, naming the script
        (require_trained). Until 24 Oct 2026 this method trained the parser
        when its pickle was missing, took train_path, force_retrain and
        verbose for that, and wrote the classifier file again from the pickle
        when only that file was missing; a build or a run now writes no model
        file.

        Args:
            model_name: "arc-eager" (default) or "arc-standard"; it also
                supplies the transition algorithm unless ``algorithm``
                overrides it.
            algorithm: Transition system, "arc-eager" or "arc-standard". Defaults
                to ``model_name`` when that names an algorithm.
            cache_dir: Override for the cache directory. Defaults to
                <project_root>/model_cache — an ABSOLUTE path, so the cache is the same
                file regardless of the process's working directory.
            max_train_sentences: Load the model trained on the first N
                sentences (train_model with the same cap), which has its own
                file names. For the out-of-repository test suite.

        Raises:
            ValueError: If the algorithm is unknown.
            TypeError: If an unrecognised keyword argument is passed — silently
                ignoring a mistyped option would load another model than the
                one meant.
            base.ModelNotTrainedError: If a file of the model is missing.
            RuntimeError: If the cached pickle is unreadable or incomplete.
        """
        self._reject_unknown_options(kwargs)
        trained = self.trained_model(
            model_name, algorithm=algorithm, cache_dir=cache_dir,
            max_train_sentences=max_train_sentences,
        )
        require_trained(self.name, trained)
        self._algorithm = _algorithm_of(model_name, algorithm)
        # The whole-parser pickle, and the classifier file parse() reads: a
        # SECOND artifact, because TransitionParser.parse() re-reads and
        # unpickles its model file on every call, so it needs a file holding
        # ONLY the classifier in the exact layout upstream wrote it.
        self._cache_path, self._classifier_path = trained.files
        self._load_cached()
        self._model_files = (
            f"{self._cache_path.name}@"
            f"{_files_digest(self._cache_path, self._classifier_path)[:12]}"
        )

    def _load_cached(self) -> None:
        """Restore the parser — feature dictionary included — from the cache.

        Raises:
            RuntimeError: If the pickle is unreadable, or holds a parser with an
                empty feature dictionary. That check is the whole point of
                caching the parser rather than the classifier: without it, the
                failure is silent and looks like a merely bad parser.
        """
        try:
            with open(self._cache_path, "rb") as handle:
                payload = pickle.load(handle)
            parser = payload["parser"]
            meta = payload["meta"]
        except Exception as exc:  # corrupt or partial cache — say so, do not limp on
            raise RuntimeError(
                f"Could not read cached model {self._cache_path}: {exc}. "
                f"Train it again with {TRAINING_SCRIPT} --force."
            ) from exc

        if not getattr(parser, "_dictionary", None):
            raise RuntimeError(
                f"Cached model {self._cache_path} has an empty feature dictionary; "
                f"it would parse every sentence from all-zero features. Train it "
                f"again with {TRAINING_SCRIPT} --force."
            )

        self._parser = parser
        self._train_meta = meta

    # --- analysis ---------------------------------------------------------

    def _analyze_dependency(self, text: str | list[list[str]]) -> DependencyParse:
        """Tag with ``pos_source``, then decode greedily with the transition model.

        Args:
            text: Either a raw string (``pos_source`` tokenizes and sentence-splits
                  it) or a list of pre-tokenized sentences.

                  Example raw:          "The cat sat."
                  Example pretokenized: [["The", "cat", "sat", "."]]

        Returns a DependencyParse with 1-based token indices, ``head_index`` 0 for
        the root, UD relation labels, and POS tags borrowed from ``pos_source``
        (see ``supported_types`` on why that is not claimed as POS tagging).

        Raises:
            RuntimeError: If load_model() was not called first.
        """
        self._ensure_loaded()

        # The tagger owns tokenization and sentence splitting too, so raw and
        # pre-tokenised input follow exactly the same path from here. For
        # pre-tokenised input the tagger's own output contract already guarantees
        # the tokens come back unchanged and in order.
        tagged = self._pos_source.analyze_pos(text)
        words_per_sentence = [[t.text for t in s.tokens] for s in tagged.sentences]
        tags_per_sentence = [[t.pos for t in s.tokens] for s in tagged.sentences]

        # A zero-token sentence would hand Configuration an empty buffer and come
        # back with nothing to read, so it is kept out of the batch and restored as
        # an empty Sentence afterwards — the one-sentence-per-input contract holds
        # either way.
        parseable = [i for i, words in enumerate(words_per_sentence) if words]
        parsed = (
            self._parser.parse(
                [
                    depgraph_for_tokens(words_per_sentence[i], tags_per_sentence[i])
                    for i in parseable
                ],
                str(self._classifier_path),
            )
            if parseable
            else []
        )
        by_index = dict(zip(parseable, parsed))

        sentences: list[Sentence] = []
        for i, words in enumerate(words_per_sentence):
            arcs = heads_and_rels(by_index[i], len(words)) if i in by_index else []
            tokens = [
                Token(
                    text=word,
                    index=position,
                    pos=tag,  # borrowed from pos_source, not produced here
                    head_index=head,
                    dep_rel=rel,
                )
                for position, (word, tag, (head, rel)) in enumerate(
                    zip(words, tags_per_sentence[i], arcs), start=1
                )
            ]
            sentences.append(Sentence(text=" ".join(words), tokens=tokens))

        return DependencyParse(sentences=sentences, analyzer_name=self.name)

    def _analyze_constituency(self, text: str | list[list[str]]) -> ConstituencyParse:
        """Not supported — a transition parser produces dependencies only.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError(
            "NLTK's TransitionParser does not support constituency parsing."
        )

    def _ensure_loaded(self) -> None:
        """Guard against calling analyze methods before model initialization."""
        self._require_loaded(self._parser is not None and self._pos_source is not None)


# --- the training (scripts/train_nltk.py) --------------------------------------


def _algorithm_of(model_name: str, algorithm: str | None) -> str:
    """The transition algorithm: `algorithm`, else `model_name`.

    Raises:
        ValueError: If it is not one of ALGORITHMS.
    """
    algorithm = algorithm or model_name
    if algorithm not in ALGORITHMS:
        raise ValueError(
            f"unknown algorithm {algorithm!r}; expected one of {sorted(ALGORITHMS)}"
        )
    return algorithm


def model_paths(
    algorithm: str = ARC_EAGER,
    cache_dir: str | Path | None = None,
    max_train_sentences: int | None = None,
) -> tuple[Path, Path]:
    """The two files of the parser trained with `algorithm`: the pickle and the classifier.

    <cache_dir>/nltk_transition-<algorithm>[-<N>sent].pkl and the same stem
    with .classifier.pkl; <project_root>/model_cache when cache_dir is None.
    A model trained on the first N sentences (the test suite) has its own
    names, so it can never be mistaken for the full one.
    """
    directory = Path(cache_dir) if cache_dir is not None else model_cache_dir()
    stem = f"nltk_transition-{algorithm}"
    if max_train_sentences:
        stem += f"-{max_train_sentences}sent"
    return directory / f"{stem}.pkl", directory / f"{stem}.classifier.pkl"


def _train_parser(
    algorithm: str,
    train_path: Path,
    pickle_path: Path,
    classifier_path: Path,
    *,
    max_train_sentences: int | None,
    verbose: bool,
) -> None:
    """Train from a CoNLL-U file and cache the WHOLE parser object.

    Pickling the whole parser — not just the classifier, as NLTK's own
    ``train()`` does — is deliberate. ``TransitionParser`` keeps
    ``_dictionary`` (feature string -> column index), ``_transition`` and
    ``_match_transition`` on the INSTANCE and saves none of them. Loading only
    the classifier into a fresh instance therefore gives a parser whose
    feature dictionary is empty: every test vector comes out all-zero, nothing
    raises, and the parser silently emits garbage that still looks like a
    bad-but-plausible score. The classifier alone is written to a second,
    derived file (by ``_LinearTransitionParser.train``) purely because
    upstream ``parse()`` insists on reading its model from disk.

    The body of NltkTransitionAnalyzer._train until 24 Oct 2026, unchanged
    but for the first line it prints, so a model it writes is the model the
    wrapper trained then.
    """
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    started = time.perf_counter()
    rows_per_sentence = conllu_to_malt_rows(
        train_path.read_text(encoding="utf-8")
    )
    if max_train_sentences:
        rows_per_sentence = rows_per_sentence[:max_train_sentences]

    n_tokens = sum(len(rows) for rows in rows_per_sentence)
    if verbose:
        print(
            f"Training the transition parser ({algorithm}) on "
            f"{len(rows_per_sentence)} sentences / {n_tokens} tokens "
            f"from {train_path.name} ..."
        )

    depgraphs = [depgraph_from_rows(rows) for rows in rows_per_sentence]

    parser = _LinearTransitionParser(algorithm)
    n_projective = sum(1 for graph in depgraphs if parser._is_projective(graph))
    parser.train(depgraphs, str(classifier_path), verbose=verbose)

    train_meta = {
        "train_file": train_path.name,
        "train_sha256": _sha256_prefix(train_path),
        "n_sentences": len(rows_per_sentence),
        "n_tokens": n_tokens,
        # The oracle drops non-projective sentences silently (it only prints a
        # tally and keeps it nowhere). "Trained on 12,544 sentences" and
        # "trained on 12,266 sentences" are different claims, and the results
        # table reports the second one.
        "n_projective": n_projective,
        "max_train_sentences": max_train_sentences,
        "train_seconds": round(time.perf_counter() - started, 1),
    }
    train_meta.update(parser._train_stats)

    with open(pickle_path, "wb") as handle:
        pickle.dump(
            {"parser": parser, "meta": train_meta},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    if verbose:
        print(
            f"Trained in {train_meta['train_seconds']}s "
            f"({train_meta['n_transitions']} transitions, "
            f"{train_meta['n_features']} features) -> "
            f"{pickle_path} "
            f"({pickle_path.stat().st_size / 1e6:.1f} MB)"
        )


# --- small helpers ------------------------------------------------------------


def _sha256_prefix(path: Path, length: int = 12) -> str:
    """Short SHA-256 of a file, for provenance rather than for verification.

    The digest itself is `checksum.sha256_of`, which this used to re-implement
    chunk for chunk. That module sits at the top level of the package precisely
    so a wrapper can use it without importing anything from evaluation/.
    """
    return sha256_of(path)[:length]


def _files_digest(*paths: Path) -> str:
    """One SHA-256 of the bytes of several files, read in the order given."""
    digest = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _nltk_version() -> str:
    """NLTK's version string, or "?" if it cannot be determined."""
    import nltk

    return getattr(nltk, "__version__", "?")
