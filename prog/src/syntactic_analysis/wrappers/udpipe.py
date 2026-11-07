"""UDPipe 1 — POS tagging (perceptron) and dependency parsing (shallow neural).

UDPipe is a trainable pipeline developed at ÚFAL (Charles University, Prague)
for tokenization, POS tagging, lemmatization, and dependency parsing. Models
are trained on Universal Dependencies treebanks.

How it works — and why this wrapper sits at TWO method levels:
  1. POS tagging uses MorphoDiTa: a dictionary + suffix guesser feeding an
     averaged perceptron over discrete feature templates. MethodLevel L1
     (STATISTICAL) for POS_TAGGING.
  2. Dependency parsing uses Parsito, which the UDPipe 1 user manual describes as
     "a transition-based parser using a neural-network classifier": form / tag /
     deprel embeddings into a single hidden layer (default 200 units), no
     recurrent or pretrained contextual encoder. MethodLevel L2 (NEURAL_SHALLOW)
     for DEPENDENCY_PARSING. Earlier versions of this project — and of the thesis
     text — called the parser an averaged perceptron; that was wrong, and the
     per-component `method_level` below is what corrects it.
  3. All output follows Universal Dependencies conventions natively
  4. The model is trained here, by scripts/train_udpipe.py on the current EWT
     train split (train_from_conllu), and only there: since 24 Oct 2026
     load_model refuses a missing model (base.require_trained) and nothing
     trains it during a build or a run. Until 23 Oct 2026 this wrapper also
     loaded the published UDPipe 1 model trained on UD_English-EWT r2.5
     ("en", downloaded from a third-party mirror of the LINDAT release);
     version 1.0 does not use it. The file may still be in model_cache/; no
     code reads it.

Key characteristics:
  - Supports POS tagging and dependency parsing, but NOT constituency parsing
  - Perceptron tagger (L1) + feed-forward neural transition parser (L2)
  - Native UD output — no tag conversion needed (unlike NLTK)
  - Trained on the same EWT treebank used for evaluation (but train split,
    not test split) — this is standard practice in UD benchmarking
  - Lightweight models (~16 MB), fast inference

Install: pip install ufal.udpipe
Model:   <project_root>/model_cache/english-ewt-retrained.udpipe, written by
         scripts/train_udpipe.py before the lock (until 24 Oct 2026 also by
         the first stage of run_pipeline.py, train_udpipe, which is gone).
"""

from __future__ import annotations

from pathlib import Path

from syntactic_analysis.base import (
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
)
from syntactic_analysis.project import model_cache_dir

# Where the retrained model is written and read: scripts/train_udpipe.py writes
# it here by default, and the "en_ewt_retrained" entry below loads it from here.
# Until 12 Oct 2026 the script built the same path a second time, under a
# comment saying the two had to match.
RETRAINED_MODEL_PATH = model_cache_dir() / "english-ewt-retrained.udpipe"

# The script that writes it, which a missing model's error names
# (TrainedModel.script).
TRAINING_SCRIPT = "scripts/train_udpipe.py"


# The UDPipe models this wrapper loads. An entry's "path" points at a pre-built
# local file (produced by train_from_conllu()), loaded as-is and never
# downloaded. The published UD 2.5 model ("en", downloaded and SHA-256 verified)
# was removed with version 1.0.
_MODELS = {
    "en_ewt_retrained": {
        # Not downloaded — produced locally by UdpipeAnalyzer.train_from_conllu()
        # (see its docstring for why: isolating annotation-guideline drift from
        # architecture when comparing against Stanza). No checksum: whatever file
        # is on disk here IS the model, there is nothing external to verify it against.
        "path": RETRAINED_MODEL_PATH,
    },
}

# The key a model gets when the caller names none.
_DEFAULT_KEYS = {"en_ewt_retrained": "udpipe_retrained"}

# The seed of a UDPipe 1 training, which UDPipe fixes in its own code: it has
# no seed option. Its parser training (Parsito) seeds its random generator
# with a constant - `mt19937 generator(42)` under the comment "Random
# generator with fixed seed for reproducibility" in parser_nn_trainer::train
# (parsito/parser/parser_nn_trainer.cpp) - and draws from it the initial
# embeddings and weights, the order of the training sentences in every
# iteration and the sentences of the search-based oracle. The tagger's
# training (MorphoDiTa's perceptron_tagger_trainer) draws no random number,
# and the tokenizer, whose training does, is not trained here (tokenizer
# "none"). The `run` option, left at its default, changes only the
# hyperparameters of a random search (UDPipe 1 User's Manual, section 3.2).
# Read on 24 Oct 2026 in the sources of the installed version, ufal.udpipe
# 1.4.0.1 (ufal/udpipe/udpipe.cpp of its sdist on PyPI); the compiled
# extension does not expose the value, so it is written here, with the
# version it was read in.
PARSITO_TRAINING_SEED = 42
PARSITO_TRAINING_SEED_READ_IN = "1.4.0.1"

# The two components of a UDPipe 1 model, as an architecture cell names them.
_TAGGER_DETAIL = "MorphoDiTa perceptron tagger"
_PARSER_DETAIL = "Parsito FFNN transition parser"


class UdpipeAnalyzer(BaseAnalyzer):

    def __init__(self, *, model: str = "en_ewt_retrained", key: str | None = None) -> None:
        """Create the analyzer.

        Args:
            model: Which entry of _MODELS load_model() loads unless it is told
                 otherwise: "en_ewt_retrained" (default and only entry), the
                 model scripts/train_udpipe.py refits on the current EWT
                 training split.
            key: Key and CSV column name, overriding the one the
                 model gives it ("udpipe_retrained").

        Raises:
            ValueError: If `model` is not a key of _MODELS.
        """
        if model not in _MODELS:
            raise ValueError(
                f"unknown model {model!r}; expected one of {sorted(_MODELS)}"
            )
        self._key = key or _DEFAULT_KEYS[model]
        self._model = None      # loaded UDPipe Model object
        self._model_name = model  # the entry declared; load_model() may load another
        self._model_version = ""  # set by load_model(); see model_version

    @property
    def name(self) -> str:
        return self._key

    @property
    def supported_types(self) -> set[AnalysisType]:
        return {AnalysisType.POS_TAGGING, AnalysisType.DEPENDENCY_PARSING}

    @property
    def method_type(self) -> MethodType:
        # Coarse label for the tool as a whole. The parser is the component that
        # defines UDPipe's place in the dependency stage, and it is neural (see module
        # docstring); the per-task truth is in `method_level`.
        return MethodType.NEURAL

    @property
    def method_detail(self) -> str:
        return f"{_TAGGER_DETAIL} / {_PARSER_DETAIL}"

    def method_detail_for(self, analysis_type: AnalysisType) -> str:
        """The tagger for POS, the parser for dependencies: one model file holds
        both, and each task is scored on one of them."""
        if analysis_type is AnalysisType.DEPENDENCY_PARSING:
            return _PARSER_DETAIL
        if analysis_type is AnalysisType.POS_TAGGING:
            return _TAGGER_DETAIL
        return self.method_detail

    @property
    def display_name(self) -> str:
        return "UDPipe (retrained)"

    @property
    def method_level(self) -> dict:
        return {
            AnalysisType.POS_TAGGING: MethodLevel.STATISTICAL,
            AnalysisType.DEPENDENCY_PARSING: MethodLevel.NEURAL_SHALLOW,
        }

    @property
    def annotation_scheme(self) -> dict:
        return {
            AnalysisType.POS_TAGGING: AnnotationScheme.UD,
            AnalysisType.DEPENDENCY_PARSING: AnnotationScheme.UD,
        }

    @property
    def training_domain(self) -> dict:
        # Both shipped and retrained models are trained on the EWT train split.
        return {
            AnalysisType.POS_TAGGING: TrainingDomain.IN,
            AnalysisType.DEPENDENCY_PARSING: TrainingDomain.IN,
        }

    @property
    def training_corpus(self) -> str:
        # Retrained on the current split rather than the published UD 2.5
        # model: see train_from_conllu() for why.
        return "UD_English-EWT train (current split, retrained locally)"

    @property
    def model_version(self) -> str:
        """The loaded model file and the first twelve hex digits of its SHA-256.

        The digest is of the file actually loaded, computed at load: the
        retrained model has no recorded digest to check it against. Empty
        until load_model() has run.
        """
        return self._model_version

    def training_seed_for(self, analysis_type: AnalysisType) -> TrainingSeed | None:
        """PARSITO_TRAINING_SEED for the loaded model; None before load_model().

        The one model this wrapper loads is the file train_from_conllu writes
        (scripts/train_udpipe.py), trained with UDPipe's default options. One
        training writes the tagger and the parser into that file, so both
        tasks state the seed of that training, and the note says which
        component it seeds.
        """
        if self._model is None or analysis_type not in self.supported_types:
            return None
        return TrainingSeed(
            PARSITO_TRAINING_SEED,
            f"fixed in UDPipe 1, which has no seed option: its parser training "
            f"(Parsito, parser_nn_trainer::train) seeds its generator with "
            f"{PARSITO_TRAINING_SEED}, and its tagger training draws no random number "
            f"(sources of ufal.udpipe {PARSITO_TRAINING_SEED_READ_IN}); trained by "
            f"scripts/train_udpipe.py with the library's default options (no tokenizer, "
            f"default tagger and parser)",
        )

    def trained_model(self, model_name: str | None = None, **load) -> TrainedModel:
        """The model file scripts/train_udpipe.py writes, which load_model reads.

        Takes load_model's keywords; only model_name says where the model is.
        Never None: every model of this wrapper is trained here (one since
        version 1.0). Since 24 Oct 2026.

        Raises:
            ValueError: If model_name is not a key of _MODELS.
        """
        name = self._model_name if model_name is None else model_name
        if name not in _MODELS:
            raise ValueError(
                f"unknown model_name {name!r}; expected one of {sorted(_MODELS)}"
            )
        return TrainedModel(TRAINING_SCRIPT, files=(Path(_MODELS[name]["path"]),))

    def load_model(
        self,
        model_name: str | None = None,
        **kwargs,
    ) -> None:
        """Load a UDPipe model.

        Args:
            model_name: Key of _MODELS. None (default) loads the model the
                        constructor declared. "en_ewt_retrained", a model
                        produced by train_from_conllu(), is loaded straight
                        from disk, never downloaded.

        The model is never trained here: a missing file is refused, naming
        scripts/train_udpipe.py, which trains it before the lock (since
        24 Oct 2026; until then run_pipeline.py trained it in its first stage,
        train_udpipe). Until 23 Oct 2026 this also downloaded the published
        UD 2.5 model on first use, into a `cache_dir` it took as an argument.

        Raises:
            TypeError:  If an unknown keyword argument is passed.
            ValueError: If model_name is not available.
            base.ModelNotTrainedError: If the model file is missing.
            RuntimeError: If the model file fails to load.
        """
        self._reject_unknown_options(kwargs)
        from ufal.udpipe import Model

        if model_name is None:
            model_name = self._model_name
        # Locally produced model (train_from_conllu) - nothing to fetch or
        # verify, the file on disk simply is the model; a missing one is
        # refused with the script that writes it.
        trained = self.trained_model(model_name)
        require_trained(self.name, trained)
        (model_path,) = trained.files

        self._model = Model.load(str(model_path))
        if not self._model:
            raise RuntimeError(f"Failed to load UDPipe model from {model_path}")

        self._model_name = model_name  # remember which config is loaded
        # The model is named by its own file name.
        self._model_version = f"{model_path.name}@{sha256_of(model_path)[:12]}"

    def _analyze_dependency(self, text: str | list[list[str]]) -> DependencyParse:
        """Run UDPipe pipeline and extract POS tags and dependency information.

        Args:
            text: Either a raw string (UDPipe tokenizes automatically) or
                  a list of pre-tokenized sentences for fair evaluation.

                  Example raw:          "The cat sat."
                  Example pretokenized: [["The", "cat", "sat", "."]]

        Returns a DependencyParse with Universal POS tags, dependency heads,
        and dependency relation labels in UD format.
        """
        self._ensure_loaded()
        from ufal.udpipe import Pipeline, ProcessingError

        if isinstance(text, list):
            # convert pretokenized input to CoNLL-U format (tokens only)
            conllu_input = self._pretokenized_to_conllu(text)
            input_format = "conllu"
        else:
            conllu_input = text
            input_format = "tokenize"

        # run the full pipeline: tagger + parser
        pipeline = Pipeline(
            self._model, input_format,
            Pipeline.DEFAULT,  # tagger
            Pipeline.DEFAULT,  # parser
            "conllu",          # output format
        )
        err = ProcessingError()
        result = pipeline.process(conllu_input, err)

        if err.occurred():
            raise RuntimeError(f"UDPipe error: {err.message}")

        return self._parse_conllu_output(result)

    def _analyze_constituency(self, text: str | list[list[str]]) -> ConstituencyParse:
        """Not supported — UDPipe is a dependency-only parser.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError(
            "UDPipe does not support constituency parsing."
        )

    @staticmethod
    def _pretokenized_to_conllu(sentences: list[list[str]]) -> str:
        """Convert pre-tokenized sentences to CoNLL-U format for UDPipe input.

        Each token gets a line with ID and FORM filled, rest as placeholders.
        Sentences are separated by blank lines.
        """
        lines = []
        for words in sentences:
            for i, word in enumerate(words):
                # CoNLL-U: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
                lines.append(f"{i+1}\t{word}\t_\t_\t_\t_\t_\t_\t_\t_")
            lines.append("")  # blank line = sentence boundary
        return "\n".join(lines)

    def _parse_conllu_output(self, conllu_text: str) -> DependencyParse:
        """Parse UDPipe's CoNLL-U output into a DependencyParse object.

        Instance method (not static) so the DependencyParse is tagged with
        `self.name` rather than a hardcoded "udpipe" — a subclass, or a second
        UdpipeAnalyzer configured with a different model_name, must be labelled
        with its own name in results tables, not silently attributed to this one.

        The reading is `data_loader.parse_conllu` in its `predicted` mode, which
        is a tool's output rather than an annotated file: every comment line is
        ignored, so a sentence's text is its forms joined by one space, and a
        LEMMA, UPOS or DEPREL the tool left as "_" comes back empty.
        """
        return parse_conllu(conllu_text, analyzer_name=self.name, predicted=True)

    def _ensure_loaded(self) -> None:
        """Guard against calling analyze methods before model initialization."""
        self._require_loaded(self._model is not None)

    # --- training (sensitivity experiment) ---------------------------------

    @staticmethod
    def _conllu_to_sentences(conllu_text: str):
        """Parse CoNLL-U text into a ufal.udpipe `Sentences` vector for training.

        Uses UDPipe's own CoNLL-U reader (InputFormat.newConlluInputFormat())
        rather than a hand-rolled parser, so training input is subject to the
        exact same format handling (comments, multiword tokens, ...) as
        evaluation-time input.

        Returns a `ufal.udpipe.Sentences` (a std::vector<sentence> wrapper),
        which is what `Trainer.train()` expects for its train/heldout arguments.

        Raises:
            RuntimeError: If UDPipe's CoNLL-U reader reports a parse error.
        """
        from ufal.udpipe import InputFormat, ProcessingError, Sentence, Sentences

        input_format = InputFormat.newConlluInputFormat()
        input_format.setText(conllu_text)

        sentences = Sentences()
        sentence = Sentence()
        error = ProcessingError()
        while input_format.nextSentence(sentence, error):
            sentences.push_back(sentence)
            sentence = Sentence()  # nextSentence fills the object in place; a fresh one is needed per iteration
        if error.occurred():
            raise RuntimeError(f"Failed to parse CoNLL-U input: {error.message}")

        return sentences

    @classmethod
    def train_from_conllu(
        cls,
        train_path: str | Path,
        out_path: str | Path,
        *,
        heldout_path: str | Path | None = None,
        **opts,
    ) -> Path:
        """Train a UDPipe 1 model from a CoNLL-U file and write it to `out_path`.

        Why this exists: the only published UDPipe 1 English model is
        morphodita_parsito trained on UD_English-EWT r2.5 (2019-12-06). The gold
        standard used elsewhere in this project is a much newer UD release, so
        any accuracy gap measured against that model conflates two different
        things: UDPipe 1's architecture (a shallow feed-forward
        transition parser) falling behind a BiLSTM parser like Stanza, AND the UD annotation
        guidelines having drifted between 2.5 and the current release (attachment
        rules for auxiliaries, copulas, "goeswith", flat structures, etc. have all
        changed across UD versions). Retraining UDPipe 1 on the *current* train
        split — using ufal.udpipe.Trainer.train() directly, the same library the
        published model was built with — removes the guideline-drift confound and
        isolates the architecture comparison. Written to RETRAINED_MODEL_PATH,
        which is scripts/train_udpipe.py's default --out and the "path" of the
        "en_ewt_retrained" entry in _MODELS, the file loads through the normal
        load_model() path as UdpipeAnalyzer(model="en_ewt_retrained"), with
        nothing to register; a model written anywhere else loads only once that
        entry points at it.

        Verified against the installed ufal.udpipe 1.4.0.1: `Trainer.train()`'s
        real signature (recovered from the SWIG overload-resolution error message,
        since the compiled extension carries no Python docstrings) is

            Trainer.train(method: str, train: Sentences, heldout: Sentences,
                           tokenizer: str, tagger: str, parser: str,
                           error: ProcessingError) -> bytes

        `Trainer.NONE` is the string "none" and `Trainer.DEFAULT` is "" (use
        UDPipe's built-in default configuration for that component). Passing
        `Trainer.NONE` for `tokenizer` skips training a tokenizer component
        entirely — appropriate here because both training and evaluation always
        supply pre-tokenized CoNLL-U, so a tokenizer is never exercised.
        There is no seed to pass: UDPipe fixes the one its parser training
        draws from (PARSITO_TRAINING_SEED), and the stage's provenance records
        it (`training_seed_for`).

        Args:
            train_path:   Path to a CoNLL-U file with training sentences (forms,
                          lemmas, UPOS, dependency heads/labels).
            out_path:      Where to write the trained model (UDPipe's binary
                          format, produced by Trainer.train() as raw bytes).
            heldout_path:  Optional CoNLL-U file used for early stopping. If
                          omitted, training runs against an empty heldout set
                          (a legal, if suboptimal, configuration).
            **opts:        Optional overrides, all passed straight to
                          Trainer.train():
                            method    (str) default "morphodita_parsito"
                            tokenizer (str) default Trainer.NONE  ("none" —
                                            input is already tokenized)
                            tagger    (str) default Trainer.DEFAULT ("")
                            parser    (str) default Trainer.DEFAULT ("")

        Returns the resolved Path of the written model file.

        Raises:
            RuntimeError: If CoNLL-U parsing or training reports an error
                          (surfaced via ufal.udpipe.ProcessingError, per the
                          library's own error-reporting convention).
            TypeError:    If an unknown **opts key is passed.
        """
        from ufal.udpipe import ProcessingError, Sentences, Trainer

        train_path = Path(train_path)
        out_path = Path(out_path)

        train_sentences = cls._conllu_to_sentences(train_path.read_text(encoding="utf-8"))
        if heldout_path is not None:
            heldout_sentences = cls._conllu_to_sentences(
                Path(heldout_path).read_text(encoding="utf-8")
            )
        else:
            heldout_sentences = Sentences()  # empty heldout is legal; just no early stopping

        method = opts.pop("method", "morphodita_parsito")
        tokenizer = opts.pop("tokenizer", Trainer.NONE)
        tagger = opts.pop("tagger", Trainer.DEFAULT)
        parser = opts.pop("parser", Trainer.DEFAULT)
        if opts:
            raise TypeError(f"Unknown train_from_conllu options: {sorted(opts)}")

        error = ProcessingError()
        model_bytes = Trainer.train(
            method, train_sentences, heldout_sentences, tokenizer, tagger, parser, error
        )
        if error.occurred():
            raise RuntimeError(f"UDPipe training failed: {error.message}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(model_bytes)
        return out_path.resolve()
