"""spaCy — neural dependency parsing and POS tagging.

spaCy is a fast, production-ready NLP framework developed by Explosion.
It uses neural models for tokenization, POS tagging, and dependency parsing.

How it works:
  1. A pre-trained model is loaded via spacy.load() — the model defines
     the full NLP pipeline (tokenizer, tagger, parser, NER, etc.)
  2. Raw text is processed through the pipeline, producing a Doc object
  3. Each token in the Doc has POS tags, dependency heads, lemmas, etc.
  4. For pre-tokenized input, a Doc is created with preset word boundaries
     and all pipeline components except the tokenizer are run manually

Key characteristics:
  - Supports POS tagging and dependency parsing, but NOT constituency parsing
  - For constituency parsing, use BeneparAnalyzer (adds constituency on top of spaCy)
  - Two architecture families: CNN (sm/md/lg models) and Transformer (_trf models)
  - Models must be downloaded manually: python -m spacy download <model_name>
  - Available English models:
      en_core_web_sm   — small (~12 MB), CNN architecture
      en_core_web_md   — medium (~40 MB), CNN architecture
      en_core_web_lg   — large (~560 MB), CNN architecture
      en_core_web_trf  — transformer (~460 MB), RoBERTa architecture

Install: pip install spacy

Two models, declared, not guessed
---------------------------------
The constructor's `model` says which pipeline an instance runs, and every
declaration follows from it (`_MODELS`):

  "en_core_web_sm" (default)  the published pipeline, trained on OntoNotes 5
                              with dependencies in the ClearNLP conversion: SD
                              for dependencies, domain OUT, and any of the three
                              SD->UD mapping tiers (default "full").
  "ewt_train"                 the pipeline scripts/train_spacy.py trains on the
                              EWT train split (model_cache/spacy_ewt/model-best):
                              UD for dependencies, domain IN, and no SD->UD
                              mapping. Its one tier is "casing": spaCy's CoNLL-U
                              converter writes the UD relation root as ROOT
                              when it reads the training file, so the pipeline
                              emits ROOT, and "casing" (ROOT -> root, nothing
                              else) gives the gold's label back.

Nothing is read off the model's name. load_model() checks the pipeline it
loaded against the declaration instead: a published pipeline must name
OntoNotes 5 among the sources in its meta.json, and the EWT pipeline must be
the one its training record describes (the digest of model-best/).

The EWT pipeline is the one model of this wrapper the project trains:
`trained_model` declares model-best/, every file under it, as its files and
scripts/train_spacy.py as the script that writes them (since 24 Oct 2026; a
missing pipeline was refused already, and nothing ever trained it at load).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from syntactic_analysis.base import (
    AnalysisType,
    AnnotationScheme,
    BaseAnalyzer,
    MethodType,
    TrainedModel,
    TrainingDomain,
    require_trained,
)
from syntactic_analysis.checksum import sha256_of
from syntactic_analysis.deprel_map import (
    HEADLINE_TIER,
    MAPPING_TIERS,
    MapTier,
    validate_tier,
)
from syntactic_analysis.models import (
    ConstituencyParse,
    DependencyParse,
    Sentence,
)
from syntactic_analysis.downloads import DownloadedFiles
from syntactic_analysis.project import model_cache_dir
from syntactic_analysis.spacy_doc import (
    PIPELINE_CORPUS,
    doc_to_depparse,
    doc_tokens,
    is_transformer_pipeline,
    library_files,
    package_files,
    pipeline_level,
    pipeline_name,
)

# Where scripts/train_spacy.py works and what it leaves there: the converted
# corpus, the config, spaCy's model-best/ and model-last/, and the record of
# the training. Under model_cache/, which git ignores.
EWT_WORK_DIR = model_cache_dir() / "spacy_ewt"
EWT_PIPELINE_DIR = EWT_WORK_DIR / "model-best"
EWT_TRAINING_RECORD = EWT_WORK_DIR / "training_record.json"

# The script that writes them, which a missing pipeline's error names
# (TrainedModel.script).
TRAINING_SCRIPT = "scripts/train_spacy.py"

# The training corpus of the EWT pipeline, as a results table prints it: named
# the way the other in-domain rows name it ("current split" is the file on
# disk, which no UD release tag identifies; data/DATASET.md).
EWT_CORPUS = "UD_English-EWT train (current split, trained locally)"

# The source a published English pipeline names in its meta.json, and the one
# the declarations of "en_core_web_sm" are true for.
_ONTONOTES_SOURCE = "OntoNotes 5"


@dataclass(frozen=True)
class _Model:
    """What one model of this wrapper is, declared once.

    Attributes:
        source:            What spacy.load() is given: a package name or a directory.
        dependency_scheme: The scheme of its relations (its UPOS is UD either way).
        domain:            Its training domain, the same for both tasks.
        corpus:            Its training corpus, as `training_corpus` reports it.
        tiers:             The SD->UD mapping tiers its relation labels admit.
        default_tier:      The tier an instance uses unless it is given one.
        default_key:       The key an instance gets unless it is given one.
        display_name:      The label of its row.
    """

    source: str | Path
    dependency_scheme: AnnotationScheme
    domain: TrainingDomain
    corpus: str
    tiers: tuple[MapTier, ...]
    default_tier: MapTier
    default_key: str
    display_name: str


_MODELS: dict[str, _Model] = {
    "en_core_web_sm": _Model(
        source="en_core_web_sm",
        # POS is UPOS, but the parser emits ClearNLP/OntoNotes labels and
        # structure - a Stanford Dependencies descendant, not UD.
        dependency_scheme=AnnotationScheme.SD,
        # OntoNotes 5, not the EWT: out-of-domain for both tasks.
        domain=TrainingDomain.OUT,
        # Declared once in spacy_doc, where the Benepar wrapper reads it too.
        corpus=PIPELINE_CORPUS,
        tiers=MAPPING_TIERS,
        default_tier=HEADLINE_TIER,
        default_key="spacy",
        display_name="spaCy",
    ),
    "ewt_train": _Model(
        source=EWT_PIPELINE_DIR,
        dependency_scheme=AnnotationScheme.UD,
        domain=TrainingDomain.IN,
        corpus=EWT_CORPUS,
        # See the module docstring: ROOT -> root and no SD->UD relabelling.
        tiers=("casing",),
        default_tier="casing",
        default_key="spacy_ewt",
        display_name="spaCy (EWT)",
    ),
}


def pipeline_digest(directory: str | Path) -> str:
    """SHA-256 of a pipeline directory: each file's relative path and digest, sorted.

    What identifies a pipeline spaCy saved as a directory, where a model of
    UDPipe is one file and is named by that file's digest.
    """
    directory = Path(directory)
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8") + b"\0")
        digest.update(sha256_of(path).encode("ascii") + b"\n")
    return digest.hexdigest()


class SpacyAnalyzer(BaseAnalyzer):

    def __init__(
        self,
        *,
        map_tier: MapTier | None = None,
        model: str = "en_core_web_sm",
        key: str | None = None,
    ) -> None:
        """Create the analyzer.

        Args:
            map_tier:  "none" | "casing" | "full"; None for the model's own
                       default ("full" for "en_core_web_sm", "casing" for
                       "ewt_train"). The tiers can isolate the casing-only step,
                       so the mapping gain is not credited to the linguistic
                       conversion when it is a string-case fix. Until 24 Oct
                       2026 a legacy switch, map_to_ud (True for "full", False
                       for "none"), could stand in for it; every value it had is
                       a tier, nothing passed it, and it was removed.
            model:     Which entry of _MODELS this instance runs (see the
                       module docstring): "en_core_web_sm" or "ewt_train".
            key:       Key and CSV column name, overriding the one
                       the model gives it ("spacy", "spacy_ewt").

        Raises:
            ValueError: If `model` is not a key of _MODELS, or the tier is not
                one the model's relation labels admit.
        """
        if model not in _MODELS:
            raise ValueError(
                f"unknown model {model!r}; expected one of {sorted(_MODELS)}"
            )
        entry = _MODELS[model]
        if map_tier is None:
            map_tier = entry.default_tier
        map_tier = validate_tier(map_tier)
        if map_tier not in entry.tiers:
            raise ValueError(
                f"map_tier {map_tier!r} does not apply to model {model!r}; expected one "
                f"of {list(entry.tiers)}. Its relations are UD already, and 'casing' "
                f"only turns spaCy's ROOT back into root."
            )
        self._model_id = model
        self._model = entry
        self._map_tier = map_tier
        self._key = key or entry.default_key
        self._nlp = None  # set by load_model(); also doubles as the "is it loaded" flag
        self._model_version = ""  # set by load_model(); see model_version

    @property
    def name(self) -> str:
        return self._key

    @property
    def supported_types(self) -> set[AnalysisType]:
        # No constituency parsing — use BeneparAnalyzer for that
        return {AnalysisType.POS_TAGGING, AnalysisType.DEPENDENCY_PARSING}

    @property
    def method_type(self) -> MethodType:
        return MethodType.NEURAL

    @property
    def method_detail(self) -> str:
        # The architecture of the loaded pipeline and no other: "CNN" for
        # en_core_web_sm and the EWT pipeline (spacy_sm, spacy_ewt),
        # "Transformer" for a transformer pipeline. Detected by inspecting the
        # loaded pipeline's component names rather than pattern-matching the model
        # name string (e.g. "_trf" in model_name) — a locally-pathed or renamed
        # transformer model would silently misreport under that heuristic.
        # Until 24 Oct 2026 it named both families with the active one in
        # brackets ("[CNN] / Transformer"), which a provenance record and an
        # Architecture cell read as two components, as "/" means for UDPipe.
        if self._nlp is None:
            return "CNN or Transformer (not loaded)"
        return "Transformer" if is_transformer_pipeline(self._nlp) else "CNN"

    @property
    def display_name(self) -> str:
        return self._model.display_name

    @property
    def annotation_scheme(self) -> dict:
        # UPOS for every model; the relations follow the model (_MODELS).
        return {
            AnalysisType.POS_TAGGING: AnnotationScheme.UD,
            AnalysisType.DEPENDENCY_PARSING: self._model.dependency_scheme,
        }

    @property
    def method_level(self) -> dict:
        # en_core_web_sm: MultiHashEmbed + CNN tok2vec feeding a transition-based
        # parser and a tagger; no pretrained vectors, no recurrent encoder. Both
        # components share that tok2vec, so both are L2. The EWT pipeline is the
        # same kind (`spacy init config --optimize efficiency`: MultiHashEmbed +
        # CNN, no static vectors) with a morphologizer for the tagger: L2 too. A
        # transformer pipeline (en_core_web_trf) would be L4 — detected from
        # pipe_names, not the name, by the rule spacy_doc.pipeline_level states
        # for both spaCy wrappers.
        level = pipeline_level(self._nlp)
        return {
            AnalysisType.POS_TAGGING: level,
            AnalysisType.DEPENDENCY_PARSING: level,
        }

    @property
    def training_domain(self) -> dict:
        # OntoNotes 5 (OUT) or the EWT train split (IN), the same for both tasks.
        return {
            AnalysisType.POS_TAGGING: self._model.domain,
            AnalysisType.DEPENDENCY_PARSING: self._model.domain,
        }

    @property
    def training_corpus(self) -> str:
        return self._model.corpus

    @property
    def model_version(self) -> str:
        """Concrete loaded-model identity, for provenance logging.

        A published pipeline is named as its own meta.json names it ("lang",
        "name" and "version"), which is the authoritative source spaCy itself
        ships alongside the model: `en_core_web_sm-3.8.0`, as the Benepar
        wrapper names the same pipeline. Until 16 Oct 2026 the language was
        left out, and the version read `core_web_sm-3.8.0`.

        The EWT pipeline's meta.json names nothing of its own (spaCy writes
        `pipeline-0.0.0` for every pipeline it trains), so it is named by its
        directory and the first twelve hex digits of `pipeline_digest`, as
        UDPipe's retrained model is named by its file and digest:
        `spacy_ewt/model-best@<12 hex>`.

        Returns:
            Empty string if load_model() has not been called yet.
        """
        return self._model_version

    def trained_model(self, model_name: str | None = None, **load) -> TrainedModel | None:
        """The EWT pipeline, which load_model reads; None for a published pipeline.

        Takes load_model's keywords, none of which says where the EWT pipeline
        is: it is model_cache/spacy_ewt/model-best/, read as a whole by
        spacy.load, so every file under it is a file of the model. A published
        pipeline is an installed package. Since 24 Oct 2026.
        """
        if self._model_id != "ewt_train":
            return None
        return TrainedModel(TRAINING_SCRIPT, directories=(EWT_PIPELINE_DIR,))

    def downloaded_files(self, model_name: str | None = None, **load) -> tuple[DownloadedFiles, ...]:
        """The installed pipeline package load_model reads, and the data files of spaCy's import.

        Takes load_model's keywords. A published pipeline is an installed
        package, and spacy.load reads its meta.json and its data directory
        (spacy_doc.package_files), under site_packages. The EWT pipeline is
        trained here (trained_model), so for it only the rest is declared:
        the data files `import spacy` reads from spaCy's and thinc's own
        package directories (spacy_doc.library_files; since 25 Oct 2026,
        until then () for the EWT pipeline). A package that is not installed
        is declared all the same, so a lock check reports it missing. Since
        24 Oct 2026.
        """
        if self._model_id == "ewt_train":
            return library_files()
        return (package_files(model_name or self._model.source), *library_files())

    def load_model(self, model_name: str | None = None, **kwargs) -> None:
        """Load the declared model, and check that it is what it is declared to be.

        Args:
            model_name: None (default) loads the model the constructor declared.
                        With "en_core_web_sm" another installed English pipeline
                        may be named (python -m spacy download <model_name>);
                        it must name OntoNotes 5 among its sources, which is
                        what the declarations of that model say. The EWT
                        pipeline is loaded from its directory only, so it takes
                        no model_name.
            **kwargs:   Passed to spacy.load().

        Raises:
            OSError:      If the model is not installed.
            ValueError:   If a model_name is given for the EWT pipeline.
            base.ModelNotTrainedError: If the EWT pipeline is missing
                          (trained_model, require_trained).
            RuntimeError: If the EWT pipeline's training record is missing,
                          if model-best/ is not the pipeline the record
                          describes, or if a published pipeline does not name
                          OntoNotes 5 among its sources.
        """
        import spacy  # deferred import so spacy is only needed at runtime

        if self._model_id == "ewt_train":
            if model_name is not None:
                raise ValueError(
                    f"model 'ewt_train' is loaded from {EWT_PIPELINE_DIR} and takes no "
                    f"model_name; got {model_name!r}"
                )
            require_trained(self.name, self.trained_model())
            recorded_digest = self._ewt_recorded_digest()
            nlp = spacy.load(EWT_PIPELINE_DIR, **kwargs)
            digest = pipeline_digest(EWT_PIPELINE_DIR)
            if digest != recorded_digest:
                raise RuntimeError(
                    f"{EWT_PIPELINE_DIR} is not the pipeline {EWT_TRAINING_RECORD.name} "
                    f"describes (digest {digest[:12]}, recorded {recorded_digest[:12]}). "
                    f"Train it again with scripts/train_spacy.py --force."
                )
            version = f"{EWT_WORK_DIR.name}/{EWT_PIPELINE_DIR.name}@{digest[:12]}"
        else:
            nlp = spacy.load(model_name or self._model.source, **kwargs)
            sources = [source.get("name") for source in nlp.meta.get("sources", [])]
            if _ONTONOTES_SOURCE not in sources:
                raise RuntimeError(
                    f"pipeline {pipeline_name(nlp)} does not name {_ONTONOTES_SOURCE} among "
                    f"its sources ({sources}); model 'en_core_web_sm' declares a pipeline "
                    f"trained on it"
                )
            version = pipeline_name(nlp)
        self._nlp = nlp  # loads full pipeline from disk
        self._model_version = version

    @staticmethod
    def _ewt_recorded_digest() -> str:
        """The digest of model-best/ that scripts/train_spacy.py recorded.

        Raises:
            RuntimeError: If the pipeline or its training record is missing.
        """
        if not EWT_PIPELINE_DIR.is_dir():
            raise RuntimeError(
                f"EWT pipeline not found: {EWT_PIPELINE_DIR}. Train it first with "
                f"scripts/train_spacy.py."
            )
        if not EWT_TRAINING_RECORD.is_file():
            raise RuntimeError(
                f"{EWT_PIPELINE_DIR} has no training record ({EWT_TRAINING_RECORD}); "
                f"it was not trained by scripts/train_spacy.py. Train it again with --force."
            )
        record = json.loads(EWT_TRAINING_RECORD.read_text(encoding="utf-8"))
        return record.get("pipeline_digest", "")

    def _analyze_dependency(self, text: str | list[list[str]]) -> DependencyParse:
        """Run the spaCy pipeline and extract POS tags and dependency information.

        Args:
            text: Either a raw string (spaCy tokenizes automatically) or
                  a list of pre-tokenized sentences for fair evaluation.

                  Example raw:          "The cat sat."
                  Example pretokenized: [["The", "cat", "sat", "."]]

        Returns a DependencyParse with Universal POS tags (pos_), language-specific
        POS tags (tag_), lemmas, dependency heads, and dependency relation labels.
        """
        self._ensure_loaded()

        # Pre-tokenized input needs special handling to skip spaCy's tokenizer
        if isinstance(text, list):
            return self._analyze_pretokenized(text)

        # Raw string: let spaCy handle tokenization + full pipeline
        doc = self._nlp(text)
        return self._doc_to_depparse(doc)

    def _analyze_constituency(self, text: str | list[list[str]]) -> ConstituencyParse:
        """Not supported — spaCy does not provide constituency parsing.

        Use BeneparAnalyzer instead (adds constituency on top of spaCy).

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError(
            "spaCy does not support constituency parsing natively. "
            "Use BeneparAnalyzer instead."
        )

    def _analyze_pretokenized(self, sentences: list[list[str]]) -> DependencyParse:
        """Process pre-tokenized sentences by creating spaCy Docs with preset tokens.

        Each word list becomes a spaCy Doc with fixed token boundaries,
        bypassing spaCy's tokenizer. All other pipeline components (tagger,
        dependency parser) are then run on each Doc individually.

        Sentence starts are deliberately NOT pinned here, unlike in the Benepar
        wrapper over the same spaCy pipeline. Benepar reads `doc.sents` and takes
        one tree per sentence, so a Doc spaCy re-segments yields several trees for
        one input and the output stops lining up with the gold — the defect that
        corrupted its constituency scores. This method never reads `doc.sents`: it
        emits one Sentence per input built from the word list, with indices over
        the Doc, so re-segmentation cannot move a token.

        It does change ATTACHMENT, and the effect was measured on 2026-10-08
        rather than assumed: spaCy re-segments 65 of the EWT test split's 2077
        pre-tokenised sentences (3.1%), and those are exactly the 65 where the
        emitted Sentence carries more than one root. Pinning changes the parse of
        those 65 and of no others, and moves the corpus figures by LAS 0.5101 ->
        0.5099 and UAS 0.5663 -> 0.5666 — two to three hundredths of a point, in
        opposite directions. It is left unpinned because that is how every
        published spaCy number here was measured, and the alternative is not
        better, only different.

        Those figures are en_core_web_sm's. The EWT pipeline is trained on
        documents of one sentence each (scripts/train_spacy.py), so it has
        seen no sentence boundary inside a document; the sentences it splits
        all the same are counted on the dev split in its training record
        (`multi_root_sentences`).

        Args:
            sentences: List of word lists, one per sentence.
        """
        from spacy.tokens import Doc  # deferred import, only needed here

        result_sentences = []
        for words in sentences:
            # Build a Doc with fixed token boundaries, bypassing spaCy's tokenizer
            doc = Doc(self._nlp.vocab, words=words)
            # Run pipeline components manually (tagger, parser, etc.) on the Doc
            for name, proc in self._nlp.pipeline:
                doc = proc(doc)
            # Indices over the whole Doc (start=0), so re-segmentation cannot
            # move a token; the conversion is shared with the Benepar wrapper.
            tokens = doc_tokens(doc, tier=self._map_tier)
            result_sentences.append(Sentence(text=" ".join(words), tokens=tokens))
        return DependencyParse(sentences=result_sentences, analyzer_name=self.name)

    def _doc_to_depparse(self, doc) -> DependencyParse:
        """Convert a spaCy Doc (with multiple sentences) to our DependencyParse format.

        Handles sentence-relative indexing: spaCy token indices are document-global,
        but our Token.index must be 1-based within each sentence.

        Args:
            doc: A processed spaCy Doc object.
        """
        return doc_to_depparse(doc, tier=self._map_tier, name=self.name)

    def _ensure_loaded(self) -> None:
        """Verify that load_model() has been called before analysis.

        Raises:
            RuntimeError: If the spaCy pipeline has not been loaded.
        """
        # Guard against calling analyze methods before loading a model
        self._require_loaded(self._nlp is not None)
