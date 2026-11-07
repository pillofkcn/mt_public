"""Stanza (Stanford NLP) — dependency parsing, constituency parsing, POS tagging.

Stanza is a neural NLP toolkit developed by the Stanford NLP Group. The library
covers many languages; this wrapper covers English only: its package table, its
constituency models and every metadata property describe Stanza's English models.

How it works:
  1. Models are organized as PIPELINES of processors, each handling one task:
     tokenize -> mwt -> pos -> lemma -> depparse -> constituency
  2. The tagger and the dependency parser are BiLSTM networks trained on
     Universal Dependencies treebanks (which ones depends on the package, see
     _PACKAGES); the constituency parser is trained on the revised Penn
     Treebank, WSJ sections 02-21, whatever the package (_CONSTITUENCY_MODELS)
  3. Each of them comes in two variants: with a character language model
     ("charlm", the default) and with the same network reading the hidden
     states of a pretrained transformer as well ("electra-large"). The
     transformer variants are chosen per configuration, by `package` for the
     tagger and the parser and by `constituency_model` for the constituency
     parser
  4. Models auto-download from Stanford servers on first load_model() call
     and are cached in Stanza's resources directory for subsequent runs; the
     transformer the ELECTRA-large variants read is fetched once from the
     Hugging Face hub through Stanza's own loader, and load_model() refuses
     any revision of it other than the one in _TRANSFORMER_REVISIONS. During
     a run nothing is downloaded (offline.py): the models and the transformer
     are read from disk, and one that is not there fails the load

Key characteristics:
  - Supports ALL three analysis types: POS tagging, dependency, constituency
    (the `ewt` and `combined_electra-large` packages, and a row built with
    constituency=False: POS tagging and dependency only)
  - POS tags and dependencies are native Universal Dependencies output — no
    tag conversion needed; constituency trees follow the Penn Treebank
  - Two pipeline modes: auto-tokenizing (raw text) and pre-tokenized
  - Model size: ~40-165 MB per model file (downloaded once, cached), plus
    1.3 GB for the ELECTRA-large weights the transformer variants read

Install: pip install stanza
"""

from __future__ import annotations

import os
from pathlib import Path

from syntactic_analysis.base import (
    AnalysisType,
    AnnotationScheme,
    BaseAnalyzer,
    MethodLevel,
    MethodType,
    TrainingDomain,
)
from syntactic_analysis.models import (
    ConstituencyParse,
    DependencyParse,
    Sentence,
    Token,
)
from syntactic_analysis.downloads import (
    DownloadedFiles,
    hf_snapshot,
    library_data,
    stanza_resources_dir,
)
from syntactic_analysis.offline import downloads_allowed, not_cached


# The Universal Dependencies release the tagger and parser models of every
# package below were trained on. Stanza stamps no release onto a model file, so
# it is established from two published sources: the release notes of Stanza
# 1.10.0 (23 Dec 2024, github.com/stanfordnlp/stanza/releases/tag/v1.10.0) say
# "we rebuild all of the models with UD 2.15", and the resources index the
# installed Stanza resolves models against (resources_1.11.0.json, in
# github.com/stanfordnlp/stanza-resources) lists for every English pos and
# depparse model named here, and for both constituency models, the same MD5
# as resources_1.10.0.json and another one than resources_1.9.0.json: the
# models of 1.11.0 are the files built for 1.10.0. The model files downloaded
# here match the MD5s of the index (checked 23 Oct 2026).
_UD_RELEASE = "UD 2.15"

# The date of the models, as plan/LOCK.json records it under `versions`
# (scripts/lock_plan.py, since 24 Oct 2026): the release they were built for,
# established from the sources above. The resources index they are resolved
# against is stanza.__resources_version__ (1.11.0).
MODELS_BUILT_FOR = "Stanza 1.10.0 (release notes of 23 Dec 2024)"

# What the "combined" models are trained on. Stanza's build script of 1.10.0
# (stanza/utils/datasets/prepare_tokenizer_treebank.py,
# build_combined_english_dataset and build_extra_combined_english_dataset)
# concatenates the TRAIN splits of UD_English-EWT, UD_English-GUM and
# UD_English-GUMReddit, the TEST splits of UD_English-PUD and
# UD_English-Pronouns, and the hand-parsed English sentences of
# github.com/stanfordnlp/handparsed-treebank (english-handparsed/english.conll);
# its dev split is the EWT dev split. Until 23 Oct 2026 this string said
# "EWT + GUM + GUMReddit + PUD + Pronouns" and left the hand-parsed sentences and
# the splits out.
#
# Neither recipe, this one or the EWT-only one below, reads UD_English-GENTLE.
# The hand-parsed sentences are PTB-style trees converted to UD with CoreNLP
# (the readme beside english.conll); apart from a one-token line "------",
# none of them is an EWT test sentence (the repository's current file,
# compared token by token on 23 Oct 2026).
#
# Stanza's documentation does not settle the release: its pages on available
# models and model history stop at UD 2.12 (Stanza 1.5.1), which the release
# notes of 1.10.0 superseded.
_COMBINED_CORPUS = (
    f"{_UD_RELEASE} combined: EWT, GUM and GUMReddit train + PUD and Pronouns "
    f"test + Stanza's hand-parsed English sentences (dep/POS)"
)

# Stanza's English packages, and what each one is trained on. The default
# "combined" package is several treebanks at once, which makes it a different
# KIND of in-domain model from one trained on the EWT alone — see TrainingDomain
# and the `package` argument below.
#
#   stanza_package  the package Stanza resolves every processor from that
#                   `overrides` does not name
#   overrides       processors whose model is named explicitly instead
#   models          the pos and depparse models this row must end up with:
#                   checked against what the pipeline loaded (load_model)
#   level, detail   method level and architecture of the tagger and the parser
#   transformer     the pretrained encoder those models read, "" for none
#   constituency    the constituency model the package carries by default, or
#                   None for a package that is a tagger and parser only
#   corpus_pos_dep  the training corpus of the tagger and the parser
_PACKAGES: dict[str, dict] = {
    "combined": {
        "stanza_package": "combined",
        "overrides": {},
        "models": {"pos": "combined_charlm", "depparse": "combined_charlm"},
        "level": MethodLevel.NEURAL_RECURRENT,
        "detail": "BiLSTM",
        "transformer": "",
        "domain": TrainingDomain.IN_PLUS,
        "constituency": "ptb3-revised_charlm",
        "label": "",
        "corpus_pos_dep": _COMBINED_CORPUS,
    },
    "ewt": {
        # pos and depparse resolve to ewt_charlm — the same architecture as
        # combined_charlm, trained on UD_English-EWT alone (its train split;
        # dev is its dev split). The package ships no constituency model, so
        # this instance declares POS and dependencies only.
        "stanza_package": "ewt",
        "overrides": {},
        "models": {"pos": "ewt_charlm", "depparse": "ewt_charlm"},
        "level": MethodLevel.NEURAL_RECURRENT,
        "detail": "BiLSTM",
        "transformer": "",
        "domain": TrainingDomain.IN,
        "constituency": None,
        "label": "EWT",
        "corpus_pos_dep": (
            f"UD_English-EWT r{_UD_RELEASE.split()[1]} train only (ewt_charlm, dep/POS)"
        ),
    },
    # Not a package of Stanza's: this wrapper's name for the "combined" package
    # with its tagger and parser replaced by the transformer models trained on
    # the same data. Everything else, the lemmatizer included, stays the
    # combined package's, so that against "combined" only the tagger and the
    # parser change. (Stanza's own "default_accurate" package also swaps the
    # lemmatizer, NER and sentiment models.) Both models keep the BiLSTM of the
    # charlm variant, with the same word vectors (conll17) and character
    # language model (1billion), and add the hidden states of
    # google/electra-large-discriminator, which they read without fine-tuning
    # it: the model files record bert_model = google/electra-large-discriminator
    # and bert_finetune = False, and hold no transformer weights of their own
    # (the parser holds the four weights that mix the layers it reads). Their
    # train and dev files are the combined package's (en_combined.train.in /
    # en_combined.dev.in in both model files).
    "combined_electra-large": {
        "stanza_package": "combined",
        "overrides": {
            "pos": "combined_electra-large",
            "depparse": "combined_electra-large",
        },
        "models": {
            "pos": "combined_electra-large",
            "depparse": "combined_electra-large",
        },
        "level": MethodLevel.NEURAL_PRETRAINED,
        "detail": "BiLSTM over frozen ELECTRA-large",
        "transformer": "google/electra-large-discriminator",
        "domain": TrainingDomain.IN_PLUS,
        "constituency": None,
        "label": "ELECTRA-large",
        "corpus_pos_dep": _COMBINED_CORPUS,
    },
}

# Stanza's English constituency models. Deliberately NOT package names: a
# package names a treebank collection for tokenize / pos / depparse, while
# constituency is a Penn Treebank model independent of them. Naming the package
# across the whole processor chain made Stanza log "Can not find constituency:
# combined ... Ignoring it" and drop the processor while supported_types still
# advertised it, so the constituency stage registered the row and scored empty
# trees.
#
# Both are in-order transition parsers with an LSTM stack, trained on
# en_ptb3-revised_train.mrg with no silver trees (silver_file = None in both
# model files), which Stanza's prepare_con_dataset.py builds from sections
# 02-21 of LDC2015T13. Both retag their input with the pipeline's own tagger
# (retag_method = xpos), so the tagger of the row is part of what a
# constituency score measures.
_CONSTITUENCY_MODELS: dict[str, dict] = {
    "ptb3-revised_charlm": {
        "level": MethodLevel.NEURAL_RECURRENT,
        "detail": "BiLSTM",
        "transformer": "",
        "label": "",
        "corpus": "PTB3 revised WSJ 02-21 (constituency)",
    },
    "ptb3-revised_electra-large": {
        # bert_model = google/electra-large-discriminator, bert_finetune = False
        "level": MethodLevel.NEURAL_PRETRAINED,
        "detail": "BiLSTM over frozen ELECTRA-large",
        "transformer": "google/electra-large-discriminator",
        "label": "ELECTRA-large constituency",
        "corpus": "PTB3 revised WSJ 02-21 (constituency)",
    },
}

# The revision of each pretrained transformer a row reads. A Stanza model file
# names its transformer by hub id only (bert_model), and the hub resolves an id
# to whatever its main branch holds, so the 1.3 GB of weights a transformer row
# depends on could change under an unchanged Stanza model. load_model() reads
# the revision the weights were loaded from and refuses any other. Recorded on
# 23 Oct 2026: the main branch of google/electra-large-discriminator (last
# modified 29 Feb 2024), whose pytorch_model.bin as downloaded matched the
# hub's SHA-256 for it (070894f07d07a499...).
_TRANSFORMER_REVISIONS: dict[str, str] = {
    "google/electra-large-discriminator": "c13c3df7efadc2162f42588bd28eb4e187d602a5",
}

# The key a package gets when the caller names none, derived as the
# UDPipe wrapper derives its own. Until 16 Oct 2026 the default was
# "stanza" whatever the package, so an "ewt" instance needed its key typed by
# hand or it collided with the combined row.
_DEFAULT_KEYS = {
    "combined": "stanza",
    "ewt": "stanza_ewt",
    "combined_electra-large": "stanza_trf",
}


def _loaded_paths(load_list, config) -> list[Path]:
    """The model files the processors of a resolved Stanza pipeline read.

    `load_list` and `config` as stanza.Pipeline resolves them (its load_list
    and config): for each processor, its model and the files it depends on
    (pretrain, charlm), which build_default_config names
    "<processor>_model_path" and "<processor>_<dependency>_path". The
    tokenizer of a pre-tokenized pipeline reads none of its files: its
    processor builds no model (stanza/pipeline/tokenize_processor.py,
    _set_up_model), which the opened files of every load showed on
    24 Oct 2026. Sorted, each once.
    """
    paths: set[Path] = set()
    for processor, _specs in load_list:
        if processor == "tokenize" and config.get("tokenize_pretokenized"):
            continue
        for key, value in config.items():
            if not (key.startswith(f"{processor}_") and key.endswith("_path")) or not value:
                continue
            for path in value if isinstance(value, (list, tuple)) else [value]:
                paths.add(Path(os.path.abspath(path)))
    return sorted(paths)


class StanzaAnalyzer(BaseAnalyzer):

    def __init__(
        self,
        *,
        key: str | None = None,
        package: str = "combined",
        constituency_model: str | None = None,
        constituency: bool = True,
    ) -> None:
        """Create the analyzer.

        Args:
            key:     Key and CSV column name, overriding the one the
                     package gives it ("stanza" for "combined", "stanza_ewt"
                     for "ewt", "stanza_trf" for "combined_electra-large").
                     Rows need distinct keys (a key names its rows in the
                     counts files), and the derived ones are. A row whose
                     constituency model is not its
                     package's default has no derived key and must be given one.
            package: Which tagger and parser to load. "combined" (the default
                     Stanza gives a user who asks for English) is trained on
                     several treebanks; "ewt" is trained on UD_English-EWT
                     alone; "combined_electra-large" is "combined" with the
                     transformer tagger and parser trained on the same data.

                     The combined / ewt pair is the only contrast in the
                     project that changes training data with the architecture
                     held exactly fixed at the neural level: same code, same
                     charlm setup, one treebank versus several. The combined /
                     combined_electra-large pair holds the data fixed and adds
                     the pretrained transformer.
            constituency_model: Which constituency model a package that carries
                     one loads: "ptb3-revised_charlm" (Stanza's default, used
                     when None) or "ptb3-revised_electra-large".
            constituency: False leaves the package's constituency model out:
                     the row is a tagger and parser only, does not support
                     constituency parsing, and its pipeline loads and runs no
                     constituency processor. For a row that is scored on POS
                     and dependencies only (stanza_comb in plan/v1.yaml), whose
                     pipeline would otherwise parse every sentence a second
                     time, into trees nobody reads. The processor runs after
                     the tagger and the parser and changes neither's output.
                     Added 23 Oct 2026; True keeps the behaviour of before.

        Raises:
            ValueError: If `package` or `constituency_model` is not one this
                wrapper knows about, if a constituency model is named for a
                package that carries none or for a row built without one
                (`constituency=False`), or if a non-default constituency
                model comes without a key.
        """
        if package not in _PACKAGES:
            raise ValueError(
                f"unknown package {package!r}; expected one of {sorted(_PACKAGES)}"
            )
        if not constituency and constituency_model is not None:
            raise ValueError(
                f"constituency_model {constituency_model!r} names a constituency model "
                f"for a row built without one (constituency=False)"
            )
        default_constituency = _PACKAGES[package]["constituency"] if constituency else None
        if constituency_model is not None:
            if constituency_model not in _CONSTITUENCY_MODELS:
                raise ValueError(
                    f"unknown constituency_model {constituency_model!r}; "
                    f"expected one of {sorted(_CONSTITUENCY_MODELS)}"
                )
            if default_constituency is None:
                carriers = sorted(
                    name for name, entry in _PACKAGES.items() if entry["constituency"]
                )
                raise ValueError(
                    f"package {package!r} carries no constituency model; "
                    f"constituency_model applies to {carriers}"
                )
            if constituency_model != default_constituency and key is None:
                # The derived key names the package, not the constituency model,
                # so it would collide with the row of the default model.
                raise ValueError(
                    f"constituency_model {constituency_model!r} is not the default "
                    f"of package {package!r}; name the row with `key`"
                )
        self._key = key or _DEFAULT_KEYS[package]
        self._package = package
        self._constituency_model = constituency_model or default_constituency
        self._nlp = None          # auto-tokenizing pipeline for raw text — built lazily, see _get_raw_pipeline
        self._nlp_pretok = None   # pipeline that preserves caller's tokenization
        self._model_name = None   # requested language/package, stashed for lazy pipeline build + provenance
        self._processors = None   # resolved processor chain, needed to build _nlp on demand
        self._load_kwargs: dict = {}  # caller kwargs from load_model(), replayed for both pipelines
        self._loaded_models: dict[str, str] = {}  # processor -> model the pipeline loaded
        self._foundation_cache = None  # holds the transformer both pipelines share, if any
        self._transformer_revisions: dict[str, str] = {}  # hub id -> revision loaded

    @property
    def name(self) -> str:
        return self._key

    @property
    def package(self) -> str:
        """Which Stanza English package this instance loads."""
        return self._package

    @property
    def constituency_model(self) -> str | None:
        """The constituency model this instance loads, None for a package with none."""
        return self._constituency_model

    @property
    def supported_types(self) -> set[AnalysisType]:
        types = {
            AnalysisType.POS_TAGGING,
            AnalysisType.DEPENDENCY_PARSING,
        }
        if self._constituency_model is not None:
            types.add(AnalysisType.CONSTITUENCY_PARSING)
        return types

    @property
    def method_type(self) -> MethodType:
        return MethodType.NEURAL

    @property
    def method_detail(self) -> str:
        # One string when every component is built the same way ("BiLSTM" for
        # the charlm rows, as before 23 Oct 2026); both when they differ.
        tagger = _PACKAGES[self._package]["detail"]
        if self._constituency_model is None:
            return tagger
        parser = _CONSTITUENCY_MODELS[self._constituency_model]["detail"]
        return tagger if parser == tagger else f"{tagger} / {parser} (constituency)"

    def method_detail_for(self, analysis_type: AnalysisType) -> str:
        """The architecture of one task: the package's for POS and dependencies,
        the constituency model's for constituency."""
        if (
            analysis_type is AnalysisType.CONSTITUENCY_PARSING
            and self._constituency_model is not None
        ):
            return _CONSTITUENCY_MODELS[self._constituency_model]["detail"]
        return _PACKAGES[self._package]["detail"]

    @property
    def display_name(self) -> str:
        labels = [_PACKAGES[self._package]["label"]]
        if self._constituency_model is not None:
            labels.append(_CONSTITUENCY_MODELS[self._constituency_model]["label"])
        labels = [label for label in labels if label]
        return f"Stanza ({', '.join(labels)})" if labels else "Stanza"

    @property
    def annotation_scheme(self) -> dict:
        scheme = {
            AnalysisType.POS_TAGGING: AnnotationScheme.UD,
            AnalysisType.DEPENDENCY_PARSING: AnnotationScheme.UD,
        }
        if self._constituency_model is not None:
            scheme[AnalysisType.CONSTITUENCY_PARSING] = AnnotationScheme.PTB
        return scheme

    @property
    def method_level(self) -> dict:
        # Tagger and biaffine parser: BiLSTM over pretrained word vectors + a
        # character LSTM, L3; the transformer variants read ELECTRA-large's
        # hidden states as well, L4 (MethodLevel). Constituency: the level of
        # the constituency model, by the same rule.
        level = _PACKAGES[self._package]["level"]
        levels = {
            AnalysisType.POS_TAGGING: level,
            AnalysisType.DEPENDENCY_PARSING: level,
        }
        if self._constituency_model is not None:
            levels[AnalysisType.CONSTITUENCY_PARSING] = (
                _CONSTITUENCY_MODELS[self._constituency_model]["level"]
            )
        return levels

    @property
    def training_domain(self) -> dict:
        # The dep/POS models follow the package: "combined" and its transformer
        # variant are EWT plus other treebanks (IN_PLUS), "ewt" is the EWT alone
        # (IN). The constituency model is PTB WSJ regardless, which is
        # out-of-domain for the EWT test (and in-domain for the WSJ-01 gold run,
        # which says so itself rather than reading this property).
        domain = _PACKAGES[self._package]["domain"]
        out = {
            AnalysisType.POS_TAGGING: domain,
            AnalysisType.DEPENDENCY_PARSING: domain,
        }
        if self._constituency_model is not None:
            out[AnalysisType.CONSTITUENCY_PARSING] = TrainingDomain.OUT
        return out

    def _constituency_corpus(self) -> str:
        if self._constituency_model is None:
            return ""
        return _CONSTITUENCY_MODELS[self._constituency_model]["corpus"]

    @property
    def training_corpus(self) -> str:
        return " / ".join(
            part
            for part in (_PACKAGES[self._package]["corpus_pos_dep"], self._constituency_corpus())
            if part
        )

    def training_corpus_for(self, analysis_type: AnalysisType) -> str:
        """The corpus of one task: the UD treebanks, release included, for POS
        and dependencies, the Penn Treebank for constituency.

        The tool-wide `training_corpus` names both, so a per-task cell built
        from it would carry the other task's corpus as well.
        """
        if analysis_type is AnalysisType.CONSTITUENCY_PARSING:
            # A package with no constituency model has no corpus for it;
            # nothing prints this cell for a row that does not support the task.
            return self._constituency_corpus() or super().training_corpus_for(analysis_type)
        return _PACKAGES[self._package]["corpus_pos_dep"]

    @property
    def model_version(self) -> str:
        """Concrete loaded-model identity, for provenance logging.

        Stanza does not stamp a semantic version onto individual downloaded
        model files — the resources.json package format only exposes model
        *names* (e.g. "combined_charlm"), not per-model version numbers. What is
        reported instead: the requested language/package, the model each
        processor of the pipeline actually loaded, the UD release the tagger
        and parser were trained on (_UD_RELEASE, with its sources), the
        transformer they read if any with the first twelve characters of the
        revision its weights were loaded from, the resources.json schema
        version those models were resolved against, and the stanza library
        version.

        Returns:
            Empty string if load_model() has not been called yet.
        """
        if self._model_name is None:
            return ""
        import stanza  # lazy import to avoid hard dependency at module level

        loaded = ", ".join(f"{proc}={model}" for proc, model in self._loaded_models.items())
        transformers = [
            f"{name}@{self._transformer_revisions.get(name, '?')[:12]}"
            for name in self._transformers()
        ]
        reads = f"; transformer {', '.join(transformers)}" if transformers else ""
        return (f"{self._model_name}/{self._package} [{loaded}]; pos/depparse trained "
                f"on {_UD_RELEASE}{reads} (stanza {stanza.__version__}, "
                f"resources {stanza.__resources_version__})")

    def _pipeline_package(self) -> str | dict[str, str]:
        """What to pass as Stanza's `package`, per processor where it matters.

        A package name selects a treebank collection, and that only applies to
        the processors trained on treebanks. Constituency is trained on the Penn
        Treebank whatever the package is, so it has no model under a package's
        name; passing the name across the whole chain makes Stanza drop the
        processor with a warning rather than fail, which is how a whole stage
        came to score empty trees. The dict form pins the treebank collection
        for everything else and names the other models explicitly: the
        constituency model, and the transformer tagger and parser of
        "combined_electra-large".
        """
        entry = _PACKAGES[self._package]
        overrides = dict(entry["overrides"])
        if self._constituency_model is not None:
            overrides["constituency"] = self._constituency_model
        if not overrides:
            return entry["stanza_package"]
        return {"default": entry["stanza_package"], **overrides}

    def _expected_models(self) -> dict[str, str]:
        """The model each scored processor must load, by processor name."""
        expected = dict(_PACKAGES[self._package]["models"])
        if self._constituency_model is not None:
            expected["constituency"] = self._constituency_model
        return expected

    def _transformers(self) -> list[str]:
        """The hub ids of the pretrained transformers this row reads, sorted."""
        names = {_PACKAGES[self._package]["transformer"]}
        if self._constituency_model is not None:
            names.add(_CONSTITUENCY_MODELS[self._constituency_model]["transformer"])
        return sorted(name for name in names if name)

    def _load_transformers(self, base=None):
        """Load the transformers this row reads into a Stanza FoundationCache.

        Both pipelines are built with download_method=None, and with that
        Stanza loads a transformer from the local hub cache only: on a machine
        that has not fetched ELECTRA-large yet the pipeline would fail to load.
        Loading it here, through Stanza's own loader (FoundationCache.load_bert),
        fetches it when it is missing, and the pipelines handed this cache reuse
        the loaded weights instead of loading them a second time. The revision
        the weights came from is checked against _TRANSFORMER_REVISIONS and kept
        for model_version. During a run (offline.py) it is read from the local
        cache only, and a transformer that is not there is refused, not fetched.

        Args:
            base: A FoundationCache the caller passed to load_model, or None.

        Returns:
            The cache to hand to stanza.Pipeline; `base` itself for a row that
            reads no transformer.

        Raises:
            RuntimeError: If a transformer cannot be loaded, or its weights come
                from another revision than the one recorded.
            offline.NotCachedError: During a run, if a transformer is not in
                the local hub cache.
        """
        names = self._transformers()
        self._transformer_revisions = {}
        if not names:
            return base
        from stanza.models.common.foundation_cache import FoundationCache

        cache = FoundationCache(base)
        revisions = {}
        for name in names:
            try:
                model, _ = cache.load_bert(name, local_files_only=True)
            except OSError as missing:
                # Not in the local hub cache yet (or only in part): fetch it,
                # unless this is a run, which downloads nothing.
                if not downloads_allowed():
                    raise not_cached(f"the transformer {name!r} of {self.name!r}",
                                     "the Hugging Face cache") from missing
                try:
                    model, _ = cache.load_bert(name, local_files_only=False)
                except OSError as exc:
                    raise RuntimeError(
                        f"could not load transformer {name!r} for {self.name!r}: {exc}"
                    ) from exc
            revision = getattr(model.config, "_commit_hash", None)
            if revision != _TRANSFORMER_REVISIONS[name]:
                raise RuntimeError(
                    f"transformer {name!r} for {self.name!r} was loaded from revision "
                    f"{revision!r}; {_TRANSFORMER_REVISIONS[name]!r} is the one recorded"
                )
            revisions[name] = revision
        self._transformer_revisions = revisions
        return cache

    def _default_processors(self) -> str:
        """The processor chain load_model builds when the caller names none."""
        processors = "tokenize,mwt,pos,lemma,depparse"
        if self._constituency_model is not None:
            processors += ",constituency"
        return processors

    def _resolved_files(self, model_name: str, options: dict) -> tuple[Path, list[Path]]:
        """Stanza's resources directory and the model files the pre-tokenized pipeline of load_model reads.

        Resolved by Stanza's own functions, the ones stanza.Pipeline calls
        before it loads anything (stanza/pipeline/core.py, Pipeline.__init__):
        process_pipeline_parameters, load_resources_json (the resources.json
        of the directory), maintain_processor_list, add_dependencies,
        Pipeline.update_kwargs and build_default_config, with the arguments
        load_model gives the pipeline: this row's package and processors, the
        caller's keywords, and tokenize_pretokenized. Nothing is loaded.
        load_model compares the result with what the pipeline it built
        resolved (_check_resolved), so the two cannot differ unnoticed.

        Raises:
            FileNotFoundError: If the directory holds no resources.json.
        """
        import inspect

        from stanza.pipeline.core import Pipeline, build_default_config
        from stanza.resources.common import (
            DEFAULT_MODEL_DIR,
            add_dependencies,
            load_resources_json,
            maintain_processor_list,
            process_pipeline_parameters,
        )

        options = dict(options)
        processors = options.pop("processors", self._default_processors())
        directory = options.pop("dir", DEFAULT_MODEL_DIR)
        model_dir = options.pop("model_dir", None)
        if model_dir is not None and directory == DEFAULT_MODEL_DIR:
            directory = model_dir
        # The keywords the pipeline takes by name are not in its kwargs; the
        # others are, and update_kwargs and the config read them.
        named = set(inspect.signature(Pipeline.__init__).parameters)
        extra = {key: value for key, value in options.items() if key not in named}
        extra["tokenize_pretokenized"] = True
        lang, directory, package, processors = process_pipeline_parameters(
            model_name, directory, self._pipeline_package(), processors,
        )
        resources = load_resources_json(directory)
        load_list = maintain_processor_list(
            resources, lang, package, processors, maybe_add_mwt=False,
        )
        load_list = add_dependencies(resources, lang, load_list)
        load_list = Pipeline.update_kwargs(extra, load_list)
        config = build_default_config(resources, lang, directory, load_list)
        config.update(extra)
        return Path(os.path.abspath(directory)), _loaded_paths(load_list, config)

    def downloaded_files(self, model_name: str = "en", **kwargs) -> tuple[DownloadedFiles, ...]:
        """The files of Stanza's resources directory load_model reads, and the transformer's.

        Takes load_model's keywords. Under stanza_resources: resources.json,
        the index the pipeline resolves its models with, and every model file
        the resolved pipeline reads (_resolved_files) - each processor's model
        with its pretrained vectors and character language models, the mwt
        and the lemmatizer included, the tokenizer's not (a pre-tokenized
        pipeline does not load it). For a row that reads ELECTRA-large, under
        hf_hub_cache the snapshot of the revision _TRANSFORMER_REVISIONS
        records, and refs/main, by which the hub cache finds that snapshot
        when _load_transformers loads the transformer by name
        (downloads.hf_snapshot). Without a resources.json there is nothing
        to resolve the models with, and resources.json alone is declared,
        so a lock check reports it missing. Under site_packages the emoji
        package's emoji.json, which `import stanza` reads (stanza/pipeline/
        core.py imports langid_processor.py, which imports emoji;
        downloads.LIBRARY_DATA; since 25 Oct 2026). Nothing is loaded.
        Since 24 Oct 2026.

        Raises:
            ValueError: If the keywords name another resources directory than
                Stanza's own, which LOCK.json could not name.
        """
        own = Path(os.path.abspath(stanza_resources_dir()))
        try:
            directory, files = self._resolved_files(model_name, kwargs)
        except FileNotFoundError:
            if (own / "resources.json").is_file():
                raise
            directory, files = own, []
        if os.path.normcase(directory) != os.path.normcase(own):
            raise ValueError(
                f"{self.name}: the pipeline would read the resources directory {directory}, "
                f"not Stanza's own; LOCK.json records the files of Stanza's own"
            )
        declared = [
            DownloadedFiles.under("stanza_resources", own, [own / "resources.json", *files])
        ]
        declared += [hf_snapshot(name, _TRANSFORMER_REVISIONS[name]) for name in self._transformers()]
        declared += library_data("emoji")
        return tuple(declared)

    def _check_resolved(self, pipeline, model_name: str, options: dict) -> None:
        """Refuse a pipeline that read other model files than downloaded_files declares.

        Raises:
            RuntimeError: If the pipeline's resources directory or the model
                files its processors read differ from _resolved_files.
        """
        directory, declared = self._resolved_files(model_name, options)
        loaded = _loaded_paths(pipeline.load_list, pipeline.config)

        def canonical(paths):
            return sorted(os.path.normcase(str(path)) for path in paths)

        if (canonical([os.path.abspath(pipeline.dir)]) != canonical([directory])
                or canonical(loaded) != canonical(declared)):
            raise RuntimeError(
                f"Stanza's pipeline for {self.name!r} read other model files than the row "
                f"declares (downloaded_files): declared {canonical(declared)}, read "
                f"{canonical(loaded)}"
            )

    def load_model(self, model_name: str = "en", **kwargs) -> None:
        """Download and load a Stanza model for the specified language.

        Args:
            model_name: Stanza's language code. "en" is the only value this
                        wrapper is written for: the package table and the
                        constituency models are English ones.
            **kwargs:   Extra keyword arguments forwarded to both underlying
                        stanza.Pipeline() constructions (e.g. tqdm), so tuning
                        applies regardless of which pipeline a given analysis
                        call ends up using. `use_gpu` defaults to False: the
                        project runs on the CPU, and Stanza would otherwise
                        take a GPU whenever torch finds one.

        First call downloads the model files from Stanford servers, and for a
        transformer variant _load_transformers fetches ELECTRA-large (1.3 GB)
        from the Hugging Face hub. Subsequent calls use the cached files.
        stanza.download() also reads Stanza's resources index from the network
        on every call, and fetches a model again whose MD5 differs from it.
        During a run (offline.py) stanza.download() is not called: the pipeline
        reads the files already in Stanza's resources directory, with the
        resources.json there, and a missing one fails the load.

        Two pipelines back this analyzer:
          1. _nlp:        For raw text input — includes Stanza's own tokenizer.
                          Built lazily on first raw-text use (see _get_raw_pipeline),
                          since every stage script in this repo only ever
                          drives the pre-tokenized path below; eagerly building
                          both here would double the resident model memory for
                          no benefit in the common case.
          2. _nlp_pretok: For pre-tokenized input — tokenizer is bypassed so
                          the exact word boundaries from the caller are preserved.
                          This is essential for fair evaluation against gold standard,
                          where tokenization differences would skew the metrics.
                          Built eagerly here since this is the path actually used.

        Raises:
            RuntimeError: If the pipeline loaded another model for pos,
                depparse or constituency than this row declares. Stanza
                resolves a package name through its resources index, so a
                changed index would otherwise change the row's model silently.
                Also if a transformer the row reads cannot be loaded or comes
                from another revision than _TRANSFORMER_REVISIONS records, and
                if the pipeline read other model files than downloaded_files
                declares (_check_resolved; since 24 Oct 2026).
            offline.NotCachedError: During a run, if a model file, Stanza's
                resources.json or the transformer is not on disk.
        """
        import stanza  # lazy import to avoid hard dependency at module level

        # What downloaded_files is given for this load, to compare with what
        # the pipeline resolves (_check_resolved).
        options = dict(kwargs)
        # The processor chain follows the package: "ewt" ships no constituency
        # model, and asking for one would either fall back to the combined
        # package's model — silently mixing two training sets in one row — or
        # fail at load time. supported_types and this default are derived from
        # the same table, so they cannot disagree. A caller's `processors` can
        # still leave constituency out; _analyze_constituency then refuses
        # rather than returning an empty tree for every sentence.
        # allow caller to override the default processor chain
        processors = kwargs.pop("processors", self._default_processors())
        kwargs.setdefault("use_gpu", False)
        package = self._pipeline_package()

        # Resolved before downloading, and both are handed to download(): the
        # per-processor package form is only legal alongside a processor list,
        # and download() would otherwise reject the dict outright. Not during a
        # run, which downloads nothing (offline.py).
        if downloads_allowed():
            stanza.download(
                model_name, processors=processors, package=package, logging_level="WARN"
            )
        # The transformer is not one of Stanza's downloads; see _load_transformers.
        foundation_cache = self._load_transformers(kwargs.pop("foundation_cache", None))

        # stash for the lazy raw-text pipeline (_get_raw_pipeline) and model_version
        self._model_name = model_name
        self._processors = processors
        self._load_kwargs = kwargs
        self._foundation_cache = foundation_cache

        # _nlp (raw text) is intentionally NOT built here — see docstring above

        # create pipeline for pre-tokenized input; this is the one every
        # stage script actually drives
        try:
            pipeline = stanza.Pipeline(
                model_name, processors=processors,
                package=package,
                tokenize_pretokenized=True,  # accept token lists instead of raw text
                download_method=None,        # skip re-download, model already cached above
                logging_level="WARN",
                foundation_cache=foundation_cache,
                **kwargs,
            )
        except FileNotFoundError as exc:
            # Stanza's errors for a missing resources.json, language directory
            # or model file are all FileNotFoundError. Outside a run they pass
            # as they are; during a run nothing was downloaded above, so say so.
            if downloads_allowed():
                raise
            raise not_cached(f"a Stanza model of {self.name!r} ({type(exc).__name__})",
                             "Stanza's resources directory") from exc
        loaded = {
            processor: ";".join(spec.package for spec in specs)
            for processor, specs in pipeline.load_list
        }
        wrong = [
            f"{processor}: declared {model!r}, loaded {loaded[processor]!r}"
            for processor, model in self._expected_models().items()
            if processor in loaded and loaded[processor] != model
        ]
        if wrong:
            raise RuntimeError(
                f"Stanza loaded other models than {self.name!r} declares: "
                f"{'; '.join(wrong)}"
            )
        # The files the lock records for this row are the ones it read.
        self._check_resolved(pipeline, model_name, options)
        self._loaded_models = loaded
        self._nlp_pretok = pipeline

    def _get_raw_pipeline(self):
        """Build (once) and return the auto-tokenizing pipeline for raw text.

        Deferred until first use — see the laziness rationale in load_model()'s
        docstring. Subsequent calls reuse the already-built pipeline.
        """
        if self._nlp is None:
            import stanza  # lazy import to avoid hard dependency at module level

            self._nlp = stanza.Pipeline(
                self._model_name, processors=self._processors,
                package=self._pipeline_package(),  # same models as the pre-tokenized pipeline
                download_method=None,  # model already downloaded in load_model()
                logging_level="WARN",
                foundation_cache=self._foundation_cache,  # the transformer, if any, loaded once
                **self._load_kwargs,
            )
        return self._nlp

    def _analyze_dependency(self, text: str | list[list[str]]) -> DependencyParse:
        """Run the full Stanza pipeline and extract dependency + POS information.

        Args:
            text: Either a raw string (Stanza tokenizes automatically) or
                  a list of pre-tokenized sentences, e.g. [["The", "cat", "sat", "."]].

                  Example raw:          "The cat sat."
                  Example pretokenized: [["The", "cat", "sat", "."]]

        Returns a DependencyParse with POS tags (UPOS), dependency heads, and
        dependency relations for each token. Stanza uses Universal Dependencies
        natively, so no tag conversion is needed (unlike NLTK).
        """
        self._ensure_loaded()

        # pick the right pipeline based on input type
        if isinstance(text, list):
            doc = self._nlp_pretok(text)  # pre-tokenized: preserve exact word boundaries
        else:
            doc = self._get_raw_pipeline()(text)  # raw text: let Stanza tokenize (built on demand)

        sentences = []
        for sent in doc.sentences:
            # map each Stanza word to our unified Token model
            tokens = [
                Token(
                    text=word.text,
                    index=word.id,
                    pos=word.upos,             # universal POS tag (UD standard)
                    xpos=word.xpos or "",      # language-specific POS, empty if missing
                    lemma=word.lemma or "",     # dictionary form of the word
                    dep_rel=word.deprel,        # dependency relation label (e.g. nsubj, root)
                    head_index=word.head,       # index of syntactic head (0 = root)
                )
                for word in sent.words
            ]
            sentences.append(Sentence(text=sent.text, tokens=tokens))
        return DependencyParse(sentences=sentences, analyzer_name=self.name)

    def _analyze_constituency(self, text: str | list[list[str]]) -> ConstituencyParse:
        """Run constituency parsing using Stanza's neural constituency parser.

        Args:
            text: Either a raw string or a list of pre-tokenized sentences
                  (same format as analyze_dependency).

        Returns a ConstituencyParse with one bracketed tree string per sentence,
        e.g. "(S (NP (DT The) (NN cat)) (VP (VBZ sat)))".

        The constituency processor must be in the loaded pipeline (it is by
        default). Without it every tree would come back empty and be scored as
        a parse failure, the silent defect described above _CONSTITUENCY_MODELS,
        so its absence is refused rather than passed through.

        Raises:
            NotImplementedError: For a package with no constituency model
                ("ewt", "combined_electra-large"), as for every analyzer that
                does not support the task.
            RuntimeError: If load_model() was not called, or built the pipeline
                without a constituency processor (a `processors` override that
                leaves it out).
        """
        if not self.supports(AnalysisType.CONSTITUENCY_PARSING):
            raise NotImplementedError(
                f"Stanza package {self._package!r} ships no constituency model; "
                f"it supports POS tagging and dependency parsing only."
            )
        self._ensure_loaded()
        if "constituency" not in self._nlp_pretok.processors:
            raise RuntimeError(
                f"Constituency processor not loaded: the pipeline was built with "
                f"processors={self._processors!r}. Load a chain that includes it."
            )

        # pick the right pipeline based on input type
        if isinstance(text, list):
            doc = self._nlp_pretok(text)
        else:
            doc = self._get_raw_pipeline()(text)  # built on demand, see _get_raw_pipeline

        trees = []
        for sent in doc.sentences:
            if sent.constituency is not None:
                trees.append(str(sent.constituency))  # convert tree object to bracket notation
            else:
                trees.append("")  # no tree for this sentence: normalize records the failure
        return ConstituencyParse(trees=trees, analyzer_name=self.name)

    def _ensure_loaded(self) -> None:
        # guard against running analysis before model initialization.
        # _nlp_pretok (not _nlp) is the reliable signal: _nlp is now built
        # lazily on first raw-text use, so it stays None long after a
        # successful load_model() call in the (common) pre-tokenized-only case.
        self._require_loaded(self._nlp_pretok is not None)
