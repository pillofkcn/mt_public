"""Analyzers built from the plan: one configuration of plan/v1.yaml, ready to run.

build(plan, key) is how version 1.0 gets an analyzer. It does what the plan
says and nothing else:

  1. refuses a configuration the plan does not hold or holds as pending
     (Plan.require_ready);
  2. builds every configuration named in `inputs` first, loads it, and passes
     the instance under its keyword: nltk_arc gets the udpipe_ewt tagger as its
     pos_source this way, never a tagger of the wrapper's own choosing;
  3. constructs the wrapper class named by `wrapper` with the keywords of
     `init`, and calls load_model with the keywords of `load`;
  4. checks the analyzer against the plan, before the load (which can take a
     minute and fetch a gigabyte) and again after it, since a wrapper may
     declare some things only from the model it loaded (spaCy's level).

The check (check_declarations) is what makes the plan the source of truth for
what an analyzer IS, not only for how it is called. For each task the
configuration is a member of, the analyzer must support the task and declare:

  name    the configuration's key, which is the column every result of it is
          written under;
  level   the member's level in that task (tasks.<task>.configs.<key>.level);
  domain  the training domain its training data implies (expected_domain);
  scheme  the annotation scheme its training data implies for that task
          (expected_scheme).

A mismatch is a DeclarationError listing every one found, never a warning: a
wrapper that declares L3 for a configuration the plan places at L4 would put
it in the wrong step of an H1a chain, and a spaCy pipeline loaded as the
published one where the plan names the one trained on EWT would put SD
relations in a UD row. Added 23 Oct 2026 (version 1.0, step 4), for the
stages of step 5.

A model this project trains (udpipe_ewt, nltk_ewt, nltk_arc, spacy_ewt,
rdr_ewt) is never trained here or in a run: its training script writes it
before the lock, and its wrapper's load_model refuses a missing file, naming
the script (base.require_trained). The wrapper declares the files
(BaseAnalyzer.trained_model, given the plan's `load`); trained_model and
trained_models below read that declaration for the plan, which is what
plan/LOCK.json records (seal.model_digests), and build() hands it to
`before_load`, with which a confirmatory stage checks the files against the
lock right before each load (seal.verify_model). Since 24 Oct 2026.

Every other file a configuration's load reads was downloaded: the Stanza
models, ELECTRA-large, benepar_en3, SuPar's crf-con-en, en_core_web_sm, the
tagger NLTK ships and the treebank sample, and the data files a library
reads from its own package directory when the load imports it (emoji for
Stanza, spaCy and thinc for spaCy; since 25 Oct 2026). The wrapper declares
those too (BaseAnalyzer.downloaded_files, given the plan's `load`):
downloaded_files and downloaded_models below read that declaration, which
plan/LOCK.json records (seal.downloaded_digests), and build() hands it to
`before_load` beside the trained model, so the same check covers both
(since 24 Oct 2026, before the lock). downloaded_files adds to every
configuration the data files of the libraries every stage imports before it
builds one (STAGE_LIBRARIES: NLTK and scikit-learn; since 25 Oct 2026).

No model library is imported here, and none when this module is imported:
the wrapper module is imported by build(), and the wrapper imports its
library in load_model.
"""
from __future__ import annotations

import importlib
from types import MappingProxyType
from typing import Any, Callable, Mapping, MutableMapping

from syntactic_analysis.base import (
    AnalysisType,
    AnnotationScheme,
    BaseAnalyzer,
    TrainedModel,
    TrainingDomain,
)
from syntactic_analysis.downloads import DownloadedFiles, library_data
from syntactic_analysis.plan import Configuration, Plan, PlanError

__all__ = [
    "STAGE_LIBRARIES",
    "TASK_TYPES",
    "DeclarationError",
    "BeforeLoad",
    "build",
    "check_declarations",
    "downloaded_files",
    "downloaded_models",
    "expected_domain",
    "expected_scheme",
    "trained_model",
    "trained_models",
]

# What build() calls right before it loads a configuration: the key, the
# model this project trains that the load will read (None for a published
# one), and the downloaded files it will read (() for none; since 24 Oct 2026).
BeforeLoad = Callable[[str, "TrainedModel | None", "tuple[DownloadedFiles, ...]"], None]

# The libraries every stage imports before it builds a configuration, so that
# every configuration's run reads their data files (downloads.LIBRARY_DATA):
# stage.run_stage imports the evaluation package, whose evalb.py imports
# nltk, and nltk imports scikit-learn (nltk/classify/scikitlearn.py).
# downloaded_files declares those files for every configuration (25 Oct 2026,
# before the lock).
STAGE_LIBRARIES: tuple[str, ...] = ("nltk", "sklearn")

# Each task code of the plan against the analysis type it stands for. Until
# 23 Oct 2026 evaluation/axes.py held the same table, for the stages before
# version 1.0, and a test kept the two equal.
TASK_TYPES: Mapping[str, AnalysisType] = MappingProxyType({
    "pos": AnalysisType.POS_TAGGING,
    "dep": AnalysisType.DEPENDENCY_PARSING,
    "con": AnalysisType.CONSTITUENCY_PARSING,
})

# The training domain a training-data family implies. TrainingDomain is
# defined against the EWT test split: IN is its own training split (any
# release) and nothing else, IN_PLUS that split with other treebanks, OUT
# anything else. The plan groups the releases of one corpus by `family`, so
# the two families that hold the EWT training split are named here; every
# other family is OUT. A new family that holds the EWT is therefore OUT until
# it is added here, and a wrapper that declares it IN is refused rather than
# believed.
_FAMILY_DOMAINS: Mapping[str, TrainingDomain] = MappingProxyType({
    "ud_english_ewt": TrainingDomain.IN,
    "ud_english_combined": TrainingDomain.IN_PLUS,
})

# The annotation scheme a wrapper must declare for a task, by the annotation
# of the configuration's training data (training_data.<id>.annotation). A
# pair missing here is refused: the plan states something no rule reads.
#
#   pos  UD for a tagger trained on UPOS. PTB for one trained on PTB tags,
#        whose wrapper converts them to UPOS (nltk_pre). OntoNotes 5 is tagged
#        with PTB tags too, but a published spaCy pipeline maps them to UPOS
#        itself (its attribute_ruler component), so what the wrapper reads is
#        UPOS and it declares UD.
#   dep  UD for UD relations; SD for OntoNotes 5, whose relations are the
#        ClearNLP conversion, a Stanford Dependencies descendant (D3).
#   con  PTB: every constituency model is trained on Penn Treebank trees.
_SCHEMES: Mapping[str, Mapping[str, AnnotationScheme]] = MappingProxyType({
    "pos": MappingProxyType({
        "ud": AnnotationScheme.UD,
        "ptb": AnnotationScheme.PTB,
        "ontonotes": AnnotationScheme.UD,
    }),
    "dep": MappingProxyType({
        "ud": AnnotationScheme.UD,
        "ontonotes": AnnotationScheme.SD,
    }),
    "con": MappingProxyType({
        "ptb": AnnotationScheme.PTB,
    }),
})


class DeclarationError(PlanError):
    """An analyzer that is not what the plan says it is, with every difference.

    A PlanError, because the plan is the source of truth: the fix is in the
    wrapper or in the plan, and nothing may run until they agree.
    """


def expected_domain(plan: Plan, key: str) -> TrainingDomain:
    """The training domain the plan implies for configuration `key`.

    Raises:
        KeyError: If `key` is not a configuration of the plan.
    """
    config = plan.configurations[key]
    family = plan.training_data[config.training].family
    return _FAMILY_DOMAINS.get(family, TrainingDomain.OUT)


def expected_scheme(plan: Plan, key: str, task: str) -> AnnotationScheme | None:
    """The scheme the plan implies for configuration `key` in `task`.

    None when no rule maps the training data's annotation for that task;
    check_declarations reports that as a problem of its own.

    Raises:
        KeyError: If `key` is not a configuration of the plan, or `task` is
            not one of TASK_TYPES.
    """
    config = plan.configurations[key]
    annotation = plan.training_data[config.training].annotation
    return _SCHEMES[task].get(annotation)


def check_declarations(plan: Plan, key: str, analyzer: BaseAnalyzer) -> None:
    """Refuse an analyzer whose declarations differ from what the plan says.

    Checks the name once and, for every task `key` is a member of, support,
    level, training domain and scheme (see the module docstring).

    Raises:
        DeclarationError: Listing every difference found.
        KeyError: If `key` is not a configuration of the plan.
    """
    config = plan.configurations[key]
    cls = type(analyzer).__name__
    where = f"configurations.{key}"
    errors: list[str] = []
    if analyzer.name != key:
        errors.append(
            f"{where} builds {cls} named {analyzer.name!r}; an analyzer's name is "
            f"its configuration's key"
        )
    domain = expected_domain(plan, key)
    for task, spec in plan.tasks.items():
        member = spec.members.get(key)
        if member is None:
            continue
        place = f"tasks.{task}.configs.{key}"
        analysis_type = TASK_TYPES[task]
        if not analyzer.supports(analysis_type):
            errors.append(f"{place}: {cls} does not support {analysis_type.name}")
            continue
        try:
            level = analyzer.method_level_for(analysis_type)
        except LookupError:
            errors.append(f"{place}: {cls} declares no level for {analysis_type.name}")
        else:
            if level is not member.level:
                errors.append(
                    f"{place}: the plan says level {member.level.code}, "
                    f"{cls} declares {level.code}"
                )
        try:
            declared_domain = analyzer.training_domain_for(analysis_type)
        except LookupError:
            errors.append(
                f"{place}: {cls} declares no training domain for {analysis_type.name}"
            )
        else:
            if declared_domain is not domain:
                errors.append(
                    f"{place}: training {config.training!r} implies domain "
                    f"{domain.value}, {cls} declares {declared_domain.value}"
                )
        scheme = expected_scheme(plan, key, task)
        annotation = plan.training_data[config.training].annotation
        if scheme is None:
            errors.append(
                f"{place}: no scheme is defined for training annotation "
                f"{annotation!r} in task {task}"
            )
        elif analyzer.scheme_for(analysis_type) is not scheme:
            errors.append(
                f"{place}: training annotation {annotation!r} implies scheme "
                f"{scheme.name} for {task}, {cls} declares "
                f"{analyzer.scheme_for(analysis_type).name}"
            )
    if errors:
        raise DeclarationError(plan.path, errors)


def build(
    plan: Plan,
    key: str,
    *,
    load: bool = True,
    built: MutableMapping[str, BaseAnalyzer] | None = None,
    before_load: BeforeLoad | None = None,
) -> BaseAnalyzer:
    """The analyzer of configuration `key`, constructed, loaded and checked.

    Args:
        plan:  The plan the run follows (plan.load(strict=True) for a run).
        key:   A configuration of the plan that is not pending.
        load:  False constructs and checks the analyzer, and the inputs it
               needs, without calling load_model on any of them: a cheap check
               of the plan against the wrappers. An analyzer built so cannot
               analyze anything.
        built: Analyzers already built from this plan, by key. An input found
               here is reused instead of built a second time (a run that
               scores udpipe_ewt and nltk_arc loads the tagger once), and an
               input built here is added to it. The analyzer of `key` itself
               is not added; the caller decides what it keeps.
        before_load: Called with the key, the model this project trains
               that the load will read (trained_model; None for a published
               model) and the downloaded files it will read (downloaded_files)
               right before each configuration it builds is
               loaded - `key` and every input built here - and not at all
               with load=False. Whatever it raises stops the build before
               that model is loaded. A confirmatory stage passes
               seal.verify_model with its lock (since 24 Oct 2026; the
               downloaded files since the same day, before the lock).

    Raises:
        PlanError: If `key` is not in the plan, is pending, or its inputs form
            a cycle.
        DeclarationError: If the analyzer, or an input, declares other than
            the plan says (check_declarations).
        Whatever the wrapper raises when it is constructed or loaded: a model
            that does not load is not replaced by another, and a model this
            project trains that is missing is refused, never trained
            (base.ModelNotTrainedError).
        Whatever `before_load` raises, and with `before_load` whatever
            declaring the downloaded files raises (downloaded_files).
    """
    return _build(plan, key, load, {} if built is None else built, (), before_load)


def _build(
    plan: Plan,
    key: str,
    load: bool,
    built: MutableMapping[str, BaseAnalyzer],
    chain: tuple[str, ...],
    before_load: BeforeLoad | None = None,
) -> BaseAnalyzer:
    if key in chain:
        raise PlanError(
            plan.path, [f"inputs form a cycle: {' -> '.join((*chain, key))}"]
        )
    plan.require_ready([key])
    config = plan.configurations[key]
    inputs: dict[str, BaseAnalyzer] = {}
    for name, target in config.inputs.items():
        if target not in built:
            built[target] = _build(plan, target, load, built, (*chain, key), before_load)
        inputs[name] = built[target]
    cls = _wrapper_class(config)
    analyzer = cls(**_thaw(config.init), **inputs)
    check_declarations(plan, key, analyzer)
    if load:
        if before_load is not None:
            before_load(
                key, trained_model(plan, key, analyzer), downloaded_files(plan, key, analyzer),
            )
        analyzer.load_model(**_thaw(config.load))
        check_declarations(plan, key, analyzer)
    return analyzer


def trained_model(
    plan: Plan, key: str, analyzer: BaseAnalyzer | None = None,
) -> TrainedModel | None:
    """The model of configuration `key` that this project trains, or None for a published one.

    What the wrapper declares (BaseAnalyzer.trained_model) for the plan's
    `load` of the configuration: the files its load_model reads and the
    script that writes them. `analyzer` is the configuration's analyzer if
    the caller has it; otherwise it is built without loading (build(...,
    load=False)), which loads no model. Since 24 Oct 2026.

    Raises:
        PlanError, DeclarationError: As build(..., load=False) raises them.
    """
    if analyzer is None:
        analyzer = build(plan, key, load=False)
    return analyzer.trained_model(**_thaw(plan.configurations[key].load))


def trained_models(plan: Plan) -> dict[str, TrainedModel]:
    """Every configuration of the plan whose model this project trains, with its declaration.

    In the plan's order. For plan/v1.yaml: rdr_ewt, nltk_ewt, udpipe_ewt,
    spacy_ewt and nltk_arc, the models of scripts/train_rdrpos.py,
    train_nltk.py (nltk_ewt and nltk_arc), train_udpipe.py and train_spacy.py.
    Each configuration is built without loading; nothing is loaded or read.
    What plan/LOCK.json records under `models` (seal.model_digests). Since
    24 Oct 2026.
    """
    built: dict[str, BaseAnalyzer] = {}
    found: dict[str, TrainedModel] = {}
    for key in plan.configurations:
        analyzer = build(plan, key, load=False, built=built)
        model = trained_model(plan, key, analyzer)
        if model is not None:
            found[key] = model
    return found


def downloaded_files(
    plan: Plan, key: str, analyzer: BaseAnalyzer | None = None,
) -> tuple[DownloadedFiles, ...]:
    """The downloaded files the run of configuration `key` reads, root by root.

    What the wrapper declares (BaseAnalyzer.downloaded_files) for the plan's
    `load` of the configuration, and after it the data files of
    STAGE_LIBRARIES, which the stage has read before it builds the
    configuration (downloads.library_data; since 25 Oct 2026, before the
    lock - until then () for a wrapper that declares nothing). `analyzer`
    is the configuration's analyzer if the caller has it; otherwise it is
    built without loading. Nothing is loaded or downloaded. Since
    24 Oct 2026.

    Raises:
        PlanError, DeclarationError: As build(..., load=False) raises them.
        Whatever the wrapper's declaration raises (a package that is not on
            disk).
    """
    if analyzer is None:
        analyzer = build(plan, key, load=False)
    declared = analyzer.downloaded_files(**_thaw(plan.configurations[key].load))
    return (*declared, *library_data(*STAGE_LIBRARIES))


def downloaded_models(plan: Plan) -> dict[str, tuple[DownloadedFiles, ...]]:
    """Every configuration of the plan with the downloaded files its load reads.

    In the plan's order, every configuration with downloaded_files: what its
    wrapper declares, and the data files of STAGE_LIBRARIES. Until
    25 Oct 2026 a configuration whose load reads nothing downloaded had ()
    (the five trained here, whose files trained_models names); since then
    every configuration has at least the files of STAGE_LIBRARIES. What
    plan/LOCK.json records under `downloaded_models`
    (seal.downloaded_digests). Each configuration is built without loading.
    Since 24 Oct 2026.
    """
    built: dict[str, BaseAnalyzer] = {}
    found: dict[str, tuple[DownloadedFiles, ...]] = {}
    for key in plan.configurations:
        analyzer = build(plan, key, load=False, built=built)
        found[key] = downloaded_files(plan, key, analyzer)
    return found


def _wrapper_class(config: Configuration) -> type[BaseAnalyzer]:
    """The class `wrapper` names; plan.load has already checked that it exists."""
    module = importlib.import_module(f"syntactic_analysis.wrappers.{config.module}")
    return getattr(module, config.class_name)


def _thaw(value: Any) -> Any:
    """The plan's read-only value as YAML gave it: mappings as dicts, lists as lists.

    plan.load freezes every mapping into a MappingProxyType and every list into
    a tuple; a wrapper is handed what the plan file says, not those types.
    """
    if isinstance(value, Mapping):
        return {name: _thaw(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value
