"""Syntactic Analysis — public API.

Everything listed in `__all__` can be imported straight from the package:

    from syntactic_analysis import BaseAnalyzer, AnalysisType, MethodLevel

The list is the whole contract: the base class and the axis enums an analyzer
is written against (dokumentacia.txt §9), and the containers it returns. Until
23 Oct 2026 it also named `AnalyzerRegistry`, the container the stage scripts
before version 1.0 registered their analyzers in; version 1.0 builds each
configuration from plan/v1.yaml (syntactic_analysis.build), and the registry
was removed with the scripts that used it.

The evaluation subpackage and the wrappers are imported explicitly - they are
the measurement machinery and the tools, not the interface.

The tests are not in this repository. Since 9 Oct 2026 the suite is kept beside
it and runs against the installed package, so a name with no caller in src/ or
scripts/ may still be pinned there. A comment that says "the out-of-repository
test suite" means that one.
"""

from syntactic_analysis.base import (
    AnalysisType,
    AnnotationScheme,
    BaseAnalyzer,
    ContractViolation,
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

__all__ = [
    # The contract an analyzer implements
    "BaseAnalyzer",        # Abstract base class: every wrapper subclasses it
    "ContractViolation",   # Raised when output breaks the alignment contract
    # The categorisation axes (dokumentacia.txt §4)
    "AnalysisType",        # Enum: POS_TAGGING, DEPENDENCY_PARSING, CONSTITUENCY_PARSING
    "MethodType",          # Enum: RULE_BASED, STATISTICAL, NEURAL (coarse, derived)
    "MethodLevel",         # IntEnum: L0 rules .. L4 pretrained transformer, per task
    "TrainingDomain",      # Enum: in / in_plus / out / none, relative to the test set
    "AnnotationScheme",    # Enum: UD / SD / PTB / NONE, relative to the gold
    # What an analysis returns
    "Token",               # Single word with POS tag, lemma and dependency info
    "Sentence",            # List of tokens with the original sentence text
    "DependencyParse",     # Result container for dependency analysis output
    "ConstituencyParse",   # Result container for constituency analysis output
]
