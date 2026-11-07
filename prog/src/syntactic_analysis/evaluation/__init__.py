"""Evaluation: gold corpora, metrics, estimates, provenance, output.

Eight modules, each with one job. Read them in this order:

  gold        where a reference annotation comes from, and what its checksum
              is: the UD EWT splits and GENTLE (CoNLL-U), the WSJ sample
  ptb         the Penn Treebank sample behind the WSJ gold set: its trees,
              cleaned, its test section and its checksum
  mctb        MCTB_en, the confirmatory constituency gold set, read into the
              same container as the WSJ trees
  metrics     the per-sentence counts of POS and dependencies, and
              corpus_metric, which pools the per-sentence counts of every task
  evalb       the EVALB parameter system and the per-sentence bracket counts
  estimate    the one statistical procedure of version 1.0: the paired
              cluster bootstrap, the intervals of a difference and of a
              difference of differences, and classify
  provenance  what produced a number: versions, commit, checksum
  report      the console helpers the scripts share, and the banner of a
              training script

The measuring stages themselves are syntactic_analysis.stage, and the analysis
is scripts/analyze.py.

Until 9 Oct 2026 five of the modules of that time were one module called
`harness`. Version 1.0 (23 Oct 2026) removed nine that served the stages,
statistics and figures before it: measure, constituency_scoring, stats, axes,
design, tables, tree_dumps, inputs and comparator.

This file re-exports only the counts and the parameter sets a caller is likely
to want by name. Everything else is imported from its own module, which is
also how the scripts do it: a name should say which job it belongs to.
"""

from syntactic_analysis.evaluation.evalb import (
    COLLINS,
    COLLINS_NML,
    ConCounts,
    EvalbParams,
    con_sentence_counts,
    plan_parameter_set,
)
from syntactic_analysis.evaluation.metrics import (
    CONTENT_DEPRELS,
    FUNCTIONAL_DEPRELS,
    DepCounts,
    PosCounts,
    corpus_metric,
    dep_sentence_counts,
    pos_sentence_counts,
)

__all__ = [
    # Per-sentence counts and the corpus metric
    "PosCounts",
    "DepCounts",
    "ConCounts",
    "pos_sentence_counts",
    "dep_sentence_counts",
    "con_sentence_counts",
    "corpus_metric",
    # The CoNLL 2018 relation sets
    "CONTENT_DEPRELS",
    "FUNCTIONAL_DEPRELS",
    # The EVALB parameter sets
    "plan_parameter_set",
    "EvalbParams",
    "COLLINS",
    "COLLINS_NML",
]
