"""The dependency stage: per-sentence attachment counts of every dependency configuration.

    python scripts/dependency_parsing.py                   # EWT test, every configuration
    python scripts/dependency_parsing.py --max-sentences 20 --out-dir results/smoke/dep
    python scripts/dependency_parsing.py --keys spacy_sm,spacy_ewt
    python scripts/dependency_parsing.py --dataset gentle --confirmatory   # after the lock only

Runs the configurations plan/v1.yaml makes members of the task dep and scores
on the dataset (spacy_sm on EWT test only, since 24 Oct 2026), in the plan's
order, on a dataset the plan lists for the task (default: its
exploratory one, the EWT test split), each built by build.build and given the
gold tokens one sentence at a time. nltk_arc takes its tags from udpipe_ewt,
which build.build builds once for both. Writes
<out-dir>/csv/counts_dep_<dataset>.csv (columns config, dataset, doc_id,
sent_id, index, n_tokens, las_correct, uas_correct, total, clas_correct,
clas_total, clas_pred_total; CoNLL 2018 conventions, relation subtypes
ignored on both sides) and its provenance beside it, and prints a summary
table. The dataset is opened through the seal before any model is built.

Everything is in syntactic_analysis.stage, whose docstring states the counts
file and the provenance. Until 23 Oct 2026 this script was the measuring and
reporting stage of the earlier design (mapping tiers, tests, timings); that
code is in the history and under the tag exploracia-2026-10-17.
"""
from __future__ import annotations

import sys

from syntactic_analysis.stage import cli

if __name__ == "__main__":
    sys.exit(cli("dep", script="scripts/dependency_parsing.py",
                 description=__doc__.splitlines()[0]))
