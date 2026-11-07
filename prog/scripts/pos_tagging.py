"""The POS stage: per-sentence UPOS counts of every POS configuration of the plan.

    python scripts/pos_tagging.py                          # EWT test, every configuration
    python scripts/pos_tagging.py --max-sentences 20 --out-dir results/smoke/pos
    python scripts/pos_tagging.py --keys nltk_ewt,stanza_ewt
    python scripts/pos_tagging.py --dataset gentle --confirmatory   # after the lock only

Runs the configurations plan/v1.yaml makes members of the task pos and scores
on the dataset (nltk_pre and spacy_sm on EWT test only, since 24 Oct 2026), in
the plan's order, on a dataset the plan lists for the task (default: its
exploratory one, the EWT test split), each built by build.build and given the
gold tokens one sentence at a time. Writes <out-dir>/csv/counts_pos_<dataset>.csv
(columns config, dataset, doc_id, sent_id, index, n_tokens, correct, total) and
its provenance beside it, and prints a summary table. The dataset is opened
through the seal before any model is built: a confirmatory one is refused
without --confirmatory and a plan/LOCK.json that holds.

Everything is in syntactic_analysis.stage, whose docstring states the counts
file and the provenance. Until 23 Oct 2026 this script was the measuring and
reporting stage of the earlier design (level x domain grid, tests, timings);
that code is in the history and under the tag exploracia-2026-10-17.
"""
from __future__ import annotations

import sys

from syntactic_analysis.stage import cli

if __name__ == "__main__":
    sys.exit(cli("pos", script="scripts/pos_tagging.py", description=__doc__.splitlines()[0]))
