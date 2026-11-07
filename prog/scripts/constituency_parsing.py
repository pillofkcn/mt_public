"""The constituency stage: per-sentence bracket counts of every constituency configuration.

    python scripts/constituency_parsing.py                 # WSJ section 01, every configuration
    python scripts/constituency_parsing.py --max-sentences 20 --out-dir results/smoke/con
    python scripts/constituency_parsing.py --keys stanza_con,benepar
    python scripts/constituency_parsing.py --gold mctb --confirmatory   # after the lock only

--gold names the treebank: wsj01 (plan dataset wsj_01, exploratory, the
default) or mctb (mctb_en, confirmatory). There is no other: the EWT mode of
the earlier design, which measured agreement between parsers without a gold
standard, is gone.

Runs the configurations plan/v1.yaml makes members of the task con, in the
plan's order, each built by build.build and given the gold tokens (with their
PTB escapes) one sentence at a time, and scores every tree under the EVALB
parameter set the plan's constituency metric names. The PCFG configurations of
D2 run only on the D2 sample of the dataset (descriptive.d2.sample, drawn over
all its sentences; a sentence over max_tokens is declined by the wrapper and
recorded as over_length); every other configuration runs on every sentence.
Writes <out-dir>/csv/counts_con_<dataset>.csv (columns config, dataset, doc_id,
sent_id, index, n_tokens, matched, gold, test, parsed, evalb_error, failure,
d2_sample) and its provenance beside it, and prints a summary table. The
dataset is opened through the seal before any model is built.

Everything is in syntactic_analysis.stage, whose docstring states the counts
file, the failure codes, the D2 sample and the provenance. Until 23 Oct 2026
this script was the measuring and reporting stage of the earlier design
(agreement on the EWT, gold F1 on WSJ 01, tests, tree dumps, timings); that
code is in the history and under the tag exploracia-2026-10-17.
"""
from __future__ import annotations

import sys

from syntactic_analysis.stage import cli

if __name__ == "__main__":
    sys.exit(cli("con", script="scripts/constituency_parsing.py",
                 description=__doc__.splitlines()[0]))
