"""Train the two NLTK models of plan/v1.yaml on the EWT training split: nltk_ewt and nltk_arc.

Why this script exists
----------------------
Two configurations of the plan run an NLTK model this project trains:
nltk_ewt, NLTK's averaged perceptron trained on EWT train with UPOS, and
nltk_arc, NLTK's TransitionParser (arc-eager) with a logistic regression in
place of NLTK's SVC. Until 24 Oct 2026 their wrappers trained them the first
time they were loaded and model_cache/ did not hold them, so a build or a run
could train a model in the middle of a measurement. The owner decided on
24 Oct 2026 that a model this project trains is trained by a training script
only, before the plan is locked: a build or a run that finds it missing
fails and names the script (base.require_trained), and plan/LOCK.json
records the SHA-256 of its files (seal.model_digests). This is that script
for the two NLTK models; scripts/train_udpipe.py, train_spacy.py and
train_rdrpos.py are the ones for udpipe_ewt, spacy_ewt and rdr_ewt.

What it does
------------
It takes from the plan the configurations whose model this script trains
(build.trained_models: the wrapper's declaration names this script) and
trains each with the wrapper's own training code, the code the wrapper ran
at load until 24 Oct 2026, unchanged:

  nltk_ewt  NltkAnalyzer.train_ewt_tagger: PerceptronTagger, five
            iterations, Python's random generator seeded with
            _PERCEPTRON_SEED (1), on the UPOS of the training split; three
            json files in <out>/nltk_perceptron_ud_ewt_train/.
  nltk_arc  NltkTransitionAnalyzer.train_model with the plan's `load`
            (model_name arc-eager): the oracle's transitions of the
            projective training sentences, a LogisticRegression (lbfgs,
            random_state 0); <out>/nltk_transition-arc-eager.pkl and
            .classifier.pkl.

It reads the training split only (data/en_ewt-ud-train.conllu; no dev, no
test). A model whose files are all there is left alone unless --force is
given; one with a file missing is trained again, every file of it. --out-dir
writes somewhere else than model_cache/, where no wrapper reads it: that is
how a retraining is compared with the models in model_cache/ without
touching them.

The hash seed
-------------
NLTK writes the classes file of nltk_ewt as a list made from a set of
strings, so the order of the classes in it is the order Python's string
hashing gives the set, and Python seeds that hashing anew in every process.
Since 24 Oct 2026 (the owner's decision) the training runs with
PYTHONHASHSEED fixed to HASH_SEED (20260101), the value
scripts/train_rdrpos.py gives its learner. Python reads the variable only
when the interpreter starts, so setting it in os.environ of the running
process would change nothing: started without it, or with another value,
the script runs itself again in a child process whose environment has it,
with the same arguments, and exits with the child's status (`run`). The
header it prints names the hash seed of the process that trains.

What a retraining does not repeat byte for byte (24 Oct 2026, comparing
retrainings with model_cache/): the classes file of nltk_ewt, whose order
follows the string hashing of the process that trained it (two retrainings
with the fixed hash seed write the same file; the one in model_cache/ was
trained before the seed was fixed and has the same 17 classes in another
order; the weights and the tag dictionary are identical), and the pickle of
nltk_arc, which records the training times (the classifier file, the feature
dictionary and the fitted coefficients are identical). Neither changes the
model: a retraining with the fixed seed and the models in model_cache/ gave
the same UPOS on all 25 094 words of the 2077 sentences of EWT test, and the
same heads and relations on all 25 094 arcs, the two parsers given the same
tags of udpipe_ewt (CHANGELOG_v1.md, third round of the owner's answers). So
a retraining is checked by its outputs, not by its bytes; the lock records
the files that are in model_cache/.

Cost: nltk_ewt about 25 s and nltk_arc about 6 min on this project's CPU
(CHANGELOG_v1.md, 24 Oct 2026); the time depends on the load of the machine
and is printed, not promised.

Usage
-----
    python scripts/train_nltk.py                    # both, into model_cache/
    python scripts/train_nltk.py --key nltk_arc     # one (repeatable)
    python scripts/train_nltk.py --force            # train again over model_cache/
    python scripts/train_nltk.py --out-dir DIR      # into DIR instead
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from syntactic_analysis.build import build, trained_models
from syntactic_analysis.checksum import sha256_of
from syntactic_analysis.evaluation.provenance import recorded_path, stage_started
from syntactic_analysis.evaluation.report import Reporter
from syntactic_analysis.plan import load
from syntactic_analysis.project import ewt_split_path, model_cache_dir
from syntactic_analysis.wrappers.nltk import NltkAnalyzer
from syntactic_analysis.wrappers.nltk_transition import NltkTransitionAnalyzer

# How the wrappers name this script in their declarations (TrainedModel.script).
SCRIPT = f"scripts/{Path(__file__).name}"

# The seed of Python's string hashing in the process that trains (see "The
# hash seed" above): the value scripts/train_rdrpos.py gives its learner
# (HASH_SEED there), the project's seed.
HASH_SEED = 20260101


def hash_seed_pinned() -> bool:
    """Whether this interpreter was started with PYTHONHASHSEED=HASH_SEED.

    Python reads the variable once, when the interpreter starts, and not at
    all under -E or -I; os.environ shows what it was started with, as long
    as nothing in the process has changed it.
    """
    return (not sys.flags.ignore_environment
            and os.environ.get("PYTHONHASHSEED") == str(HASH_SEED))


def pinned_environment() -> dict[str, str]:
    """This process's environment with PYTHONHASHSEED set to HASH_SEED."""
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = str(HASH_SEED)
    return environment


def run(argv: list[str] | None = None) -> int:
    """The command line: main() in an interpreter started with PYTHONHASHSEED=HASH_SEED.

    Setting PYTHONHASHSEED in os.environ of a running process changes no
    hash of that process, so when this interpreter was not started with it
    (or was started with another value), the script runs again in a child
    process whose environment has it - the same interpreter, the same
    arguments, otherwise the same environment - and the child's exit status
    is returned. Nothing is trained in this process then.
    """
    arguments = sys.argv[1:] if argv is None else list(argv)
    if hash_seed_pinned():
        return main(arguments)
    current = os.environ.get("PYTHONHASHSEED", "not set")
    if sys.flags.ignore_environment:
        current += ", ignored (-E or -I)"
    print(f"PYTHONHASHSEED {current}: running {SCRIPT} again with "
          f"PYTHONHASHSEED={HASH_SEED}", flush=True)
    command = [sys.executable, str(Path(__file__).resolve()), *arguments]
    return subprocess.run(command, env=pinned_environment(), check=False).returncode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the NLTK models of plan/v1.yaml (nltk_ewt, nltk_arc) on the "
                    "EWT train split, with the wrappers' own training code."
    )
    parser.add_argument(
        "--key", action="append", default=None, metavar="KEY",
        help="a configuration whose model this script trains (repeatable; default: all of them)",
    )
    parser.add_argument(
        "--train", type=Path, default=ewt_split_path("train"),
        help="CoNLL-U training file (default: the EWT train split).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None, metavar="DIR",
        help="where to write the models (default: model_cache/, where the wrappers read them)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="train again over models whose files are all there",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = stage_started()
    out_dir = args.out_dir if args.out_dir is not None else model_cache_dir()
    plan = load(strict=True)
    targets = [key for key, model in trained_models(plan).items() if model.script == SCRIPT]
    keys = targets if args.key is None else args.key
    unknown = [key for key in keys if key not in targets]
    if unknown:
        print(f"{unknown}: not a configuration whose model {SCRIPT} trains; "
              f"expected some of {targets}", file=sys.stderr)
        return 2

    Reporter().header(
        "TRAINING THE NLTK MODELS OF THE PLAN ON THE EWT TRAINING SPLIT",
        {
            "started": started,
            "configurations": ", ".join(keys),
            "train": recorded_path(args.train),
            "output": recorded_path(out_dir),
            "hash seed": f"PYTHONHASHSEED={HASH_SEED}" if hash_seed_pinned()
                         else "not fixed (started without PYTHONHASHSEED="
                              f"{HASH_SEED}; the command line fixes it)",
        },
    )
    if not args.train.exists():
        print(f"training file not found: {args.train}", file=sys.stderr)
        print("Run setup.sh to download the treebank.", file=sys.stderr)
        return 1

    for key in keys:
        load_options = dict(plan.configurations[key].load)
        analyzer = build(plan, key, load=False)
        files = analyzer.trained_model(**load_options, cache_dir=out_dir).files
        if all(path.is_file() for path in files) and not args.force:
            print(f"\n{key}: already present ({', '.join(recorded_path(f) for f in files)}). "
                  f"Nothing to do; pass --force to train again over it.")
            continue
        print(f"\n{key}: training ({plan.configurations[key].wrapper})", flush=True)
        clock = time.perf_counter()
        if isinstance(analyzer, NltkAnalyzer):
            written = NltkAnalyzer.train_ewt_tagger(train_path=args.train, cache_dir=out_dir)
        elif isinstance(analyzer, NltkTransitionAnalyzer):
            written = NltkTransitionAnalyzer.train_model(
                **load_options, train_path=args.train, cache_dir=out_dir,
            )
        else:  # a declaration naming this script on a wrapper it does not know
            print(f"{key}: {type(analyzer).__name__} is not a wrapper {SCRIPT} trains",
                  file=sys.stderr)
            return 1
        elapsed = time.perf_counter() - clock
        print(f"{key}: trained in {elapsed:.0f} s")
        for path in written:
            print(f"  {recorded_path(path)}  sha256 {sha256_of(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
