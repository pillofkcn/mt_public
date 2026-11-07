"""Retrain UDPipe 1 on the current EWT training split: the model of the plan's udpipe_ewt.

Why this script exists
----------------------
The only published UDPipe 1 English model was trained on UD_English-EWT r2.5
(December 2019). The gold standard everything else here is scored against is a
much newer release, and the UD guidelines moved in between — attachment rules
for auxiliaries, copulas, `goeswith` and flat structures all changed, and the
gold test set contains relation subtypes (`obl:unmarked`, `nmod:desc`,
`nsubj:outer`) that did not exist in 2.5 at all. So a gap measured between the
published UDPipe model and a modern parser conflates two different things:
the architecture falling behind, and five years of annotation drift.

Retraining the same library on the current train split separates them. Before
version 1.0 the difference between the published and the retrained UDPipe rows
was reported as the treebank-version cost (the RQ2 contrast dep-UD-SKEW).

Version 1.0 has no published UDPipe model (it was dropped on 23 Oct 2026):
the model this script writes is the only one, the plan's udpipe_ewt (UPOS
descriptively, the L2 step of the dependency chain, and the tags of nltk_arc).

It is trained here and only here, before the plan is locked. Until 24 Oct 2026
it was also the first stage of scripts/run_pipeline.py (train_udpipe), which
trained the model when it was missing; since then a run trains nothing, and a
build that finds the model missing fails and names this script (the owner's
decision of 24 Oct 2026). plan/LOCK.json records the SHA-256 of the file.

Cost: single-threaded and slow on the full split - about 100 minutes on this
project's CPU, the cost the pipeline before version 1.0 stated for the stage
(a measured time, rounded hard; scripts/run_pipeline.py under the tag
exploracia-2026-10-17), since the training record of the model in
model_cache/ holds no time. The model is written once and reused; re-running
is a no-op unless --force is given or the cached model was trained on a
truncated split, which is retrained rather than reused.

Usage
-----
    python scripts/train_udpipe.py                     # full train split
    python scripts/train_udpipe.py --max-train-sentences 200   # smoke test
    python scripts/train_udpipe.py --force             # retrain over the cache
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from syntactic_analysis.evaluation.provenance import recorded_path, stage_started
from syntactic_analysis.evaluation.report import Reporter
from syntactic_analysis.project import ewt_split_path
from syntactic_analysis.wrappers.udpipe import RETRAINED_MODEL_PATH, UdpipeAnalyzer

DEFAULT_OUT = RETRAINED_MODEL_PATH


def record_path(model_path: Path) -> Path:
    """Where the training record for a model file lives."""
    return model_path.with_suffix(model_path.suffix + ".json")


def write_record(model_path: Path, record: dict) -> None:
    """Note how a cached model was trained, beside the model.

    Without it a cached file reports only its own size, and a model trained on a
    hundred sentences for a smoke test is indistinguishable from one trained on
    the full split — which is how a deliberately useless model came to be scored
    and reported as a treebank-version effect.
    """
    record_path(model_path).write_text(json.dumps(record, indent=2), encoding="utf-8")


def describe_cached(model_path: Path) -> str:
    """One line saying how the cached model was trained."""
    path = record_path(model_path)
    if not path.exists():
        return "unknown (no training record beside the model)"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("truncated"):
        return f"a TRUNCATED split of {record.get('train_sentences')} sentences"
    return f"the full split at {record.get('train_path', '?')}"


def is_truncated(model_path: Path) -> bool:
    """True when the cached model is known to be a smoke artefact.

    Unknown provenance counts as fine: a model built before this record existed
    is a full-split model, and refusing it would break a working cache for
    nothing.
    """
    path = record_path(model_path)
    if not path.exists():
        return False
    return bool(json.loads(path.read_text(encoding="utf-8")).get("truncated"))


def truncated_copy(source: Path, destination: Path, max_sentences: int) -> Path:
    """Write the first `max_sentences` CoNLL-U sentences of `source`.

    For smoke runs only. A model trained on a handful of sentences is useless as
    a parser, but it exercises the whole path — training, caching, loading,
    scoring — in seconds instead of the time a full training takes.

    0 keeps no sentence, as the gold loaders read 0. Until 16 Oct 2026 the count
    was tested after a sentence had been kept, so 0 kept one; the command line
    now refuses a count below 1 anyway (`parse_args`).
    """
    kept: list[str] = []
    for block in source.read_text(encoding="utf-8").split("\n\n"):
        if len(kept) >= max_sentences:
            break
        if not block.strip():
            continue
        kept.append(block)
    destination.write_text("\n\n".join(kept) + "\n\n", encoding="utf-8")
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain UDPipe 1 on the current EWT train split."
    )
    parser.add_argument(
        "--train",
        default=str(ewt_split_path("train")),
        help="CoNLL-U training file (default: the EWT train split).",
    )
    parser.add_argument(
        "--heldout",
        default=str(ewt_split_path("dev")),
        help="CoNLL-U held-out file for early stopping; ignored if absent.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Where to write the model (default: the path load_model expects).",
    )
    parser.add_argument(
        "--max-train-sentences",
        type=int,
        default=None,
        help="Train on only the first N sentences. For smoke runs; the resulting "
             "model is not a usable parser.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if the output file already exists.",
    )
    args = parser.parse_args(argv)
    # A model needs at least one training sentence; a count of 0 stops here
    # with the reason rather than inside the trainer. (Until 24 Oct 2026
    # run_pipeline.py handed its --max-sentences on as this flag.)
    if args.max_train_sentences is not None and args.max_train_sentences < 1:
        parser.error("--max-train-sentences must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = stage_started()
    train_path = Path(args.train)
    heldout_path = Path(args.heldout)
    out_path = Path(args.out)
    # The held-out set a training would use: none when the file is absent, and
    # none in a smoke run, where a held-out set is meaningless at that size.
    heldout: Path | None = (
        heldout_path
        if heldout_path.exists() and args.max_train_sentences is None
        else None
    )

    Reporter().header(
        "STAGE train_udpipe: UDPIPE RETRAINING — THE CURRENT EWT TRAINING SPLIT",
        {
            "started": started,
            "train": recorded_path(train_path),
            "heldout": recorded_path(heldout) if heldout else "(none)",
            "output": recorded_path(out_path),
        },
    )

    if not train_path.exists():
        print(f"training file not found: {train_path}", file=sys.stderr)
        print("Run setup.sh to download the treebank.", file=sys.stderr)
        return 1

    if out_path.exists() and not args.force:
        size_mb = out_path.stat().st_size / (1 << 20)
        print(f"Model already present: {recorded_path(out_path)} ({size_mb:.1f} MB)")
        print(f"  trained on: {describe_cached(out_path)}")
        if is_truncated(out_path):
            # The trap this closes: until 24 Oct 2026 the pipeline forwarded
            # --max-sentences to this script, its first stage, so a smoke run of
            # the pipeline baked a model trained on a handful of sentences into
            # the cache, and the next run reused it silently and reported it as
            # "UDPipe (retrained)" — which is how a deliberately useless model
            # came to be scored as a treebank-version effect. A smoke model can
            # still be written by hand (--max-train-sentences); retraining
            # rather than failing replaces it with a real one instead of
            # keeping it.
            print("  It is a smoke artefact, so it is being retrained rather than reused.")
        else:
            print("Nothing to do. Pass --force to retrain over it.")
            return 0

    if not heldout_path.exists():
        # Legal, just without early stopping — say so rather than failing, since
        # the dev split is not required for the comparison this model serves.
        print(f"held-out file absent ({heldout_path.name}); training without early stopping")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.max_train_sentences is not None:
        train_path = truncated_copy(
            train_path,
            out_path.parent / "english-ewt-train-truncated.conllu",
            args.max_train_sentences,
        )
        print(f"SMOKE RUN: training on the first {args.max_train_sentences} sentences only.")
        print("The resulting model is not a usable parser; it exists to exercise the path.")

    print("Training UDPipe 1 (morphodita_parsito). This is single-threaded and slow.", flush=True)

    training_started = time.perf_counter()
    UdpipeAnalyzer.train_from_conllu(train_path, out_path, heldout_path=heldout)
    elapsed = time.perf_counter() - training_started

    size_mb = out_path.stat().st_size / (1 << 20)
    write_record(
        out_path,
        {
            "train_path": recorded_path(train_path),
            "train_sentences": args.max_train_sentences,
            "truncated": args.max_train_sentences is not None,
            "heldout_path": recorded_path(heldout) if heldout else "",
            "minutes": round(elapsed / 60, 2),
        },
    )
    print(f"\nWrote {recorded_path(out_path)} ({size_mb:.1f} MB) in {elapsed / 60:.1f} min.")
    print('Load it with UdpipeAnalyzer(model="en_ewt_retrained").load_model().')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
