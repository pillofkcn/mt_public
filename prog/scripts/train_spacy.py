"""Train a spaCy pipeline on the EWT training split: the plan's `spacy_ewt`.

Why this stage exists
---------------------
spaCy's English pipelines (en_core_web_*) are trained on OntoNotes 5 with
dependencies in the ClearNLP conversion, a descendant of Stanford
Dependencies. Scored against the UD gold, en_core_web_sm differs from the
UD-trained tools in two things at once: its training data and its annotation
scheme. The pipeline trained here keeps the architecture and changes the data:
the same library and the same kind of model (a CNN tok2vec without pretrained
vectors, so L2), trained on UD_English-EWT train, the corpus every in-domain
configuration of the plan is trained on. H2a compares the two (`pos_spacy`,
`dep_spacy`), and D4 compares this pipeline's UPOS with the NLTK perceptron
trained on the same split.

What it does, with spaCy's own command line and nothing of this project's in
between:

  1. `spacy convert` the train and dev CoNLL-U files of data/ into spaCy's
     binary format, one sentence per document. The wrapper gives the pipeline
     one gold sentence per Doc, and a pipeline trained on documents of one
     sentence learns no sentence boundary inside one. Multiword-token range
     lines and empty nodes are skipped by the converter, so the pipeline is
     trained on the syntactic words the gold is scored on.
  2. `spacy init config --lang en --pipeline morphologizer,parser --optimize
     efficiency`, the generated config unchanged: a tok2vec of MultiHashEmbed
     and a CNN encoder (no static vectors), the morphologizer (UPOS together
     with FEATS) and the transition-based parser.
  3. `spacy train` on the CPU (--gpu-id -1) with a fixed seed (SEED), the dev
     split as spaCy's evaluation set: it keeps the best-scoring pipeline in
     model-best/. The test split is not read. Besides the seed and the device,
     the command line sets one value of the config, for both corpora
     (OVERRIDES): gold_preproc = true. spaCy's default (false) re-tokenizes the
     text of every training and dev sentence with its own tokenizer and aligns
     the result to the gold words; with true the pipeline sees the gold words
     themselves. Every configuration of the plan is given the gold words, and
     the wrapper gives them to this pipeline the same way (a Doc built from the
     word list), so it is trained, and model-best is chosen, on the input it
     is scored on.
  4. Writes training_record.json beside the pipeline: the commands, the
     versions, the seed, the digests of the data and of the config, the
     training settings read from the config, the training time, spaCy's table
     of evaluations (the step of the best one, the last step and why training
     stopped), the best dev scores as spaCy reports them, a digest of
     model-best, and the dev scores of model-best in this project's
     conventions (`project_dev_scores`, through the wrapper and
     evaluation.metrics).

spaCy's dev scores are not the plan's metrics: its parser scorer leaves
punctuation out of UAS and LAS and compares relations with their subtypes,
where the plan's LAS counts punctuation and strips subtypes on both sides
(CoNLL 2018). Both are recorded, under their own names.

The output goes to model_cache/spacy_ewt/, which git ignores. The wrapper
loads model_cache/spacy_ewt/model-best (SpacyAnalyzer(model="ewt_train")) and
refuses it without the training record.

Cost: about an hour on a CPU (spaCy's default schedule: at most 20 000 steps,
stopping after 1 600 without improvement). The time depends on the load of
the machine and is recorded, not promised.

Training starts from nothing every time: what an earlier run left in
model_cache/spacy_ewt (the converted corpus, the config, model-best/,
model-last/, the record and the log) is removed first, so a run that stops
half-way cannot leave a pipeline beside a record of another one.

Usage
-----
    python scripts/train_spacy.py            # convert, configure, train, record
    python scripts/train_spacy.py --force    # train again over model_cache/spacy_ewt
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from syntactic_analysis.checksum import sha256_of
from syntactic_analysis.evaluation.provenance import recorded_path, stage_started
from syntactic_analysis.evaluation.report import Reporter
from syntactic_analysis.project import ewt_split_path
from syntactic_analysis.wrappers.spacy import (
    EWT_PIPELINE_DIR,
    EWT_TRAINING_RECORD,
    EWT_WORK_DIR,
    pipeline_digest,
)

# The training seed, spaCy's [system] seed. One value, set on the command line
# of `spacy train` and recorded.
SEED = 20260101

# What `spacy init config` is asked for.
PIPELINE = "morphologizer,parser"
OPTIMIZE = "efficiency"

# Sentences per converted document (`spacy convert --n-sents`); see step 1.
SENTENCES_PER_DOC = 1

# The config values `spacy train` is given on its command line besides the
# seed and the device; see step 3 of the module docstring.
OVERRIDES = {
    "corpora.train.gold_preproc": "true",
    "corpora.dev.gold_preproc": "true",
}

# A row of spaCy's console table: epoch, step, then numbers, the last the score.
_TABLE_ROW = re.compile(r"^\s*(\d+)\s+(\d+)((?:\s+-?\d+(?:\.\d+)?)+)\s*$")


def _spacy(*arguments: str) -> list[str]:
    """A spaCy command line run by this interpreter."""
    return [sys.executable, "-m", "spacy", *arguments]


def _shown(command: list[str]) -> str:
    """A command as the record keeps it: `python -m spacy ...`, repository paths relative."""
    shown = ["python"]
    for part in command[1:]:
        shown.append(recorded_path(part) if Path(part).is_absolute() else part)
    return " ".join(shown)


def run(command: list[str], log) -> None:
    """Run one spaCy command, echo its output and append it to the log.

    Raises:
        RuntimeError: If the command exits with a non-zero status.
    """
    print(f"$ {_shown(command)}", flush=True)
    log.write(f"$ {_shown(command)}\n")
    # UTF-8 both ways, and no colour codes in the log.
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", ANSI_COLORS_DISABLED="1")
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        encoding="utf-8", errors="replace", env=env,
    ) as process:
        for line in process.stdout:
            # The log keeps spaCy's output as it is; a console that cannot
            # show a character (a Windows code page and spaCy's symbols) gets a
            # replacement instead of an error.
            encoding = sys.stdout.encoding or "utf-8"
            print(line.encode(encoding, "replace").decode(encoding), end="", flush=True)
            log.write(line)
            log.flush()
    if process.returncode != 0:
        raise RuntimeError(f"{_shown(command)} exited with status {process.returncode}")


def corpus_size(path: Path) -> dict:
    """Documents and tokens of a converted .spacy file."""
    import spacy
    from spacy.tokens import DocBin

    docs = list(DocBin().from_disk(path).get_docs(spacy.blank("en").vocab))
    return {"docs": len(docs), "tokens": sum(len(doc) for doc in docs)}


def spacy_dev_scores(pipeline_dir: Path) -> dict:
    """The best dev scores as spaCy saved them in model-best/meta.json."""
    meta = json.loads((pipeline_dir / "meta.json").read_text(encoding="utf-8"))
    performance = meta.get("performance", {})
    keys = ("pos_acc", "morph_acc", "dep_uas", "dep_las", "sents_f")
    return {key: performance.get(key) for key in keys}


def training_settings(config_path: Path) -> dict:
    """The schedule of `spacy train` as the generated config sets it."""
    from thinc.api import Config

    training = Config().from_disk(config_path, interpolate=False)["training"]
    keys = ("max_steps", "max_epochs", "patience", "eval_frequency", "dropout")
    return {
        **{key: training[key] for key in keys},
        "learn_rate": training["optimizer"]["learn_rate"],
        "score_weights": dict(training["score_weights"]),
    }


# The dev scores in spaCy's table, counted from the end of a row: POS_ACC,
# MORPH_ACC, DEP_UAS and DEP_LAS come before SENTS_F and SCORE.
_TABLE_SCORES = {"pos_acc": -6, "morph_acc": -5, "dep_uas": -4, "dep_las": -3}


def training_table(log_text: str, settings: dict, best_scores: dict) -> dict:
    """spaCy's table of evaluations on dev, read from its console output.

    One row per evaluation: epoch, step, the three losses, the dev scores in
    points and the weighted score (0-1) spaCy keeps model-best by, each printed
    with two decimals. Training stops at max_steps or, before that, when
    `patience` steps have gone by without a better score.

    The row of model-best is the last one whose four printed dev scores are
    model-best's own (`best_scores`, from its meta.json) rounded the same way:
    spaCy saves model-best whenever a score equals the best so far, and the
    printed weighted score alone ties too often to say which row it was.
    """
    rows = []
    for line in log_text.splitlines():
        match = _TABLE_ROW.match(line)
        if match:
            values = [float(value) for value in match.group(3).split()]
            rows.append((int(match.group(1)), int(match.group(2)), values))
    if not rows:
        return {"evaluations": 0}
    best_rows = [
        row for row in rows
        if all(
            best_scores.get(key) is not None
            and f"{row[2][index]:.2f}" == f"{best_scores[key] * 100:.2f}"
            for key, index in _TABLE_SCORES.items()
        )
    ]
    last = rows[-1]
    best_step = best_rows[-1][1] if best_rows else None
    if last[1] >= settings["max_steps"] > 0:
        stopped_by = "max_steps"
    elif best_step is not None and last[1] - best_step >= settings["patience"] > 0:
        stopped_by = "patience"
    else:
        stopped_by = "neither max_steps nor patience"
    return {
        "evaluations": len(rows),
        "last_epoch": last[0],
        "last_step": last[1],
        "best_step": best_step,
        "best_score": best_rows[-1][2][-1] if best_rows else None,
        "stopped_by": stopped_by,
    }


def clear_outputs(work: Path) -> None:
    """Remove what an earlier run left in the working directory (see the docstring)."""
    for name in ("corpus", "model-best", "model-last"):
        if (work / name).is_dir():
            shutil.rmtree(work / name)
    for name in ("config.cfg", "train.log", EWT_TRAINING_RECORD.name):
        (work / name).unlink(missing_ok=True)


def project_dev_scores(dev_path: Path) -> dict:
    """model-best on the dev split, through the wrapper, in the plan's conventions.

    UPOS accuracy, and LAS, UAS and CLAS as conll18_ud_eval.py counts them
    (punctuation included, subtypes stripped on both sides), each the
    corpus micro-average in points. And the sentences that came back with
    more than one root: the wrapper leaves sentence starts unpinned, and a
    sentence the pipeline splits has one root per piece.
    """
    from syntactic_analysis.evaluation.gold import load_gold_pretok
    from syntactic_analysis.evaluation.metrics import (
        corpus_metric,
        dep_sentence_counts,
        pos_sentence_counts,
    )
    from syntactic_analysis.wrappers.spacy import SpacyAnalyzer

    gold = load_gold_pretok(dev_path)
    analyzer = SpacyAnalyzer(model="ewt_train")
    analyzer.load_model()
    parse = analyzer.analyze_dependency(gold.pretok)
    pairs = list(zip(parse.sentences, gold.parse.sentences))
    pos = [pos_sentence_counts(p, g) for p, g in pairs]
    dep = [dep_sentence_counts(p, g) for p, g in pairs]
    return {
        "sentences": gold.n_sentences,
        "tokens": gold.n_tokens,
        "upos": round(corpus_metric(pos), 2),
        "las": round(corpus_metric(dep, "las"), 2),
        "uas": round(corpus_metric(dep, "uas"), 2),
        "clas": round(corpus_metric(dep, "clas"), 2),
        "multi_root_sentences": sum(
            sum(token.head_index == 0 for token in sentence.tokens) > 1
            for sentence in parse.sentences
        ),
        "model_version": analyzer.model_version,
    }


def write_record(record: dict) -> None:
    EWT_TRAINING_RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a spaCy pipeline (morphologizer, parser) on the EWT train split."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Convert, configure and train again over model_cache/spacy_ewt.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = stage_started()
    train_path = ewt_split_path("train")
    dev_path = ewt_split_path("dev")
    work = EWT_WORK_DIR
    corpus_dir = work / "corpus"
    config_path = work / "config.cfg"
    train_spacy = corpus_dir / f"{train_path.stem}.spacy"
    dev_spacy = corpus_dir / f"{dev_path.stem}.spacy"

    Reporter().header(
        "STAGE train_spacy: SPACY PIPELINE TRAINED ON THE EWT TRAINING SPLIT",
        {
            "started": started,
            "train": recorded_path(train_path),
            "dev": recorded_path(dev_path),
            "output": recorded_path(work),
            "seed": SEED,
        },
    )

    for path in (train_path, dev_path):
        if not path.exists():
            print(f"not found: {path}. Run setup.sh to download the treebank.", file=sys.stderr)
            return 1

    if EWT_PIPELINE_DIR.exists() and EWT_TRAINING_RECORD.exists() and not args.force:
        record = json.loads(EWT_TRAINING_RECORD.read_text(encoding="utf-8"))
        print(f"Pipeline already present: {recorded_path(EWT_PIPELINE_DIR)}")
        if "project_dev_scores" in record:
            print("Nothing to do. Pass --force to train again over it.")
            return 0
        print("Its record has no project_dev_scores yet; scoring model-best on dev.")
        record["project_dev_scores"] = project_dev_scores(dev_path)
        write_record(record)
        print(json.dumps(record["project_dev_scores"], indent=2))
        return 0

    import spacy
    import thinc

    clear_outputs(work)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    commands = {
        "convert_train": _spacy(
            "convert", str(train_path), str(corpus_dir), "--converter", "conllu",
            "--n-sents", str(SENTENCES_PER_DOC), "--file-type", "spacy",
        ),
        "convert_dev": _spacy(
            "convert", str(dev_path), str(corpus_dir), "--converter", "conllu",
            "--n-sents", str(SENTENCES_PER_DOC), "--file-type", "spacy",
        ),
        "init_config": _spacy(
            "init", "config", str(config_path), "--lang", "en",
            "--pipeline", PIPELINE, "--optimize", OPTIMIZE, "--force",
        ),
        "train": _spacy(
            "train", str(config_path), "--output", str(work),
            "--paths.train", str(train_spacy), "--paths.dev", str(dev_spacy),
            "--system.seed", str(SEED), "--gpu-id", "-1",
            *(part for name, value in OVERRIDES.items() for part in (f"--{name}", value)),
        ),
    }

    log_path = work / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        for step in ("convert_train", "convert_dev", "init_config"):
            run(commands[step], log)
        print("Training on the CPU. spaCy prints a row per evaluation on dev.", flush=True)
        training_started = time.perf_counter()
        run(commands["train"], log)
        elapsed = time.perf_counter() - training_started

    settings = training_settings(config_path)
    best_scores = spacy_dev_scores(EWT_PIPELINE_DIR)
    record = {
        "started": started,
        "spacy_version": spacy.__version__,
        "thinc_version": thinc.__version__,
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": platform.processor(),
        "device": "cpu",
        "seed": SEED,
        "pipeline": PIPELINE,
        "optimize": OPTIMIZE,
        "sentences_per_doc": SENTENCES_PER_DOC,
        "overrides": dict(OVERRIDES),
        "training_settings": settings,
        "commands": {step: _shown(command) for step, command in commands.items()},
        "train_path": recorded_path(train_path),
        "train_sha256": sha256_of(train_path),
        "train_corpus": corpus_size(train_spacy),
        "dev_path": recorded_path(dev_path),
        "dev_sha256": sha256_of(dev_path),
        "dev_corpus": corpus_size(dev_spacy),
        "config_path": recorded_path(config_path),
        "config_sha256": sha256_of(config_path),
        "training_minutes": round(elapsed / 60, 2),
        "training_table": training_table(
            log_path.read_text(encoding="utf-8"), settings, best_scores
        ),
        "pipeline_dir": recorded_path(EWT_PIPELINE_DIR),
        "pipeline_digest": pipeline_digest(EWT_PIPELINE_DIR),
        "log": recorded_path(log_path),
        # spaCy's scorer: punctuation out of UAS and LAS, relation subtypes kept.
        "spacy_dev_scores": best_scores,
    }
    write_record(record)
    print(f"\nTrained in {elapsed / 60:.1f} min; record: {recorded_path(EWT_TRAINING_RECORD)}")

    record["project_dev_scores"] = project_dev_scores(dev_path)
    write_record(record)
    print(json.dumps({k: record[k] for k in ("spacy_dev_scores", "project_dev_scores")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
