"""The measuring stages of version 1.0: per-sentence counts and what produced them.

`run_stage` measures one task of plan/v1.yaml (pos, dep or con) on one dataset
the plan lists for that task, with the members of the task the plan scores on
that dataset (Plan.members_on), and writes two files into <run>/csv/:

  counts_<task>_<dataset>.csv        one row per (configuration, sentence)
  counts_<task>_<dataset>.prov.json  what produced the rows

scripts/pos_tagging.py, scripts/dependency_parsing.py and
scripts/constituency_parsing.py are command lines over it (`cli`), and
scripts/analyze.py reads nothing else. The two files are the contract between
the measuring stages and the analysis; it is stated here once and the analysis
follows it.

Order of work
-------------
  1. The plan, loaded strictly. The dataset must be one the plan lists for the
     task, and every configuration asked for a member of the task that the
     plan scores on the dataset. A member the plan restricts to other
     datasets (configurations.<key>.datasets: nltk_pre and spacy_sm are
     scored on EWT test only, the owner's decision of 24 Oct 2026) is neither
     built nor run on this one, `keys` naming it is refused, and the
     provenance names it under `not_run_by_plan`.
  2. With `confirmatory`, seal.verify_lock() before anything else, whatever
     the dataset: a confirmatory run scores the exploratory sets again beside
     the confirmatory ones, and both have to come from the locked code. A
     confirmatory stage scores the whole set with every configuration the
     plan scores on it: with `max_sentences` or `keys` it is refused before
     the lock is read (since 24 Oct 2026).
  3. The dataset is opened through seal.open_dataset BEFORE any model is
     built, so a refused confirmatory dataset stops the stage before a model
     is loaded, let alone run on it.
  4. Each configuration the plan scores on the dataset, in the order the
     plan lists the task's members, is built by build.build (constructed,
     loaded and checked against the plan; a model this project trains that
     is missing is refused, naming its training script, and never trained:
     since 24 Oct 2026 no build or run trains a model). In a confirmatory
     stage build.build is given seal.verify_model with the lock: right before
     each configuration is loaded, inputs included, the files of a model this
     project trains and the downloaded files its load reads (since the same
     day, before the lock: the Stanza models, ELECTRA-large, benepar_en3,
     SuPar's model, en_core_web_sm, NLTK's tagger and treebank sample) must
     be the ones plan/LOCK.json records, and a mismatch, a missing or an
     extra file stops the stage before that model is loaded (a SealedError,
     which is not one configuration's failure). Each built configuration is
     refused if its provenance could not name the version of its tool
     and of its model (`_check_recordable`),
     the random generators of Python, NumPy and torch are seeded with the
     plan's statistics.bootstrap.seed, and it is run on the sentences one at a
     time, with the gold tokens as its input. One sentence per call, as every
     stage of this project ran before version 1.0: a sentence's output then
     cannot depend on the sentences batched with it. A constituency
     configuration is given the gold tokens in the PTB convention, the one
     of WSJ 01, converted from the one the plan declares for the dataset
     (datasets.<id>.tokens; ptb_conventions.parser_tokens: on MCTB_en a raw
     " becomes `` or ''), and the gold words are put back into its tree
     before it is scored (ptb_conventions.restore_words). Since 24 Oct 2026.
  5. The per-sentence counts are the functions of step 3 of version 1.0
     (metrics.pos_sentence_counts, metrics.dep_sentence_counts,
     evalb.con_sentence_counts under the parameter set the plan's
     constituency metric names). Nothing here scores a sentence any other way.
  6. Both files are written again after every configuration, so a stage that
     stops half way leaves the configurations it finished on disk; the
     provenance says `completed: false` until the last one is done.

A configuration that cannot be built or run is reported with its error, the
stage goes on with the others, and its exit status is 1: nothing is put in its
place, and the counts file has no row of it.

The counts CSV
--------------
UTF-8, LF line ends, one header row, written by the csv module. One row per
configuration and sentence, rows ordered by configuration (the plan's order of
the task's members) and then by `index`. Columns (COLUMNS):

  config    the configuration's key in the plan
  dataset   the dataset's id in the plan
  doc_id    the sentence's document or file: EWT test and GENTLE -
            Sentence.doc_id (the newdoc id, else the sent_id prefix, as the
            loader sets it); WSJ 01 - the file id (wsj_0101.mrg); for these
            it is the resampling unit. MCTB_en - the domain; MCTB_en is
            resampled by sentence, so there it is not the unit
  sent_id   EWT test and GENTLE - the sent_id; WSJ 01 - "<file id>#<position
            in the file>"; MCTB_en - "<domain>-<line>"
  index     the sentence's 0-based position in the dataset as open_dataset
            returns it, which --max-sentences does not change
  n_tokens  the gold words of the sentence (syntactic words of a CoNLL-U
            file, the leaves of a tree once traces are removed)

and the task's counts:

  pos  correct, total
  dep  las_correct, uas_correct, total, clas_correct, clas_total, clas_pred_total
  con  matched, gold, test, parsed, evalb_error, failure, d2_sample

  parsed       1 if the configuration returned a tree that was scored, else 0
  evalb_error  1 if EVALB would skip the sentence (evalb.ConCounts), else 0
  failure      "" for a parsed sentence; else unknown_word, no_derivation or
               over_length, the reasons the PCFG wrapper declares (oov,
               no_derivation, too_long), or `error` for anything else: an
               empty tree with no declared reason, a tree EVALB cannot read,
               a ValueError or StopIteration of the parser. The message of an
               `error` goes to the stage's output, not to the CSV.
  d2_sample    1 if the sentence is in the D2 sample of the dataset, else 0,
               on every row

A sentence with no tree counts 0 test and 0 matched brackets and its gold
brackets (evalb.con_sentence_counts).

The D2 sample
-------------
plan descriptive.d2.sample: `sentences` positions drawn over ALL sentences of
each dataset in d2.datasets, `sorted(random.Random(seed).sample(range(n),
sentences))` (sample_indices; the function the constituency stage used until
23 Oct 2026, so the sample of WSJ 01 is the one of the earlier runs). The PCFG
configurations (d2.configs) run on the sampled sentences only, and a sentence
over max_tokens is declined by the wrapper's own cap (plan.py checks that the
cap is max_tokens) and recorded as over_length. Every other configuration runs
on every sentence: H1a and H2b use whole sets. With max_sentences, the PCFG
runs on the sampled sentences among the first max_sentences.

Provenance
----------
`schema`; `task`; `metric`; `columns`; `dataset` (id, status, path or
nltk_sections, the SHA-256 of the gold set, sentences, tokens, resampling unit
and units); `plan` (path, sha256); `git` (commit, branch, and `dirty` from
`git status --porcelain -- .` run in the project folder, so a change in the
thesis folder beside it does not count; `dirty_scope` names that folder);
`environment` (Python, platform, CPU, and the version of every library a
wrapper imports, "not installed" where one is absent); `offline` (since
24 Oct 2026: `switches`, the value of HF_HUB_OFFLINE and
SYNTACTIC_ANALYSIS_OFFLINE in the stage's environment, null where unset, and
`downloads_allowed`, false in a run of scripts/run_pipeline.py, which sets
both: offline.py); `configurations`, per
key: wrapper, tool (`_tool`: the versions of the libraries the wrapper runs
its model with, or the project's own commit for a wrapper that runs it on
this project's code), init, load, inputs, level, training (id and name), model
(the plan's text), model_version (the analyzer's model_version_for the task),
training_corpus, method_detail, scheme and domain for the task, and `seed`
(`training_seed`: from a training record of the model, or as the wrapper
states it in training_seed_for; plus the inference seed this stage sets);
`not_run_by_plan` (since 24 Oct 2026: each member of the task the plan does
not score on the dataset, with the plan field that says so and the datasets
it names; empty where the plan scores every member there); `failed`; `d2`
and `input_tokens` (con only; `input_tokens`: the dataset's
token convention
as the plan declares it, the convention every configuration is given, and
how many tokens of the sentences scored the conversion changed); `run`
(command line, started, finished, ISO 8601 with the UTC offset,
max_sentences, confirmatory, sentences_scored, completed). Paths are
relative to the project folder; a path outside it is written as "<outside
prog>/<name>", never as a path of this machine. A configuration whose tool or
model version would be empty is not run: it goes to `failed`
(`_check_recordable`), so every configuration in `configurations` names both.

No model library is imported when this module is imported: build() imports a
wrapper module, and the wrapper its library when it loads a model. Added
23 Oct 2026 (version 1.0, step 5), in place of the measuring code of the
three stage scripts, which is in the history and under the tag
exploracia-2026-10-17.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

from syntactic_analysis.base import TrainingSeed
from syntactic_analysis.build import TASK_TYPES, build
from syntactic_analysis.models import REASON_NO_DERIVATION, REASON_OOV, REASON_TOO_LONG
from syntactic_analysis.offline import record as offline_record
from syntactic_analysis.plan import TASKS, Plan, PlanError, load
from syntactic_analysis.project import CSV_SUBDIR, project_root
from syntactic_analysis.seal import SealedError, open_dataset, verify_lock, verify_model

__all__ = [
    "SCHEMA", "ID_COLUMNS", "COUNT_COLUMNS", "COLUMNS", "CON_GOLD", "FAILURE_CODES",
    "ERROR", "LIBRARIES", "PUBLISHED_MODEL", "NONE_RECORDED", "DETERMINISTIC_INDUCTION",
    "SentenceRef", "StageResult", "counts_path", "provenance_path", "task_datasets",
    "default_dataset", "run_datasets", "check_dataset", "select_keys",
    "not_run_by_plan", "sample_indices", "indices_digest", "d2_sample", "sentence_refs",
    "failure_code", "git_state", "environment", "now", "portable",
    "training_seed", "seed_everything", "default_out_dir", "run_stage",
    "summary_table", "build_parser", "cli",
]

# The version of the two files' layout. A reader refuses one it was not
# written for.
SCHEMA = 1

# The columns every counts file starts with, and the counts of each task.
ID_COLUMNS: tuple[str, ...] = ("config", "dataset", "doc_id", "sent_id", "index", "n_tokens")
COUNT_COLUMNS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "pos": ("correct", "total"),
    "dep": ("las_correct", "uas_correct", "total", "clas_correct", "clas_total",
            "clas_pred_total"),
    "con": ("matched", "gold", "test", "parsed", "evalb_error", "failure", "d2_sample"),
})
COLUMNS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    task: ID_COLUMNS + counts for task, counts in COUNT_COLUMNS.items()
})

# How the constituency stage's --gold names the two treebanks it scores, and
# the dataset of the plan each one is. The stage refuses a value whose dataset
# the plan does not list for the task.
CON_GOLD: Mapping[str, str] = MappingProxyType({"wsj01": "wsj_01", "mctb": "mctb_en"})

# The failure column: the PCFG wrapper's declared reasons, and `error` for
# anything else.
FAILURE_CODES: Mapping[str, str] = MappingProxyType({
    REASON_OOV: "unknown_word",
    REASON_NO_DERIVATION: "no_derivation",
    REASON_TOO_LONG: "over_length",
})
ERROR = "error"

# The libraries a wrapper imports, and the ones they stand on, by their
# distribution name. Their versions go into every provenance record.
LIBRARIES: tuple[str, ...] = (
    "numpy", "scipy", "scikit-learn", "nltk", "stanza", "torch", "transformers",
    "tokenizers", "huggingface-hub", "safetensors", "sentencepiece", "protobuf",
    "spacy", "thinc", "en_core_web_sm", "ufal.udpipe", "benepar", "supar",
    "torch-struct", "pyyaml",
)

# The libraries each wrapper module runs its model with, so that every
# configuration's record names the version of its tool beside its model.
_WRAPPER_LIBRARIES: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "nltk": ("nltk",),
    "nltk_transition": ("nltk", "scikit-learn"),
    "udpipe": ("ufal.udpipe",),
    "spacy": ("spacy", "thinc"),
    "stanza": ("stanza", "torch", "transformers"),
    "benepar": ("benepar", "spacy", "torch", "transformers"),
    "supar": ("supar", "torch"),
    "rdrpos": (),
})

# Wrapper modules whose model runs on this project's own code, not on a
# library: wrappers/rdrpos.py implements the RDRPOSTagger inference itself
# (the authors' program is GPL and is neither imported nor run at inference)
# and reads the model files, which its model_version names by digest. The
# version of that tool is the project's code: the git commit of the record,
# and whether the project folder had uncommitted changes (`_project_code`).
_PROJECT_CODE_TOOLS: Mapping[str, str] = MappingProxyType({
    "rdrpos": "syntactic_analysis/wrappers/rdrpos.py",
})

# What `seed.training` says where neither a training record nor the wrapper
# gives a seed (training_seed). The PCFG of pcfg_words and pcfg_tags is
# induced when the model is loaded by counting the rules of the treebank's
# trees (nltk.induce_pcfg), which draws no random number: since 24 Oct 2026
# DETERMINISTIC_INDUCTION says so; until then it was NONE_RECORDED.
PUBLISHED_MODEL = "published model"
NONE_RECORDED = "none recorded"
DETERMINISTIC_INDUCTION = "deterministic induction, no random draw"

# The metric the summary table prints, per task: the plan's primary metric.
_SUMMARY = {"pos": "UPOS", "dep": "LAS", "con": "F1"}

# How often a long run says how far it is.
_PROGRESS_EVERY = 250


# --- names and places --------------------------------------------------------


def counts_path(out_dir: str | Path, task: str, dataset: str) -> Path:
    """<out_dir>/csv/counts_<task>_<dataset>.csv."""
    return Path(out_dir) / CSV_SUBDIR / f"counts_{task}_{dataset}.csv"


def provenance_path(csv_path: str | Path) -> Path:
    """The provenance beside a counts file: counts_<task>_<dataset>.prov.json."""
    return Path(csv_path).with_suffix(".prov.json")


def default_out_dir(day: date | None = None) -> Path:
    """The run directory of a full run: results/run_<YYYY-MM-DD>."""
    day = day or date.today()
    return project_root() / "results" / f"run_{day.isoformat()}"


def portable(path: str | Path) -> str:
    """A path as a provenance record writes it: never a path of this machine.

    Inside the project folder, relative to it with forward slashes
    (results/run_2026-10-24/csv); in an nltk_data directory, from nltk_data on;
    anywhere else, "<outside prog>/" and the last component.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(project_root().resolve()).as_posix()
    except ValueError:
        pass
    parts = resolved.parts
    if "nltk_data" in parts:
        start = len(parts) - 1 - parts[::-1].index("nltk_data")
        return "/".join(parts[start:])
    return f"<outside prog>/{resolved.name}"


def now() -> str:
    """The time, ISO 8601 to the second with the UTC offset: 2026-10-24T09:00:00+02:00."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


# --- what the plan says ------------------------------------------------------


def task_datasets(plan: Plan, task: str, statuses: Iterable[str] | None = None) -> tuple[str, ...]:
    """The datasets the plan lists for `task`, in plan order; only `statuses` if given."""
    wanted = None if statuses is None else set(statuses)
    return tuple(
        key for key, dataset in plan.datasets.items()
        if task in dataset.tasks and (wanted is None or dataset.status in wanted)
    )


def default_dataset(plan: Plan, task: str) -> str:
    """The one exploratory dataset of `task`.

    Raises:
        ValueError: If the plan lists none or several.
    """
    found = task_datasets(plan, task, ("exploratory",))
    if len(found) != 1:
        raise ValueError(
            f"task {task} has {len(found)} exploratory datasets in the plan ({list(found)}); "
            f"name one with --dataset"
        )
    return found[0]


def run_datasets(plan: Plan, task: str, *, confirmatory: bool) -> tuple[str, ...]:
    """What a pipeline run scores for `task`: its exploratory datasets, and with
    `confirmatory` its confirmatory ones after them."""
    exploratory = task_datasets(plan, task, ("exploratory",))
    if not confirmatory:
        return exploratory
    return exploratory + task_datasets(plan, task, ("confirmatory",))


def check_dataset(plan: Plan, task: str, dataset: str) -> None:
    """Refuse a dataset the plan does not list for `task`.

    Raises:
        ValueError: Naming the datasets the plan lists for the task.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {list(TASKS)}")
    listed = task_datasets(plan, task)
    if dataset not in listed:
        raise ValueError(
            f"dataset {dataset!r} is not one the plan lists for task {task}; "
            f"expected one of {list(listed)}"
        )


def select_keys(plan: Plan, task: str, keys: Sequence[str] | None, *,
                dataset: str | None = None) -> tuple[str, ...]:
    """The configurations to run, in the plan's order of the task's members.

    None means every member the plan scores on `dataset` (Plan.members_on),
    or every member of the task when no dataset is given. A key that is not
    a member is refused, and so is one the plan does not score on `dataset`
    (configurations.<key>.datasets, since 24 Oct 2026).

    Raises:
        ValueError: Naming every key that is not a member of the task, or
            every one the plan does not score on the dataset.
    """
    members = plan.members(task)
    scored = members if dataset is None else plan.members_on(task, dataset)
    if keys is None:
        return scored
    strangers = [key for key in keys if key not in members]
    if strangers:
        raise ValueError(
            f"{strangers} not configurations of task {task}; expected some of {list(members)}"
        )
    excluded = list(dict.fromkeys(key for key in keys if key not in scored))
    if excluded:
        raise ValueError(
            f"{excluded} not scored on {dataset} by the plan ("
            + "; ".join(f"configurations.{key}.datasets: {list(plan.configurations[key].datasets)}"
                        for key in excluded)
            + f"); expected some of {list(scored)}"
        )
    chosen = set(keys)
    return tuple(key for key in members if key in chosen)


def not_run_by_plan(plan: Plan, task: str, dataset: str) -> dict[str, dict[str, Any]]:
    """The members of `task` the plan does not score on `dataset`, as provenance records them.

    Per key: the plan field that restricts it and the datasets that field
    names. Empty where the plan scores every member of the task there.
    """
    scored = plan.members_on(task, dataset)
    return {
        key: {
            "plan_field": f"configurations.{key}.datasets",
            "datasets": list(plan.configurations[key].datasets or ()),
        }
        for key in plan.members(task) if key not in scored
    }


# --- the D2 sample -----------------------------------------------------------


def sample_indices(n_sentences: int, sample: int | None, seed: int) -> tuple[int, ...]:
    """The sentence positions a sample of `sample` out of `n_sentences` holds, sorted.

    A seeded simple random sample, without stratification by length:
    `sorted(random.Random(seed).sample(range(n_sentences), sample))`. Every
    position when `sample` is None or at least `n_sentences`. Moved here on
    23 Oct 2026 from scripts/constituency_parsing.py, unchanged, so the PCFG
    sample of WSJ 01 (250 sentences, seed 20260101) is the one of every earlier
    run.
    """
    if sample is None or sample >= n_sentences:
        return tuple(range(n_sentences))
    return tuple(sorted(random.Random(seed).sample(range(n_sentences), sample)))


def indices_digest(indices: Iterable[int]) -> str:
    """SHA-256 of sorted positions written as ASCII decimals joined by commas: "3,17,40"."""
    text = ",".join(str(index) for index in sorted(indices))
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def d2_sample(plan: Plan, dataset: str, n_sentences: int) -> tuple[int, ...] | None:
    """The D2 sample of `dataset`, drawn over all its sentences; None if D2 does not use it.

    Raises:
        ValueError: If the plan draws the sample in a way not implemented here.
    """
    if dataset not in plan.d2.datasets:
        return None
    sample = plan.d2.sample
    if sample.drawn_from != "all_sentences":
        raise ValueError(
            f"descriptive.d2.sample.drawn_from {sample.drawn_from!r}: only all_sentences "
            f"is implemented"
        )
    return sample_indices(n_sentences, sample.sentences, sample.seed)


# --- the sentences ------------------------------------------------------------


@dataclass(frozen=True)
class SentenceRef:
    """What a counts row says about its sentence; see the module docstring."""

    doc_id: str
    sent_id: str
    index: int
    n_tokens: int


def sentence_refs(plan: Plan, dataset: str, gold) -> list[SentenceRef]:
    """One SentenceRef per sentence of `gold`, in its order.

    A CoNLL-U gold set (GoldSet) gives the loader's doc_id and sent_id; a
    treebank (ConstituencyGoldSet) the file and the position in it for a
    dataset read from NLTK's WSJ sample (`nltk_sections`), the domain and
    "<domain>-<line>" for one read from files (MCTB_en).
    """
    if hasattr(gold, "parse"):
        return [
            SentenceRef(sentence.doc_id, sentence.sent_id, index, len(sentence.tokens))
            for index, sentence in enumerate(gold.parse.sentences)
        ]
    from_nltk = plan.datasets[dataset].nltk_sections is not None
    refs = []
    for index, (sentence, words) in enumerate(zip(gold.sentences, gold.pretok)):
        if from_nltk:
            refs.append(SentenceRef(sentence.fileid, f"{sentence.fileid}#{sentence.index}",
                                    index, len(words)))
        else:
            refs.append(SentenceRef(sentence.domain, sentence.sent_id, index, len(words)))
    return refs


def _units(plan: Plan, dataset: str, refs: Sequence[SentenceRef]) -> int:
    """How many resampling units the sentences make, by the plan's unit."""
    if plan.datasets[dataset].resampling.unit == "sentence":
        return len(refs)
    return len({ref.doc_id for ref in refs})


def failure_code(reason: str) -> str:
    """The failure column for a sentence without a tree, from the wrapper's reason."""
    return FAILURE_CODES.get(reason, ERROR)


# --- provenance ---------------------------------------------------------------


def git_state(root: str | Path | None = None) -> dict[str, Any]:
    """Commit, branch and whether the project folder has uncommitted changes.

    `dirty` comes from `git status --porcelain -- .` run in the project folder,
    so the thesis folder beside it in the same repository does not count;
    `dirty_scope` is that folder as git names it (`git rev-parse
    --show-prefix`). None for anything git cannot say.
    """
    root = project_root() if root is None else Path(root)

    def git(*args: str) -> subprocess.CompletedProcess | None:
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True,
                                  text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None

    def out(result: subprocess.CompletedProcess | None) -> str | None:
        if result is None or result.returncode != 0:
            return None
        return result.stdout.strip()

    status = git("status", "--porcelain", "--", ".")
    dirty = None if out(status) is None else bool(status.stdout.strip())
    return {
        "commit": out(git("rev-parse", "HEAD")) or None,
        "branch": out(git("rev-parse", "--abbrev-ref", "HEAD")) or None,
        "dirty": dirty,
        "dirty_scope": out(git("rev-parse", "--show-prefix")),
    }


def environment(libraries: Iterable[str] = LIBRARIES) -> dict[str, Any]:
    """Python, platform, CPU and library versions (importlib.metadata)."""
    import importlib.metadata as metadata

    versions = {}
    for name in libraries:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "cpu": f"{platform.processor() or 'unknown processor'} ({platform.machine() or 'unknown machine'})",
        "libraries": versions,
    }


def _training_records() -> tuple[tuple[Path, str, str, Callable[[Mapping, str], bool]], ...]:
    """The training records scripts of this project write, and how to tell their model.

    Each entry: the record, the field that holds the seed, what that seed
    seeded, and whether a loaded analyzer's model_version names the model the
    record describes (its digest, as the wrapper writes it). The places come
    from the wrappers that load those models.
    """
    from syntactic_analysis.wrappers import rdrpos, spacy, udpipe

    def spacy_model(record: Mapping, version: str) -> bool:
        digest = str(record.get("pipeline_digest") or "")[:12]
        prefix = f"{spacy.EWT_WORK_DIR.name}/{spacy.EWT_PIPELINE_DIR.name}@"
        return bool(digest) and version.startswith(prefix + digest)

    def rdr_model(record: Mapping, version: str) -> bool:
        files = record.get("files") or {}
        digest = str((files.get("rules") or {}).get("sha256") or "")[:12]
        prefix = f"{rdrpos.EWT_WORK_DIR.name}/{rdrpos.EWT_RULES.name}@"
        return bool(digest) and version.startswith(prefix + digest)

    def udpipe_model(record: Mapping, version: str) -> bool:
        return version.startswith(f"{udpipe.RETRAINED_MODEL_PATH.name}@")

    udpipe_record = udpipe.RETRAINED_MODEL_PATH.with_name(udpipe.RETRAINED_MODEL_PATH.name + ".json")
    return (
        (spacy.EWT_TRAINING_RECORD, "seed", "spaCy --system.seed of the training run", spacy_model),
        (rdrpos.EWT_TRAINING_RECORD, "hash_seed", "PYTHONHASHSEED of the learner run", rdr_model),
        (udpipe_record, "seed", "seed of the UDPipe training", udpipe_model),
    )


def training_seed(plan: Plan, key: str, task: str, model_version: str,
                  stated: TrainingSeed | None = None) -> dict[str, Any]:
    """The seed the model of configuration `key` was trained with, as a record or its code states it.

    Read, never inferred:
      * a training record whose model is the one loaded (`_training_records`)
        gives its seed;
      * otherwise the seed the wrapper states for the loaded model
        (`stated`, BaseAnalyzer.training_seed_for: a seed fixed in the
        wrapper's code or its library's, or read from the loaded model), with
        the wrapper's note; `record` still names a training record of the
        model that holds no seed (udpipe_ewt);
      * a training record that holds no seed, with no seed stated: "none
        recorded";
      * a model trained by this project from a corpus file of the plan
        (training_data.<id>.path) with neither: "none recorded";
      * the PCFG of a constituency configuration of the NLTK wrapper, which
        the wrapper induces from its treebank when it is loaded by counting
        rules, with no random draw: "deterministic induction, no random
        draw" (DETERMINISTIC_INDUCTION), with the treebank named in `note`;
      * anything else is a downloaded model: "published model".
    Until 24 Oct 2026 no wrapper stated a seed, and nltk_ewt, nltk_arc and
    udpipe_ewt were "none recorded"; so was the PCFG.
    """
    for path, field_name, what, describes in _training_records():
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict) or not describes(record, model_version):
            continue
        seed = record.get(field_name)
        if seed is not None:
            return {"training": seed, "record": portable(path), "note": what}
        if stated is not None:
            return {"training": stated.seed, "record": portable(path), "note": stated.note}
        return {"training": NONE_RECORDED, "record": portable(path),
                "note": f"the training record holds no {field_name}"}
    if stated is not None:
        return {"training": stated.seed, "record": None, "note": stated.note}
    config = plan.configurations[key]
    training = plan.training_data[config.training]
    if training.path is not None:
        return {"training": NONE_RECORDED, "record": None,
                "note": f"trained by this project on {training.path}; no training record "
                        f"with a seed"}
    if task == "con" and config.wrapper == "nltk.NltkAnalyzer":
        return {"training": DETERMINISTIC_INDUCTION, "record": None,
                "note": f"grammar induced when the model is loaded, from {training.name}"}
    return {"training": PUBLISHED_MODEL, "record": None, "note": None}


def seed_everything(seed: int) -> list[str]:
    """Seed Python's random, NumPy's global generator and torch if it is loaded.

    Returns the generators seeded. torch is seeded only once a wrapper has
    imported it: importing it here would cost seconds for configurations that
    do not use it.
    """
    random.seed(seed)
    seeded = ["random"]
    import numpy

    numpy.random.seed(seed)
    seeded.append("numpy")
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
        seeded.append("torch")
    return seeded


def _plain(value: Any) -> Any:
    """The plan's frozen value as JSON holds it: mappings as dicts, tuples as lists."""
    if isinstance(value, Mapping):
        return {str(name): _plain(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _project_code(git: Mapping[str, Any]) -> str:
    """This project's code as a tool version: the commit, and uncommitted changes if any."""
    text = f"this project's code at git commit {git.get('commit') or '(unknown)'}"
    if git.get("dirty") is None:
        return text + ", uncommitted changes not known"
    if git.get("dirty"):
        return text + f", with uncommitted changes in {git.get('dirty_scope') or 'the project folder'}"
    return text


def _tool(plan: Plan, key: str, libraries: Mapping[str, str],
          git: Mapping[str, Any]) -> dict[str, str]:
    """What `configurations.<key>.tool` holds: the version of what runs the model.

    The versions `environment` read of the libraries the configuration's
    wrapper runs its model with, and for a wrapper that runs it on this
    project's own code (_PROJECT_CODE_TOOLS) that code's version.
    """
    module = plan.configurations[key].module
    tool = {name: libraries.get(name, "not recorded") for name in _WRAPPER_LIBRARIES.get(module, ())}
    if module in _PROJECT_CODE_TOOLS:
        tool[_PROJECT_CODE_TOOLS[module]] = _project_code(git)
    return tool


def _check_recordable(plan: Plan, task: str, key: str, analyzer,
                      tool: Mapping[str, str]) -> None:
    """Refuse a built configuration whose provenance could not say what ran.

    Check G of the version 1.0 brief: the provenance of every configuration
    gives the version of its tool, its model, its data and its seed. The data
    and the seed are the stage's own; the tool and the model come from the
    wrapper, and a configuration whose wrapper names neither a library nor
    this project's code (`_tool`), or no model version for the task
    (`model_version_for`), is not run: it goes to `failed`, like a model
    that does not load.

    Raises:
        RuntimeError: Naming what is missing.
    """
    config = plan.configurations[key]
    if not analyzer.model_version_for(TASK_TYPES[task]):
        raise RuntimeError(
            f"{config.wrapper} names no model version for {task} (model_version is empty), "
            f"so the provenance could not say which model ran"
        )
    if not tool:
        raise RuntimeError(
            f"no tool version is known for the wrapper module {config.module!r} "
            f"(stage._WRAPPER_LIBRARIES), so the provenance could not say what ran the model"
        )


def _configuration_record(plan: Plan, task: str, key: str, analyzer,
                          tool: Mapping[str, str]) -> dict[str, Any]:
    """What `configurations.<key>` of the provenance holds for a configuration that ran.

    `tool` is `_tool`'s; `model_version` is the analyzer's model_version_for
    the task.
    """
    config = plan.configurations[key]
    member = plan.tasks[task].members[key]
    analysis_type = TASK_TYPES[task]
    version = analyzer.model_version_for(analysis_type) or ""
    return {
        "wrapper": config.wrapper,
        "tool": dict(tool),
        "init": _plain(config.init),
        "load": _plain(config.load),
        "inputs": _plain(config.inputs),
        "level": member.level.code,
        "training": {"id": config.training, "name": plan.training_data[config.training].name},
        "model": config.model,
        "model_version": version,
        "training_corpus": analyzer.training_corpus_for(analysis_type),
        "method_detail": analyzer.method_detail_for(analysis_type),
        "scheme": analyzer.scheme_for(analysis_type).name,
        "domain": analyzer.training_domain_for(analysis_type).value,
        "seed": {
            **training_seed(plan, key, task, version,
                            analyzer.training_seed_for(analysis_type)),
            "inference": plan.statistics.bootstrap.seed,
        },
    }


# --- the stage -----------------------------------------------------------------


@dataclass
class StageResult:
    """What a stage wrote and how each configuration came out."""

    task: str
    dataset: str
    csv_path: Path
    provenance_path: Path
    summary: list[dict[str, Any]] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed


def _failure_text(exc: BaseException) -> str:
    """One line: the exception's type and the first line of its message."""
    lines = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {lines[0] if lines else ''}"


def _metric(counts: list, task: str) -> float:
    """The plan's primary metric of the counts, NaN when there are none."""
    from syntactic_analysis.evaluation.metrics import corpus_metric

    if not counts:
        return math.nan
    return corpus_metric(counts)


def _run_pos_dep(analyzer, task: str, gold, indices: Sequence[int], emit) -> list:
    """One counts object per sentence of `indices`: UPOS or attachment counts."""
    from syntactic_analysis.evaluation.metrics import dep_sentence_counts, pos_sentence_counts

    count = pos_sentence_counts if task == "pos" else dep_sentence_counts
    analyze = analyzer.analyze_pos if task == "pos" else analyzer.analyze_dependency
    out = []
    for done, index in enumerate(indices):
        _progress(done, len(indices), emit)
        predicted = analyze([gold.pretok[index]])
        out.append(count(predicted.sentences[0], gold.parse.sentences[index]))
    return out


def _progress(done: int, total: int, emit) -> None:
    """A line every _PROGRESS_EVERY sentences, so a long run shows it is running."""
    if done and done % _PROGRESS_EVERY == 0:
        emit(f"    {done}/{total} sentences")


def _given_tokens(tokens, words: Sequence[str]) -> list[str]:
    """One sentence as a constituency configuration is given it (ptb_conventions.parser_tokens)."""
    from syntactic_analysis.ptb_conventions import parser_tokens

    return parser_tokens(words, brackets=tokens.brackets, quotes=tokens.quotes)


def _run_con(analyzer, gold, indices: Sequence[int], refs: Sequence[SentenceRef], params,
             emit, tokens) -> list[tuple[Any, str]]:
    """(ConCounts, failure) per sentence of `indices`.

    `tokens` is the dataset's token convention (plan datasets.<id>.tokens).
    The parser is given each sentence in the PTB convention, converted from
    it (`_given_tokens`), and the gold words are put back into its tree where
    the conversion changed them before the tree is scored
    (ptb_conventions.restore_words); on WSJ 01 nothing is converted and the
    tree is scored as the parser wrote it. A tree the gold words cannot be
    put back into position by position is not scored: like a tree EVALB
    cannot read, it is the sentence's `error`.

    The parser's ValueError or StopIteration, an empty tree and a tree EVALB
    cannot read are the sentence's `error` (with the message emitted); an
    empty tree with a reason the wrapper declares is that reason's code. A
    ContractViolation, or any other exception, stops the configuration: it is
    a defect of the wrapper or of this code, and scoring it as the parser's
    failure would hide it.
    """
    from syntactic_analysis.evaluation.evalb import con_sentence_counts
    from syntactic_analysis.ptb_conventions import restore_words

    out = []
    for done, index in enumerate(indices):
        _progress(done, len(indices), emit)
        words, gold_tree = gold.pretok[index], gold.trees[index]
        given = _given_tokens(tokens, words)
        expected = len(words)
        # The gold tree on its own first: a gold tree that cannot be scored is
        # a defect of the data, and stops the stage rather than the sentence.
        no_tree = con_sentence_counts(None, gold_tree, params=params, expected_leaves=expected)
        where = f"{analyzer.name}: sentence {index} ({refs[index].sent_id})"
        try:
            result = analyzer.analyze_constituency([given])
        except (ValueError, StopIteration) as exc:
            emit(f"    {where}: error: {_failure_text(exc)}")
            out.append((no_tree, ERROR))
            continue
        tree = result.trees[0] if result.trees else ""
        reason = result.failures[0] if result.failures else ""
        if not tree:
            code = failure_code(reason)
            if code == ERROR:
                emit(f"    {where}: error: no tree, reason {reason or '(none given)'!r}")
            out.append((no_tree, code))
            continue
        try:
            counts = con_sentence_counts(restore_words(tree, words, given), gold_tree,
                                         params=params, expected_leaves=expected)
        except ValueError as exc:
            emit(f"    {where}: error: the tree cannot be scored: {_failure_text(exc)}")
            out.append((no_tree, ERROR))
            continue
        out.append((counts, ""))
    return out


def _write_counts(path: Path, task: str, rows: Iterable[Sequence[Any]]) -> None:
    """The counts file: header and rows, UTF-8 with LF line ends."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS[task])
        writer.writerows(rows)


def _write_json(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")


def run_stage(
    task: str,
    dataset: str | None = None,
    *,
    out_dir: str | Path,
    confirmatory: bool = False,
    max_sentences: int | None = None,
    keys: Sequence[str] | None = None,
    plan: Plan | None = None,
    command: str = "",
    build_analyzer: Callable[..., Any] = build,
    open_gold: Callable[..., Any] = open_dataset,
    emit: Callable[[str], None] = print,
) -> StageResult:
    """Measure `task` on `dataset` and write its counts file and provenance.

    Args:
        task:          pos, dep or con.
        dataset:       A dataset the plan lists for the task; None for the
                       task's one exploratory dataset.
        out_dir:       The run directory; the files go into its csv/.
        confirmatory:  The run was started with --confirmatory: the lock is
                       verified first, and a confirmatory dataset may open.
                       Refused with max_sentences or keys.
        max_sentences: Score the first N sentences only (a smoke run); the
                       D2 sample is still drawn over all of them.
        keys:          Some of the task's configurations the plan scores on
                       the dataset; None for all of them (select_keys).
        plan:          The plan; load(strict=True) when None.
        command:       The command line, as provenance records it.
        build_analyzer, open_gold: build.build and seal.open_dataset; a test
                       passes others. With `confirmatory`, build_analyzer is
                       also given before_load: seal.verify_model with the lock.
        emit:          Where progress and the summary go.

    Raises:
        SealedError: The lock does not hold, the dataset is sealed, or in a
            confirmatory stage the files of a model this project trains, or
            the downloaded files a configuration's load reads, are not the
            ones the lock records when it is about to be loaded (the stage
            stops there).
        PlanError, ValueError: A task, dataset or key the plan does not have,
            a confirmatory stage with max_sentences or keys, or a gold set
            that is not what the plan expects.
    """
    from syntactic_analysis.evaluation.report import quiet_third_party

    started = now()
    plan = load(strict=True) if plan is None else plan
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {list(TASKS)}")
    dataset = default_dataset(plan, task) if dataset is None else dataset
    check_dataset(plan, task, dataset)
    # The members the plan scores on this dataset (24 Oct 2026: nltk_pre and
    # spacy_sm on EWT test only); the others are not built, run or scored.
    chosen = select_keys(plan, task, keys, dataset=dataset)
    not_run = not_run_by_plan(plan, task, dataset)
    if max_sentences is not None and max_sentences < 1:
        raise ValueError(f"max_sentences must be at least 1, not {max_sentences}")
    if confirmatory and (max_sentences is not None or keys is not None):
        # Brief step 9: the confirmatory run scores whole sets with every
        # configuration, once. A stage may not score part of one beside it.
        raise ValueError(
            "a confirmatory stage scores the whole set with every configuration of the "
            "task the plan scores on it, once: it takes no max_sentences (--max-sentences) "
            "and no keys (--keys)"
        )
    # The lock, in a confirmatory stage: verified here (the plan, the code,
    # the data of every dataset - the exploratory ones since 25 Oct 2026 -,
    # the trained models and the downloaded ones), and the files of each
    # configuration checked against it again right before it is loaded
    # (build's before_load). Since 24 Oct 2026 for the models.
    lock = verify_lock(plan) if confirmatory else None
    checks = {} if lock is None else {"before_load": partial(verify_model, lock)}

    # Opened before any model is built: a sealed dataset stops the stage here.
    gold = open_gold(plan, dataset, confirmatory=confirmatory)

    spec = plan.datasets[dataset]
    refs = sentence_refs(plan, dataset, gold)
    n = len(refs) if max_sentences is None else min(max_sentences, len(refs))
    scored = range(n)
    seed = plan.statistics.bootstrap.seed
    analysis_type = TASK_TYPES[task]

    params = sample = None
    pcfg: frozenset[str] = frozenset()
    converted = 0
    if task == "con":
        from syntactic_analysis.evaluation.evalb import plan_parameter_set

        if spec.tokens is None:
            raise ValueError(
                f"datasets.{dataset} declares no tokens, so its sentences cannot be given to "
                f"the parsers in the PTB convention"
            )
        params = plan_parameter_set(plan.tasks["con"].metric.options)
        sample = d2_sample(plan, dataset, len(refs))
        pcfg = frozenset(plan.d2.configs)
        converted = sum(
            word != sent
            for index in scored
            for word, sent in zip(gold.pretok[index],
                                  _given_tokens(spec.tokens, gold.pretok[index]))
        )
    in_sample = frozenset(sample or ())

    csv_path = counts_path(out_dir, task, dataset)
    prov_path = provenance_path(csv_path)
    result = StageResult(task, dataset, csv_path, prov_path)

    record: dict[str, Any] = {
        "schema": SCHEMA,
        "task": task,
        "metric": {"name": plan.tasks[task].metric.name,
                   "options": _plain(plan.tasks[task].metric.options)},
        "columns": list(COLUMNS[task]),
        "dataset": {
            "id": dataset,
            "status": spec.status,
            **({"path": spec.path} if spec.path is not None
               else {"nltk_sections": list(spec.nltk_sections or ())}),
            "sha256": gold.sha256,
            "sentences": len(refs),
            "tokens": gold.n_tokens,
            "unit": spec.resampling.unit,
            "units": _units(plan, dataset, refs),
        },
        "plan": {"path": portable(plan.path), "sha256": plan.sha256},
        "git": git_state(),
        "environment": environment(),
        "offline": offline_record(),
        "configurations": {},
        "not_run_by_plan": not_run,
        "failed": {},
        "run": {
            "command": command,
            "started": started,
            "finished": None,
            "max_sentences": max_sentences,
            "confirmatory": confirmatory,
            "sentences_scored": n,
            "configurations": list(chosen),
            "completed": False,
        },
    }
    if task == "con":
        record["evalb_parameter_set"] = params.name
        record["d2"] = None if sample is None else {
            "sentences": plan.d2.sample.sentences,
            "seed": plan.d2.sample.seed,
            "max_tokens": plan.d2.sample.max_tokens,
            "drawn_from": plan.d2.sample.drawn_from,
            "configs": list(plan.d2.configs),
            "sampled": len(sample),
            "sampled_sha256": indices_digest(sample),
            "sampled_scored": sum(1 for index in sample if index < n),
        }
        record["input_tokens"] = {
            "dataset": {"brackets": spec.tokens.brackets, "quotes": spec.tokens.quotes},
            "given": "ptb",
            "rule": "syntactic_analysis.ptb_conventions.parser_tokens",
            "converted": converted,
        }

    emit(f"STAGE {task} on {dataset} ({spec.status}): {len(chosen)} configuration(s), "
         f"{n} of {len(refs)} sentences")
    emit(f"  plan {portable(plan.path)} sha256 {plan.sha256}")
    emit(f"  gold sha256 {gold.sha256}")
    offline = record["offline"]
    emit(f"  downloads: {'allowed' if offline['downloads_allowed'] else 'none'} ("
         + ", ".join(f"{name}={value if value is not None else 'unset'}"
                     for name, value in offline["switches"].items()) + ")")
    if not_run:
        emit("  not run by the plan: " + "; ".join(
            f"{key} ({entry['plan_field']}: {', '.join(entry['datasets'])})"
            for key, entry in not_run.items()))
    if task == "con":
        emit(f"  tokens: brackets {spec.tokens.brackets}, quotes {spec.tokens.quotes}; given to "
             f"every configuration in the ptb convention, {converted} converted among those scored")
    if task == "con" and sample is not None:
        emit(f"  D2 sample: {len(sample)} of {len(refs)} sentences (seed {plan.d2.sample.seed}), "
             f"{record['d2']['sampled_scored']} among those scored")

    if task == "con" and sample is None and any(key in pcfg for key in chosen):
        raise ValueError(
            f"descriptive.d2.datasets does not list {dataset!r}, and the PCFG "
            f"configurations {sorted(pcfg & set(chosen))} run on a D2 sample only"
        )

    # A configuration another one takes as an input (udpipe_ewt, nltk_arc's
    # tagger) is built once and kept for the run; every other one is dropped
    # once it has run.
    inputs = {target for key in chosen for target in plan.configurations[key].inputs.values()}
    built: dict[str, Any] = {}
    rows: dict[str, list[tuple]] = {}
    for key in chosen:
        emit(f"{key}: building ({plan.configurations[key].wrapper})")
        analyzer = results = entry = None
        indices = [i for i in scored if i in in_sample] if key in pcfg else list(scored)
        try:
            analyzer = built.get(key) or build_analyzer(plan, key, built=built, **checks)
            tool = _tool(plan, key, record["environment"]["libraries"], record["git"])
            _check_recordable(plan, task, key, analyzer, tool)
            if task == "con":
                # Every configuration is given its sentences in the PTB
                # convention, converted from the dataset's (_run_con); a
                # wrapper with a separate entry point for such input (Benepar)
                # is told so. Gold tags are given to none: no configuration of
                # the plan parses over them.
                escaped = getattr(analyzer, "set_ptb_escaped_input", None)
                if escaped is not None:
                    escaped(True)
            seed_everything(seed)
            with quiet_third_party():
                if task == "con":
                    results = _run_con(analyzer, gold, indices, refs, params, emit, spec.tokens)
                else:
                    results = [(counts, None) for counts in
                               _run_pos_dep(analyzer, task, gold, indices, emit)]
            entry = _configuration_record(plan, task, key, analyzer, tool)
        except SealedError as exc:
            # A trained model that is not the locked one: not a failure of
            # this configuration but of the lock, so the stage stops here,
            # before the model is loaded. The provenance says so.
            record["failed"][key] = _failure_text(exc)
            _write_json(prov_path, record)
            emit(f"  STOPPED: {exc}")
            raise
        except Exception as exc:  # noqa: BLE001 - reported, recorded, and the exit status says so
            result.failed[key] = _failure_text(exc)
            record["failed"][key] = result.failed[key]
            emit(f"  FAILED: {result.failed[key]}")
            results = None
        finally:
            if analyzer is not None and key in inputs:
                built[key] = analyzer
            analyzer = None
            gc.collect()
        if results is None:
            _write_json(prov_path, record)
            continue
        rows[key] = []
        for index, (counts, failure) in zip(indices, results):
            ref = refs[index]
            row = [key, dataset, ref.doc_id, ref.sent_id, ref.index, ref.n_tokens]
            if task == "pos":
                row += [counts.correct, counts.total]
            elif task == "dep":
                row += [counts.las_correct, counts.uas_correct, counts.total,
                        counts.clas_correct, counts.clas_total, counts.clas_pred_total]
            else:
                row += [counts.matched, counts.gold, counts.test, counts.parsed,
                        counts.evalb_error, failure, int(index in in_sample)]
            rows[key].append(tuple(row))
        record["configurations"][key] = entry
        metric = _metric([counts for counts, _ in results], task)
        summary = {"config": key, "sentences": len(results), "metric": metric}
        if task == "con":
            summary["parsed"] = sum(counts.parsed for counts, _ in results)
            failures: dict[str, int] = {}
            for _, failure in results:
                if failure:
                    failures[failure] = failures.get(failure, 0) + 1
            summary["failures"] = failures
        result.summary.append(summary)
        emit(f"  {_summary_line(task, summary)}")
        _write_counts(csv_path, task, (row for key in chosen for row in rows.get(key, ())))
        _write_json(prov_path, record)

    record["run"]["finished"] = now()
    record["run"]["completed"] = True
    _write_counts(csv_path, task, (row for key in chosen for row in rows.get(key, ())))
    _write_json(prov_path, record)
    for line in summary_table(result):
        emit(line)
    return result


def _summary_line(task: str, summary: Mapping[str, Any]) -> str:
    metric = summary["metric"]
    value = "-" if math.isnan(metric) else f"{metric:.2f}"
    text = f"{summary['config']}: {summary['sentences']} sentences, {_SUMMARY[task]} {value}"
    if task == "con":
        text += f", {summary['parsed']} parsed"
        if summary.get("failures"):
            text += " (" + ", ".join(f"{code} {count}" for code, count in
                                     sorted(summary["failures"].items())) + ")"
    return text


def summary_table(result: StageResult) -> list[str]:
    """The table a stage prints at the end: configuration, sentences, metric."""
    metric = _SUMMARY[result.task]
    header = ["config", "sentences", metric] + (["parsed"] if result.task == "con" else [])
    lines = ["", f"SUMMARY {result.task} on {result.dataset}", " | ".join(header),
             " | ".join("---" for _ in header)]
    for summary in result.summary:
        value = summary["metric"]
        cells = [summary["config"], str(summary["sentences"]),
                 "-" if math.isnan(value) else f"{value:.2f}"]
        if result.task == "con":
            cells.append(str(summary["parsed"]))
        lines.append(" | ".join(cells))
    for key, message in result.failed.items():
        lines.append(f"{key} | FAILED: {message}")
    lines.append(f"written: {portable(result.csv_path)}")
    lines.append(f"         {portable(result.provenance_path)}")
    return lines


# --- the command line ---------------------------------------------------------


def build_parser(task: str, description: str | None = None) -> argparse.ArgumentParser:
    """The command line of the stage of `task`.

    pos and dep take --dataset (default: the task's exploratory dataset); con
    takes --gold wsj01|mctb instead (CON_GOLD). All three take --confirmatory,
    --out-dir, --max-sentences and --keys.
    """
    parser = argparse.ArgumentParser(description=description)
    if task == "con":
        parser.add_argument(
            "--gold", choices=tuple(CON_GOLD), default="wsj01",
            help="the treebank to score against: wsj01 (plan dataset wsj_01, the "
                 "default) or mctb (mctb_en, confirmatory)",
        )
    else:
        parser.add_argument(
            "--dataset", default=None, metavar="ID",
            help="a dataset the plan lists for the task (default: its exploratory one)",
        )
    parser.add_argument(
        "--confirmatory", action="store_true",
        help="a confirmatory run: verify plan/LOCK.json first; only then does a "
             "confirmatory dataset open. Scores the whole set with every "
             "configuration the plan scores on it: refused with --max-sentences "
             "or --keys",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None, metavar="DIR",
        help="the run directory; the files go into DIR/csv/ (default: "
             "results/run_<YYYY-MM-DD>)",
    )
    parser.add_argument(
        "--max-sentences", type=int, default=None, metavar="N",
        help="score the first N sentences only (a smoke run; recorded in provenance)",
    )
    parser.add_argument(
        "--keys", default=None, metavar="K1,K2",
        help="some configurations of the task the plan scores on the dataset, "
             "comma-separated (default: all of them)",
    )
    return parser


def _command(script: str, args: argparse.Namespace, task: str) -> str:
    """The command line as provenance records it: no path of this machine."""
    parts = ["python", script]
    if task == "con":
        parts += ["--gold", args.gold]
    elif args.dataset is not None:
        parts += ["--dataset", args.dataset]
    if args.confirmatory:
        parts.append("--confirmatory")
    if args.out_dir is not None:
        parts += ["--out-dir", portable(args.out_dir)]
    if args.max_sentences is not None:
        parts += ["--max-sentences", str(args.max_sentences)]
    if args.keys is not None:
        parts += ["--keys", args.keys]
    return " ".join(parts)


def cli(task: str, argv: Sequence[str] | None = None, *, script: str,
        description: str | None = None) -> int:
    """Run the stage of `task` from its command line; the exit status.

    0 when every configuration ran; 1 when one failed or the stage was
    refused (the reason goes to stderr); 2 for a command line argparse
    refuses.
    """
    from syntactic_analysis.evaluation.report import utf8_console

    utf8_console(stderr=True)
    parser = build_parser(task, description)
    args = parser.parse_args(argv)
    if args.max_sentences is not None and args.max_sentences < 1:
        parser.error("--max-sentences must be at least 1")
    keys = None
    if args.keys is not None:
        keys = [key.strip() for key in args.keys.split(",") if key.strip()]
        if not keys:
            parser.error("--keys names no configuration")
    dataset = CON_GOLD[args.gold] if task == "con" else args.dataset
    out_dir = args.out_dir if args.out_dir is not None else default_out_dir()
    try:
        result = run_stage(
            task, dataset, out_dir=out_dir, confirmatory=args.confirmatory,
            max_sentences=args.max_sentences, keys=keys,
            command=_command(script, args, task),
        )
    except (PlanError, ValueError, RuntimeError) as exc:
        print(f"{script}: refused: {exc}", file=sys.stderr)
        return 1
    return 0 if result.ok else 1
