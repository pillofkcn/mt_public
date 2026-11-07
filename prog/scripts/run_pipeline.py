"""Run the experiment of plan/v1.yaml stage by stage, into one run directory.

    pos -> dep -> con -> analyze -> figures

  pos, dep      scripts/pos_tagging.py and scripts/dependency_parsing.py, once
                per dataset the run scores (--dataset).
  con           scripts/constituency_parsing.py, once per treebank (--gold).
  analyze       scripts/analyze.py --run-dir <run>: the estimates and verdicts,
                from the counts files alone.
  figures       scripts/make_figures.py --run-dir <run>: the three figures of
                the thesis (H1, H2, H2c), from analyze's thesis tables alone.

Which datasets a run scores comes from the plan. An exploratory run (the
default) scores each task's exploratory dataset: the EWT test split for pos
and dep, WSJ section 01 for con. A run started with --confirmatory scores the
exploratory AND the confirmatory datasets (pos and dep: EWT test and GENTLE;
con: WSJ 01 and MCTB_en) in one run at one commit, so that H2b's in-domain and
out-of-domain counts come from the same locked code, and then analyzes them
and draws the figures.
It verifies plan/LOCK.json before the first stage (seal.verify_lock), and
every stage verifies it again; before step 8 of version 1.0 there is no lock
and the run stops there. A confirmatory run scores whole sets: --confirmatory
with --max-sentences is refused.

Every measuring stage writes into <run>/csv/ (counts_<task>_<dataset>.csv and
its provenance, syntactic_analysis.stage); each command's console output goes
to <run>/log/run_<stage>[_<dataset>].txt as well as to this console. Nothing
is timed but the run as a whole: run.json and <run>/log/pipeline.txt record its
total time, so that it can be seen how long a run took; no stage records
timings, and no figure or test reads them.

    results/run_<YYYY-MM-DD>/     a full run (refused if the directory exists:
                                  pass --out-dir to resume one)
    results/smoke/v1_<stamp>/     a run with --max-sentences (git ignores it)
        csv/        counts files and their provenance; analyze writes thesis_*.csv
        figures/    fig1_h1, fig2_h2 and fig3_h2c as PNG and PDF, figures.prov.json
        log/        one console transcript per command, and pipeline.txt
        run.json    the invocations of this script on the directory, and per
                    stage the record of the invocation that last ran it

A stage that fails stops the run: every later stage reads what the earlier
ones wrote. --skip and --only resume a run in an existing --out-dir.

A run downloads nothing. Every stage starts with HF_HUB_OFFLINE=1 and
SYNTACTIC_ANALYSIS_OFFLINE=1 in its environment (offline.OFFLINE_SWITCHES,
set whatever the shell says), so a model or data package that is not on disk
fails its configuration, the stage and the run instead of being fetched;
setup.sh downloads everything a run loads. Each stage's provenance records
the two switches (`offline`), and run.json the ones the stages were given.
Since 24 Oct 2026.

A run trains nothing either. The five models this project trains (udpipe_ewt,
nltk_ewt, nltk_arc, spacy_ewt, rdr_ewt) are written by their training scripts
(scripts/train_udpipe.py, train_nltk.py, train_spacy.py, train_rdrpos.py;
setup.sh runs them) before the lock, and a missing one fails its
configuration, with the script named, and so the stage and the run. Since
24 Oct 2026; until then the run started with a stage train_udpipe, which ran
scripts/train_udpipe.py, and the NLTK wrappers trained nltk_ewt and nltk_arc
when they were missing.

Usage:
    python scripts/run_pipeline.py                              # exploratory run
    python scripts/run_pipeline.py --max-sentences 20           # smoke run
    python scripts/run_pipeline.py --max-sentences 20 --skip analyze figures
    python scripts/run_pipeline.py --out-dir results/run_2026-10-24 --skip pos
    python scripts/run_pipeline.py --out-dir results/run_2026-10-24 --only analyze figures
    python scripts/run_pipeline.py --out-dir results/run_2026-10-24 --only figures
    python scripts/run_pipeline.py --confirmatory               # step 9, after the lock

Rewritten 23 Oct 2026 for version 1.0 (steps 5 and 7): no agreement stage on
the EWT, no speed benchmark, no statistics or figures of the earlier design;
the earlier pipeline is in the history and under the tag
exploracia-2026-10-17. The figures stage, after analyze, since step 10.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from syntactic_analysis.offline import OFFLINE_SWITCHES
from syntactic_analysis.plan import Plan, load
from syntactic_analysis.project import CSV_SUBDIR, LOG_SUBDIR, project_root
from syntactic_analysis.seal import SealedError, verify_lock
from syntactic_analysis.stage import (
    CON_GOLD,
    default_out_dir,
    environment,
    git_state,
    now,
    portable,
    run_datasets,
)

SCRIPTS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Stage:
    """One stage of the pipeline.

    Attributes:
        key:    The name --skip and --only take.
        script: Its file in scripts/.
        title:  What it does, for the console.
        task:   The task of a measuring stage (pos, dep, con), None otherwise.
    """

    key: str
    script: str
    title: str
    task: str | None = None


PIPELINE: tuple[Stage, ...] = (
    Stage("pos", "pos_tagging.py", "POS: per-sentence UPOS counts", "pos"),
    Stage("dep", "dependency_parsing.py", "dependencies: per-sentence attachment counts", "dep"),
    Stage("con", "constituency_parsing.py", "constituency: per-sentence bracket counts", "con"),
    Stage("analyze", "analyze.py", "estimates, intervals and verdicts from the counts"),
    Stage("figures", "make_figures.py", "the three figures, from the thesis tables"),
)

STAGE_KEYS = tuple(stage.key for stage in PIPELINE)

# The --gold value of the constituency stage for each of its datasets.
_GOLD_OF = {dataset: alias for alias, dataset in CON_GOLD.items()}


@dataclass(frozen=True)
class Command:
    """One invocation of a stage's script: its argv and where its log goes."""

    stage: str
    argv: tuple[str, ...]
    log: str
    dataset: str | None = None


def stage_commands(
    stage: Stage,
    run_dir: Path,
    plan: Plan,
    *,
    max_sentences: int | None = None,
    confirmatory: bool = False,
) -> list[Command]:
    """The commands that run `stage` for this run, in order.

    A measuring stage runs once per dataset the run scores for its task
    (stage.run_datasets); analyze and figures are handed the run directory.
    Until 24 Oct 2026 a stage train_udpipe came first and was handed
    --max-sentences as its --max-train-sentences.

    `sys.executable` rather than "python": every stage must run in this
    interpreter, or a stage picks up other library versions than the
    provenance of its neighbours records.
    """
    script = str(SCRIPTS_DIR / stage.script)
    base = [sys.executable, script]
    if stage.task is None:
        extra = ["--run-dir", str(run_dir)]
        return [Command(stage.key, tuple(base + extra), f"run_{stage.key}.txt")]
    commands = []
    for dataset in run_datasets(plan, stage.task, confirmatory=confirmatory):
        argv = list(base)
        argv += ["--gold", _GOLD_OF[dataset]] if stage.task == "con" else ["--dataset", dataset]
        argv += ["--out-dir", str(run_dir)]
        if max_sentences is not None:
            argv += ["--max-sentences", str(max_sentences)]
        if confirmatory:
            argv.append("--confirmatory")
        commands.append(Command(stage.key, tuple(argv), f"run_{stage.key}_{dataset}.txt", dataset))
    return commands


def recorded_argv(argv: Sequence[str]) -> list[str]:
    """A command as run.json records it: "python" for the interpreter, project paths relative."""
    out = ["python"]
    for arg in argv[1:]:
        out.append(portable(arg) if ("/" in arg or "\\" in arg) else arg)
    return out


def child_environment() -> dict[str, str]:
    """UTF-8 and unbuffered output for every stage, whatever the console's code page,
    and no downloads.

    This script echoes every stage's output, so a stage that wrote in the
    console's legacy code page would arrive as replacement characters; and a
    stage whose output sits in a block buffer shows no progress at all.
    `setdefault` leaves an explicit choice of whoever started the run alone.

    The offline switches (offline.OFFLINE_SWITCHES: HF_HUB_OFFLINE=1,
    SYNTACTIC_ANALYSIS_OFFLINE=1) are set, not defaulted: a run downloads
    nothing, whatever the shell it was started from says.
    """
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.update(OFFLINE_SWITCHES)
    return env


def run_command(command: Command, log_dir: Path) -> int:
    """Run one command, streaming its output to the console and to its log; the exit code."""
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / command.log, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(" ".join(recorded_argv(command.argv)) + "\n\n")
        handle.flush()
        try:
            process = subprocess.Popen(
                list(command.argv), cwd=project_root(), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                bufsize=1, env=child_environment(),
            )
        except OSError as exc:
            message = f"could not start {command.argv[1]}: {exc}\n"
            sys.stdout.write(message)
            handle.write(message)
            return 127
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
        return process.wait()


@dataclass
class StageRecord:
    """One entry of run.json's `stages`: what an invocation did with a stage."""

    key: str
    script: str
    status: str                     # ok | failed | skipped | not run
    commands: list[list[str]] = field(default_factory=list)
    returncodes: list[int] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    invocation: str = ""


# The statuses that say a stage ran. run.json keeps such a record until the
# stage runs again; "skipped" and "not run" leave the earlier record in place.
RAN_STATUSES = ("ok", "failed")


def read_run_record(path: Path) -> dict[str, object]:
    """The run.json an earlier invocation left, or {} (said on stderr when unreadable)."""
    if not path.exists():
        return {}
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = None
    if not isinstance(record, dict):
        print("NOTE: the existing run.json could not be read and is replaced.", file=sys.stderr)
        return {}
    return record


def earlier_stage_records(previous: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """The records of the stages an earlier invocation ran, by key."""
    stages = previous.get("stages")
    return {
        str(old["key"]): old
        for old in (stages if isinstance(stages, list) else ())
        if isinstance(old, dict) and old.get("status") in RAN_STATUSES and "key" in old
    }


def format_duration(seconds: float) -> str:
    """A run's total time as it is read: 42 s, 17 min, 3 h 05 min."""
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} min"
    hours, rest = divmod(int(round(minutes)), 60)
    return f"{hours} h {rest:02d} min"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description=(
            "Measure pos, dep and con on the datasets of plan/v1.yaml, then "
            "analyze the counts and draw the figures, in order, into one run "
            "directory. The models trained here must be trained first "
            "(scripts/train_*.py); a run trains none."
        ),
    )
    parser.add_argument(
        "--max-sentences", type=int, default=None, metavar="N",
        help="score the first N sentences of every dataset (a smoke run, into "
             "results/smoke/v1_<stamp>/ unless --out-dir is given); omit for whole sets",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None, metavar="DIR",
        help="the run directory (default: results/run_<YYYY-MM-DD>, which must not exist yet)",
    )
    parser.add_argument(
        "--confirmatory", action="store_true",
        help="score the exploratory and the confirmatory datasets at the locked "
             "commit (plan/LOCK.json must hold), then analyze them and draw the figures",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--skip", nargs="+", choices=STAGE_KEYS, default=(), metavar="STAGE",
        help="stages to leave out: " + " ".join(STAGE_KEYS) + ". What a skipped "
             "stage writes must already be in --out-dir.",
    )
    selection.add_argument(
        "--only", nargs="+", choices=STAGE_KEYS, default=(), metavar="STAGE",
        help="run only these stages, in pipeline order; the rest must already be in --out-dir",
    )
    args = parser.parse_args(argv)
    if args.max_sentences is not None and args.max_sentences < 1:
        parser.error("--max-sentences must be at least 1")
    if args.confirmatory and args.max_sentences is not None:
        parser.error("--confirmatory scores whole sets, once: it takes no --max-sentences")
    return args


def resolve_run_dir(args: argparse.Namespace, started: datetime) -> Path:
    """The run directory: --out-dir, a smoke directory, or today's run directory."""
    if args.out_dir is not None:
        return args.out_dir.resolve()
    if args.max_sentences is not None:
        return project_root() / "results" / "smoke" / f"v1_{started:%Y%m%d_%H%M%S}"
    return default_out_dir(started.date())


def main(argv: Sequence[str] | None = None) -> int:
    from syntactic_analysis.evaluation.report import utf8_console

    utf8_console(stderr=True)
    args = parse_args(argv)
    started_at = datetime.now().astimezone()
    started = started_at.isoformat(timespec="seconds")
    clock = time.perf_counter()

    plan = load(strict=True)
    run_dir = resolve_run_dir(args, started_at)
    if args.out_dir is None and args.max_sentences is None and run_dir.exists():
        print(f"{portable(run_dir)} exists already. Pass --out-dir to resume that run, "
              f"or choose another directory.", file=sys.stderr)
        return 2
    if args.confirmatory:
        try:
            verify_lock(plan)
        except SealedError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 1

    only = set(args.only)
    skip = {key for key in STAGE_KEYS if key not in only} if only else set(args.skip)
    selected = [stage for stage in PIPELINE if stage.key not in skip]
    (run_dir / CSV_SUBDIR).mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / LOG_SUBDIR

    print("=" * 78)
    print("  RUN PIPELINE - plan/v1.yaml")
    print(f"  run dir       : {portable(run_dir)}")
    print(f"  plan sha256   : {plan.sha256}")
    print(f"  run           : {'confirmatory' if args.confirmatory else 'exploratory'}")
    print(f"  max sentences : {args.max_sentences if args.max_sentences is not None else 'all'}")
    print(f"  downloads     : none ({', '.join(f'{k}={v}' for k, v in OFFLINE_SWITCHES.items())})")
    print(f"  stages        : {' -> '.join(s.key for s in selected) or '(none)'}")
    if skip:
        print(f"  skipping      : {', '.join(k for k in STAGE_KEYS if k in skip)}")
    print("=" * 78)

    previous = read_run_record(run_dir / "run.json")
    earlier = earlier_stage_records(previous)
    invocations = previous.get("invocations")
    invocations = list(invocations) if isinstance(invocations, list) else []

    records: list[StageRecord] = []
    failed: StageRecord | None = None
    for number, stage in enumerate(PIPELINE, start=1):
        record = StageRecord(stage.key, f"scripts/{stage.script}", "skipped", invocation=started)
        records.append(record)
        if stage.key in skip:
            print(f"\n[{number}/{len(PIPELINE)}] {stage.key} - SKIPPED")
            continue
        if failed is not None:
            record.status = "not run"
            continue
        print(f"\n[{number}/{len(PIPELINE)}] {stage.key} - {stage.title}")
        record.status = "ok"
        for command in stage_commands(stage, run_dir, plan, max_sentences=args.max_sentences,
                                      confirmatory=args.confirmatory):
            print(f"  $ {' '.join(recorded_argv(command.argv))}")
            print("-" * 78)
            code = run_command(command, log_dir)
            record.commands.append(recorded_argv(command.argv))
            record.returncodes.append(code)
            record.logs.append(f"{LOG_SUBDIR}/{command.log}")
            print("-" * 78)
            if code != 0:
                record.status = "failed"
                failed = record
                print(f"  {stage.key}: FAILED (exit code {code}); log {LOG_SUBDIR}/{command.log}")
                break
        if record.status == "ok":
            print(f"  {stage.key}: ok")

    total = time.perf_counter() - clock
    finished = now()
    git = git_state()
    invocation = {
        "started": started,
        "finished": finished,
        "total_seconds": round(total, 1),
        "git": git,
        "confirmatory": args.confirmatory,
        "max_sentences": args.max_sentences,
        "selection": "only" if only else ("skip" if args.skip else "all"),
        "selected": [stage.key for stage in selected],
        "completed": failed is None,
    }
    run_record = {
        "schema": 1,
        "run_dir": portable(run_dir),
        "plan": {"path": portable(plan.path), "sha256": plan.sha256},
        "environment": environment(),
        "offline": dict(OFFLINE_SWITCHES),
        "stages": [
            record.__dict__ if record.status in RAN_STATUSES
            else earlier.get(record.key, record.__dict__)
            for record in records
        ],
        "invocations": [*invocations, invocation],
    }
    (run_dir / "run.json").write_text(
        json.dumps(run_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n",
    )

    lines = [
        f"run dir     : {portable(run_dir)}",
        f"plan sha256 : {plan.sha256}",
        f"git         : {git['commit']} ({git['branch']}), dirty={git['dirty']}",
        f"started     : {started}",
        f"finished    : {finished}",
        f"total time  : {format_duration(total)}" + (f" ({total:.0f} s)" if total >= 60 else ""),
        "",
        "stage         status    commands",
    ]
    lines += [f"{record.key:<13} {record.status:<9} {len(record.commands)}" for record in records]
    if failed is not None:
        lines += [
            "",
            f"STOPPED at stage {failed.key}; its output is in {failed.logs[-1]}.",
            "Later stages were not run: each reads what the earlier ones wrote.",
            "After the fix, resume with:",
            f"    python scripts/run_pipeline.py --out-dir {portable(run_dir)} "
            f"--skip {' '.join(r.key for r in records if r.status in ('ok', 'skipped'))}",
        ]
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "pipeline.txt", "a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n\n")
    print()
    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    for line in lines:
        print(f"  {line}")
    print("=" * 78)
    return 1 if failed is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
