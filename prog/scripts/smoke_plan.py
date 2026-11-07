"""Smoke test of the plan: build every configuration and run it on a few sentences.

    python scripts/smoke_plan.py                     # every configuration of plan/v1.yaml
    python scripts/smoke_plan.py --key spacy_ewt     # one (repeatable)
    python scripts/smoke_plan.py --sentences 5       # fewer sentences (at most 20)

Every configuration of the plan, loaded strictly, is built the way a stage
builds it (build.build: constructed from `wrapper` and `init`, given its
`inputs`, loaded with `load`, and checked against what the plan says it is)
and run on the first sentences of the exploratory dataset of each task it is
a member of: EWT test for pos and dep, WSJ section 01 for con, since the EWT
has no constituency gold. Which dataset that is comes from the plan (the one
exploratory dataset listing the task), and seal.open_dataset opens it; a
confirmatory dataset is never opened here.

At most 20 sentences per configuration and dataset: the brief allows smoke
tests of that size at any time, and full runs only on the owner's word. The
PCFG keeps its cap (max_sentence_length in the plan): a longer sentence is
its declared failure and counts as not parsed.

Per configuration it reports whether it was built and loaded, the load time
and the model_version the wrapper records, and per task the sentences
processed (for con, how many got a tree) and the plan's metric on them:
UPOS accuracy, LAS (with UAS), bracket F1 under the plan's EVALB parameter
set. Scores of 20 sentences say that a configuration runs and that its output
lines up with the gold, not how good it is; the times depend on the load of
the machine. A failure of one configuration is recorded with its message and
the others still run; the exit status is 1 if any failed.

Output: results/smoke/plan_<stamp>/ (git ignores results/smoke/), smoke.txt
with the table printed and smoke.json with everything, the provenance of the
run included. Added 23 Oct 2026 (version 1.0, step 4).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from syntactic_analysis.build import build
from syntactic_analysis.evaluation.evalb import con_sentence_counts, plan_parameter_set
from syntactic_analysis.evaluation.metrics import (
    corpus_metric,
    dep_sentence_counts,
    pos_sentence_counts,
)
from syntactic_analysis.evaluation.provenance import (
    collect_provenance,
    recorded_path,
    run_stamp,
    script_path,
    stage_started,
)
from syntactic_analysis.plan import TASKS, Plan, load
from syntactic_analysis.project import project_root
from syntactic_analysis.seal import open_dataset

# The most sentences a smoke test may run per configuration and dataset.
MAX_SENTENCES = 20


def exploratory_dataset(plan: Plan, task: str) -> str:
    """The one exploratory dataset of `task` in the plan.

    Raises:
        ValueError: If the plan has none or several.
    """
    found = [
        key for key, dataset in plan.datasets.items()
        if task in dataset.tasks and dataset.status == "exploratory"
    ]
    if len(found) != 1:
        raise ValueError(
            f"task {task} needs exactly one exploratory dataset for a smoke test; "
            f"the plan has {found}"
        )
    return found[0]


def run_task(analyzer, task: str, gold, n: int, params) -> dict:
    """Run `analyzer` on the first `n` sentences of `gold` for `task`, and score them."""
    words = gold.pretok[:n]
    started = time.perf_counter()
    if task == "con":
        parse = analyzer.analyze_constituency(words)
        seconds = time.perf_counter() - started
        counts = [
            con_sentence_counts(tree, gold_tree, params=params, expected_leaves=len(sentence))
            for tree, gold_tree, sentence in zip(parse.trees, gold.trees[:n], words)
        ]
        return {
            "sentences": len(parse.trees),
            "parsed": sum(c.parsed for c in counts),
            "bracket_f1": round(corpus_metric(counts), 2),
            "seconds": round(seconds, 1),
        }
    parse = analyzer.analyze_pos(words) if task == "pos" else analyzer.analyze_dependency(words)
    seconds = time.perf_counter() - started
    pairs = list(zip(parse.sentences, gold.parse.sentences[:n]))
    if task == "pos":
        counts = [pos_sentence_counts(p, g) for p, g in pairs]
        return {
            "sentences": len(parse.sentences),
            "tokens": sum(c.total for c in counts),
            "upos": round(corpus_metric(counts), 2),
            "seconds": round(seconds, 1),
        }
    counts = [dep_sentence_counts(p, g) for p, g in pairs]
    return {
        "sentences": len(parse.sentences),
        "tokens": sum(c.total for c in counts),
        "las": round(corpus_metric(counts, "las"), 2),
        "uas": round(corpus_metric(counts, "uas"), 2),
        "multi_root": sum(
            sum(1 for token in s.tokens if token.head_index == 0) > 1 for s in parse.sentences
        ),
        "seconds": round(seconds, 1),
    }


def _failure(exc: BaseException) -> str:
    """One line for a failure: the exception's type and the first line of its message."""
    lines = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {lines[0] if lines else ''}"


def smoke(plan: Plan, keys: list[str], n: int, golds: dict) -> list[dict]:
    """Build each configuration in `keys` and run it on every task it is a member of.

    `golds` maps each dataset exploratory_dataset names to its gold set.
    """
    datasets = {task: exploratory_dataset(plan, task) for task in TASKS}
    params = plan_parameter_set(plan.tasks["con"].metric.options)
    # An input built for one configuration is kept for the next that needs it,
    # and a configuration that is another's input is kept once built.
    inputs = {target for config in plan.configurations.values() for target in config.inputs.values()}
    built: dict = {}
    rows = []
    for key in keys:
        config = plan.configurations[key]
        tasks = [task for task in TASKS if key in plan.tasks[task].members]
        row = {"key": key, "wrapper": config.wrapper, "tasks": tasks, "loaded": False}
        print(f"{key}: building ({config.wrapper})", flush=True)
        started = time.perf_counter()
        try:
            analyzer = built.get(key) or build(plan, key, built=built)
        except Exception as exc:  # noqa: BLE001 - recorded, and the exit status says so
            row["failure"] = f"build: {_failure(exc)}"
            print(f"  FAILED to build: {row['failure']}", flush=True)
            rows.append(row)
            continue
        row["loaded"] = True
        row["load_seconds"] = round(time.perf_counter() - started, 1)
        row["model_version"] = analyzer.model_version
        row["results"] = {}
        for task in tasks:
            dataset = datasets[task]
            try:
                result = run_task(analyzer, task, golds[dataset], n, params)
            except Exception as exc:  # noqa: BLE001 - recorded, and the exit status says so
                result = {"failure": _failure(exc)}
            result["dataset"] = dataset
            row["results"][task] = result
            print(f"  {task} on {dataset}: {result}", flush=True)
        rows.append(row)
        if key in inputs:
            built[key] = analyzer
        del analyzer
        gc.collect()
    return rows


def _cell(result: dict | None) -> str:
    if result is None:
        return ""
    if "failure" in result:
        return "FAILED"
    if "bracket_f1" in result:
        return f"{result['parsed']}/{result['sentences']} parsed, F1 {result['bracket_f1']:.2f}"
    if "upos" in result:
        return f"{result['sentences']} sent, UPOS {result['upos']:.2f}"
    return f"{result['sentences']} sent, LAS {result['las']:.2f}"


def table(rows: list[dict]) -> list[str]:
    """The report: one line per configuration, one column per task."""
    header = ("key", "loaded", "load s", "pos", "dep", "con")
    lines = [" | ".join(header), " | ".join("---" for _ in header)]
    for row in rows:
        results = row.get("results", {})
        lines.append(" | ".join((
            row["key"],
            "yes" if row["loaded"] else "NO",
            str(row.get("load_seconds", "")),
            *(_cell(results.get(task)) for task in TASKS),
        )))
    failures = [
        f"{row['key']} {task}: {result['failure']}"
        for row in rows for task, result in row.get("results", {}).items() if "failure" in result
    ] + [f"{row['key']}: {row['failure']}" for row in rows if "failure" in row]
    if failures:
        lines += ["", "Failures:", *(f"  {line}" for line in failures)]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--key", action="append", dest="keys", metavar="KEY",
                        help="a configuration of the plan (repeatable); default: all")
    parser.add_argument("--sentences", type=int, default=MAX_SENTENCES,
                        help=f"sentences per configuration and dataset, 1-{MAX_SENTENCES}")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: results/smoke/plan_<stamp>")
    args = parser.parse_args(argv)
    if not 1 <= args.sentences <= MAX_SENTENCES:
        parser.error(f"--sentences must be between 1 and {MAX_SENTENCES}")

    started = stage_started()
    plan = load(strict=True)
    keys = args.keys or list(plan.configurations)
    unknown = [key for key in keys if key not in plan.configurations]
    if unknown:
        parser.error(f"unknown configuration(s) {unknown}; expected some of {list(plan.configurations)}")
    out_dir = args.out_dir or project_root() / "results" / "smoke" / f"plan_{run_stamp(started)}"

    datasets = sorted({exploratory_dataset(plan, task) for task in TASKS})
    golds = {dataset: open_dataset(plan, dataset) for dataset in datasets}
    rows = smoke(plan, keys, args.sentences, golds)
    lines = table(rows)
    print()
    print("\n".join(lines))

    provenance = collect_provenance(
        script=script_path(__file__),
        timestamp=started,
        config={
            "plan": recorded_path(plan.path),
            "plan_sha256": plan.sha256,
            "sentences": args.sentences,
            "datasets": {key: gold.sha256 for key, gold in golds.items()},
        },
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "smoke.json").write_text(
        json.dumps({"provenance": asdict(provenance), "rows": rows}, indent=2,
                   ensure_ascii=False, default=str),
        encoding="utf-8", newline="\n",
    )
    print(f"\nwritten to {recorded_path(out_dir)}")
    failed = any("failure" in row or any("failure" in r for r in row.get("results", {}).values())
                 for row in rows)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
