# Comparison of tools for generating syntactic analysis

The program of the diploma thesis *Porovnanie nástrojov na generovanie syntaktickej analýzy* (Comparison of tools for generating syntactic analysis), Constantine the Philosopher University in Nitra. Author: Martin Studený.

The work compares off-the-shelf English syntactic analysis tools on three tasks: part-of-speech tagging (UPOS), dependency parsing and constituency parsing. It asks one question: **how much of the differences between the tools does the model paradigm explain, and how much the training resources?**

## Version 1.0

- The experiment is stated once, as data, in `prog/plan/v1.yaml`: 16 configurations, four test sets, the hypotheses H1 and H2, and one statistical procedure (a paired bootstrap with a 95 % confidence interval and a threshold of δ = 1 point).
- The exploratory run took place on 24 Oct 2026, on EWT test and WSJ 01.
- The plan was locked on 25 Oct 2026.
- The confirmatory run followed the same day, on GENTLE and MCTB_en.
- **H1 holds, H2 does not.** Both runs are frozen in `prog/results/evidence/`.

The design, how to run it and the results are described in [`prog/README.md`](prog/README.md), and the data in [`prog/data/DATASET.md`](prog/data/DATASET.md).

## Running

You need Python 3.13, `git` and `bash` (Git Bash on Windows). From `prog/`:

```bash
LOCKED=1 bash setup.sh            # environment, models, data and the five models trained in the project (about 3 h 15 min)
source .venv/Scripts/activate     # Windows (Git Bash); on Linux and Mac .venv/bin/activate
python scripts/smoke_plan.py      # every configuration on 20 sentences
python scripts/run_pipeline.py    # the exploratory run (about 2 h)
```

The confirmatory test sets can be scored only after locking a checkout of the plan of one's own. `prog/README.md`, "The lock and the confirmatory run", gives the steps.

## Licences

- The code is MIT-licensed (`prog/LICENSE`).
- The tools, models and corpora have their own licences (`prog/NOTICE`).
- No corpus or model is part of this repository: `setup.sh` downloads them and checks their SHA-256.
