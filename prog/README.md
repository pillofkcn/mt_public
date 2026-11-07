# Syntactic Analysis

A comparison of off-the-shelf English syntactic analysis tools on three tasks: part-of-speech tagging (UPOS), dependency parsing and constituency parsing. Version 1.0 asks one question of them: **how much of the differences between the tools does the model paradigm explain, and how much the training resources?**

The experiment is stated once, as data, in **`plan/v1.yaml`**: the 16 configurations, their method level and roles in each task, the four test sets with their status, the hypotheses and the statistical procedure. The stages, the analysis and the figures take the configurations, the contrasts and the parameters from the plan, not from code, and `syntactic_analysis.plan.load()` refuses a plan that does not hold together (an unknown key, a missing wrapper, a broken reference, a chain that does not climb in level).

**Results of version 1.0.** The exploratory run took place on 24 Oct 2026, the plan was locked on 25 Oct 2026 (`plan/LOCK.json`) and the confirmatory run followed the same day: **H1 holds, H2 does not.** Both runs are frozen in `results/evidence/` ("Results of version 1.0" below); what was found after the confirmatory run is in `plan/deviations.md`.

## Research design (version 1.0)

**Method levels.** The level is declared per task, because one tool can sit at two levels — UDPipe's tagger is an averaged perceptron (L1) while its parser is Parsito, a transition-based parser with a feed-forward neural-network classifier (L2).

| Level | Paradigm |
|---|---|
| L0 | rules |
| L1 | statistical model over discrete features |
| L2 | shallow neural network (feed-forward or convolutional) |
| L3 | BiLSTM encoder |
| L4 | pretrained transformer |

**RQ1 – paradigm.** How does accuracy change when the paradigm changes and the training data stay the same?

- **H1.** With the same training data a higher-level model is more accurate (H1a), and the step to the pretrained transformer (L4) gains more for dependencies than for POS (H1b). L0 does not enter H1; for POS neither does L2 (`spacy_ewt`), because the literature predicts L1 ≈ L2 there (Kyle and Eguchi 2024).
  - **H1a** – for every step of an H1a chain, the lower bound of the 95 % CI of Δ = M(higher) − M(lower) lies above 0.
  - **H1b** – the step `stanza_comb` → `stanza_trf`, LAS against UPOS on the same documents: the lower bound of the 95 % CI of Δ_LAS − Δ_UPOS lies above 0. Computed on the EWT test set (exploratory) and on GENTLE (confirmatory); GENTLE gives the verdict. Constituents do not enter H1b.

**RQ2 – resources.** How does accuracy change when the data change and the paradigm stays the same, and is that change larger than a paradigm step?

- **H2.** A mismatch between the training data and the test changes accuracy more than any paradigm step of H1a; related data help only when they match the test. Δ_step,max is the largest point estimate of an H1a step in the same task on the same test set.
  - **H2a – training data.** The same algorithm with its shipped model (other training resources) is worse on EWT than after training on EWT train: for every H2a pair, the lower bound of the 95 % CI of |Δ_source| − Δ_step,max lies above 0. Computed on the EWT test set only (`hypotheses.h2a.datasets` in the plan), so H2a is exploratory only; its pairs are not computed on GENTLE, not even descriptively, and `nltk_pre` and `spacy_sm` are not scored on GENTLE at all.
  - **H2b – test data.** The same model loses more than Δ_step,max on a new genre (EWT → GENTLE for POS and dependencies, WSJ 01 → MCTB_en for constituents): for every model of an H1a chain, the lower bound of the 95 % CI of (M_in − M_out) − Δ_step,max lies above 0, with Δ_step,max taken on the in-domain set.
  - **H2c – related data.** Stanza's `combined` package (EWT + GUM + others) against its `ewt` package: practically equivalent on EWT (the whole 95 % CI inside (−δ, +δ)), and better on GENTLE, which follows the conventions of GUM (the lower bound of the CI above +δ).
- A hypothesis with several parts holds only if every part holds, decided separately for each status. A whole hypothesis (H1 = H1a + H1b, H2 = H2a + H2b + H2c) is decided, for a status, from the parts the plan defines for that status: a component with no set of that status – H2a on the confirmatory sets, H2b on the exploratory ones – does not enter it, and the note of its row in `thesis_verdicts.csv` says so. Speed is not measured and is not part of any hypothesis.

**Descriptive additions** (no hypothesis, no criterion; the labels D2–D4 are those of the plan):

- **D2 – PCFG coverage.** The share of sentences the PCFG gives a tree, by failure reason (unknown word, no derivation, over the length limit), and the F1 of the PCFG and of the neural parsers on exactly the sentences the PCFG processed.
- **D3 – annotation scheme.** spaCy `en_core_web_sm` (ClearNLP/SD relations mapped to UD) against the UD tools, LAS and CLAS, on the EWT test set only.
- **D4 – the L1 → L2 step for POS.** `nltk_ewt` against `spacy_ewt`, UPOS, Δ with its CI.

**Status.** The EWT test set and WSJ section 01 are **exploratory**: they were analysed in the earlier phase of the work, and the hypotheses of version 1.0 were formulated after those results. GENTLE and MCTB_en are **confirmatory**: their first and only evaluation took place after the plan was locked, and the confirmatory verdicts come from them alone. Exploratory and confirmatory results are reported separately, in separate rows, report sections and figure markers.

**The statistical procedure (one).** Δ = M(A) − M(B) in percentage points, where M is the task's metric recomputed from summed per-sentence counts. Its 95 % CI comes from a paired percentile bootstrap, B = 10 000, seed 20260101: the resampling units of a set are drawn with replacement, with the same indices for every configuration scored on that set, and the metric is recomputed from the summed counts of each replicate. A difference of differences (H1b; |Δ_source| − Δ_step,max in H2a) is taken from the same replicates; the two sets of H2b are resampled independently, with Δ_step,max from the in-domain replicate. Each CI is read against the threshold δ = 1.0 point (sensitivity at 0.5 and 2.0, and with the sentence as the resampling unit) by one function, `classify(ci, delta)`, into one of six classes: better by more than δ, better but not shown by more than δ, practically equivalent, inconclusive, and the two mirror classes for worse. No p-values and no corrections: each claim is decided by its own interval, and the half-width of every CI is reported beside it. One more sensitivity row is descriptive only (the plan's `statistics.subset_sensitivity: evalb_both_parsers`): a sentence EVALB cannot score adds nothing to its parser's F1, so each Δ between two constituency parsers – every H1a step of the task on every set, and the Δ_step,max term of its H2b rows on the source set – is also computed on the sentences EVALB scores for both parsers, with their number, by the same unit and index matrix (a filter on the counts, not a new draw); it has no class and no verdict and changes none. The code is `src/syntactic_analysis/evaluation/estimate.py` and `scripts/analyze.py`.

## Configurations

The 16 configurations of `plan/v1.yaml`. A task cell gives the configuration's level in that task and its roles there, as the plan names them; – means the configuration is not a member of the task.

| Key | POS | Dependencies | Constituents | Tool and model | Training data |
|---|---|---|---|---|---|
| `rdr_ewt` | L0: descriptive | – | – | RDRPOSTagger retrained on EWT train with its authors' learner (`scripts/train_rdrpos.py`); optional in the plan | `ewt_train_snapshot` |
| `nltk_pre` | L1: H2a | – | – | NLTK `averaged_perceptron_tagger_eng` as shipped; PTB tags converted to UPOS | `nltk_tagger_unstated` |
| `nltk_ewt` | L1: H1a, H2a, H2b, D4 | – | – | NLTK averaged perceptron trained here on EWT train | `ewt_train_snapshot` |
| `udpipe_ewt` | L1: descriptive | L2: H1a, H2b | – | UDPipe 1 refitted on EWT train by `scripts/train_udpipe.py`; its tags feed `nltk_arc` | `ewt_train_snapshot` |
| `spacy_sm` | L2: H2a | L2: H2a, D3 | – | spaCy `en_core_web_sm` 3.8.0; ClearNLP (SD) relations mapped to UD | `ontonotes5` |
| `spacy_ewt` | L2: H2a, D4 | L2: H2a | – | spaCy pipeline (tok2vec, morphologizer, parser) trained on EWT train by `scripts/train_spacy.py` | `ewt_train_snapshot` |
| `stanza_ewt` | L3: H1a, H2b, H2c | L3: H1a, H2b, H2c | – | Stanza, package `ewt` (`ewt_charlm`) | `ewt_train_r2_15` |
| `stanza_comb` | L3: H1a, H1b, H2b, H2c | L3: H1a, H1b, H2b, H2c | – | Stanza, package `combined` (`combined_charlm`), without its constituency parser | `ud_combined_r2_15` |
| `stanza_trf` | L4: H1a, H1b, H2b | L4: H1a, H1b, H2b | – | Stanza `combined` with pos and depparse `combined_electra-large` (BiLSTM over frozen ELECTRA-large) | `ud_combined_r2_15` |
| `nltk_arc` | – | L1: H1a, H2b | – | NLTK TransitionParser, arc-eager, logistic regression in place of NLTK's SVC; UPOS from `udpipe_ewt` | `ewt_train_snapshot` |
| `pcfg_words` | – | – | L1: D2 | NLTK ViterbiParser, PCFG induced from WSJ 00 (treebank grammar), word terminals | `wsj_00_nltk_sample` |
| `pcfg_tags` | – | – | L1: D2 | the same PCFG over tag terminals; tags from the shipped NLTK perceptron | `wsj_00_nltk_sample` |
| `stanza_con` | – | – | L3: H1a, H2b | Stanza constituency `ptb3-revised_charlm`; tags from `combined_charlm` | `ptb_revised_wsj_02_21` |
| `stanza_con_trf` | – | – | L4: H1a, H2b | Stanza constituency `ptb3-revised_electra-large`; tags from `combined_charlm` | `ptb_revised_wsj_02_21` |
| `benepar` | – | – | L4: H1a, H2b | Benepar `benepar_en3` (T5-small encoder) over spaCy `en_core_web_sm` | `ptb_revised_wsj_02_21` |
| `supar_con` | – | – | L3: descriptive | SuPar `crf-con-en` (CharLSTM) | `ptb3_wsj_02_21` |

Training data (`training_data` in the plan; `data/DATASET.md` has the details):

- `ewt_train_snapshot` – UD_English-EWT train, `data/en_ewt-ud-train.conllu`: the repository's files of 23 Mar 2026, byte for byte those of release r2.17
- `ewt_train_r2_15` – UD_English-EWT r2.15 train (Stanza's `ewt` package)
- `ud_combined_r2_15` – UD 2.15: EWT, GUM and GUMReddit train, PUD and Pronouns test, and Stanza's hand-parsed English sentences (Stanza's `combined` package)
- `nltk_tagger_unstated` – the corpus is not stated in NLTK's documentation (PTB tags)
- `ontonotes5` – OntoNotes 5, dependencies in the ClearNLP conversion
- `wsj_00_nltk_sample` – WSJ section 00 of the NLTK Penn Treebank sample (1 921 sentences)
- `ptb_revised_wsj_02_21` – Penn Treebank revised (LDC2015T13), WSJ sections 02–21
- `ptb3_wsj_02_21` – Penn Treebank 3, WSJ sections 02–21

What the hypotheses compare (all from the plan):

- **H1a chains** – POS: `nltk_ewt` → `stanza_ewt` and `stanza_comb` → `stanza_trf`; dependencies: `nltk_arc` → `udpipe_ewt` → `stanza_ewt` and `stanza_comb` → `stanza_trf`; constituents: `stanza_con` → `stanza_con_trf` and `stanza_con` → `benepar`. In the EWT chains `stanza_ewt` is trained on EWT r2.15, not on r2.17; the plan records that as an approximation (`approximations`).
- **H1b** – `stanza_comb` → `stanza_trf`, LAS against UPOS, on `ewt_test` and `gentle`; the verdict is `gentle`'s.
- **H2a pairs** – `nltk_pre` → `nltk_ewt` (POS), `spacy_sm` → `spacy_ewt` (POS and dependencies), on `ewt_test` only.
- **H2b transitions** – `ewt_test` → `gentle` for the POS and dependency members of the H1a chains, `wsj_01` → `mctb_en` for `stanza_con`, `stanza_con_trf` and `benepar`.
- **H2c** – `stanza_ewt` → `stanza_comb` for POS and dependencies: equivalence on `ewt_test`, superiority on `gentle`.
- **D2** – `pcfg_words` and `pcfg_tags` against `stanza_con`, `stanza_con_trf`, `benepar` and `supar_con`, on `wsj_01` and `mctb_en`. The PCFG runs on a sample of 250 sentences (seed 20260101) and refuses sentences over 40 tokens; every other configuration runs on whole sets.
- **D3** – `spacy_sm` against `nltk_arc`, `udpipe_ewt`, `spacy_ewt`, `stanza_ewt`, `stanza_comb` and `stanza_trf`, LAS and CLAS. **D4** – `nltk_ewt` → `spacy_ewt`, UPOS. D3 and D4 are computed on the sets of their task where the plan scores every one of their members: D3 on `ewt_test` only, D4 on `ewt_test` and `gentle`.
- **Which sets a configuration is scored on** – every set of its tasks, unless the plan restricts it (`configurations.<key>.datasets`). `nltk_pre` and `spacy_sm` are scored on `ewt_test` only: their H2a pairs are computed there only, so they have no score on GENTLE, in POS or in dependencies, and a stage on GENTLE neither builds nor runs them. `plan.py` refuses a restriction that leaves out a set a hypothesis or a descriptive section needs the configuration on. In dependencies, where its only role is H2a, `spacy_ewt` stays scored on GENTLE, as a score only.
- The drop from the source to the target set of a transition is also reported, descriptively, for every member of the task whose only role there is `descriptive` and which the plan scores on both sets (`supar_con`; `udpipe_ewt` and `rdr_ewt` for POS).

## Metrics

One primary metric per task, micro-averaged over the corpus from summed per-sentence counts; the input is always the gold tokenization and the gold sentence boundaries.

| Task | Metric | Rule |
|---|---|---|
| POS | UPOS accuracy | correct / all tokens |
| Dependencies | LAS | as the official CoNLL 2018 script: relation subtypes are stripped on both sides (`obl:tmod` → `obl`), punctuation counts |
| Constituents | bracket F1 | EVALB with `COLLINS.prm`; `NML` nodes collapsed in the prediction and in the gold (their children move to the parent); a sentence without a tree contributes 0 proposed and 0 matched brackets |

UAS and CLAS (CoNLL 2018, without subtypes) are computed for D3 and as checks. The per-sentence counts (`evaluation/metrics.py`, `evaluation/evalb.py`) were checked against the official `conll18_ud_eval.py` and against the EVALB program by the project's tests (not published here), with two limits: the official script refuses a file with a multi-root tree, which spaCy's `en_core_web_sm` sometimes produces, and `nltk_arc` too when NLTK's transition parser leaves a word unattached (an extra root with an empty relation), so the agreement holds for single-root trees (the counts score such a tree word by word); and `supar_con` writes no POS tags, so its trees are scored with the gold tag of each word on its preterminals – a rule EVALB itself does not have.

spaCy's `en_core_web_sm` emits ClearNLP (Stanford Dependencies) relations, which `deprel_map.py` converts to UD ([de Marneffe et al., 2014](https://aclanthology.org/L14-1045/)) at one of three tiers: `none` (raw output), `casing` (only `ROOT` → `root`, notation rather than linguistics) and `full` (casing plus genuine relabelling, `dobj` → `obj`, …). The plan uses `full` for `spacy_sm` and `casing` for `spacy_ewt`, whose relations are UD already and whose `ROOT` is spaCy's converter's spelling of `root`. Mapping can only rename labels: SD and UD also differ in head attachment for prepositions, copulas, auxiliaries and coordination, which no relabelling fixes – which is why D3 reports CLAS (content-word LAS, which leaves those function words out) beside LAS.

## Datasets

| Plan id | Corpus | Task | Status | Resampling unit | Size |
|---|---|---|---|---|---|
| `ewt_test` | UD_English-EWT test, release r2.17 | POS, dependencies | exploratory | document (`# newdoc id`, else the `sent_id` prefix): 316 | 2 077 sentences |
| `wsj_01` | Penn Treebank WSJ section 01, NLTK sample | constituents | exploratory | file (`wsj_0100` … `wsj_0199`): 100 | 1 993 sentences |
| `gentle` | UD_English-GENTLE, UD 2.18 | POS, dependencies | confirmatory | document: 26 | 1 334 sentences |
| `mctb_en` | MCTB_en (Yang et al. 2022), five domains | constituents | confirmatory | sentence | 5 000 trees |

`data/DATASET.md` records each one's source, version, SHA-256, counts, licence and the properties a result on it depends on, and the training data the project trains on. No corpus is part of this repository: `setup.sh` downloads them and checks each file against its recorded SHA-256 (`data/*.sha256`).

**The seal.** GENTLE and MCTB_en may be downloaded, hashed and counted, and nothing else, until the plan is locked: no analyzer runs on them and nothing of them is printed. A stage can open a test set only through `syntactic_analysis.seal.open_dataset`, which refuses a confirmatory one unless the stage was started with `--confirmatory` and `plan/LOCK.json` holds.

## How to run

Every command runs from `prog/`. The examples use `python` inside the activated virtual environment; `.venv\Scripts\python.exe` (Windows) or `.venv/bin/python` without activating it does the same.

### 1. Setup

```bash
LOCKED=1 bash setup.sh          # the exact versions of version 1.0 (requirements-lock.txt), Python 3.13
bash setup.sh                   # or the newest versions the ranges of pyproject.toml allow
source .venv/Scripts/activate   # Windows (Git Bash)
source .venv/bin/activate       # Linux/Mac
```

`setup.sh` does everything a run needs, in this order:

1. It checks the Python version (`PYTHON=python3.13`, or `PYTHON="py -3.13"` on Windows, picks the interpreter).
2. It creates `.venv` and installs the project with every optional group. With `LOCKED=1` it installs the exact versions of `requirements-lock.txt`; without it, the newest versions the ranges of `pyproject.toml` allow.
3. It downloads the published models:
   - spaCy `en_core_web_sm`;
   - the NLTK data: `punkt_tab`, `averaged_perceptron_tagger_eng` and the `treebank` sample;
   - `benepar_en3`;
   - the models of every Stanza configuration: the `ewt` and `combined` packages, the ELECTRA-large POS, dependency and constituency models, and the transformer `google/electra-large-discriminator` they read (about 1.3 GB, at the revision the Stanza wrapper records);
   - SuPar's `crf-con-en`.
4. It downloads UD_English-EWT r2.17, plus GENTLE and MCTB_en (`scripts/fetch_data.py`), and checks every file against `data/*.sha256`.
5. It trains the five models of step 3: about 3 h 15 min from zero, seconds when they are there.

It builds and loads every Stanza and Benepar configuration the way a run does (`build.build`), so each model lands where its wrapper looks for it.

A run itself downloads nothing and trains nothing. A published model that is not on disk, or a model trained here that is missing, fails the configuration that needs it, the stage and the run.

### 2. The test sets

```bash
python scripts/fetch_data.py                 # download GENTLE and MCTB_en, check them, count them
python scripts/fetch_data.py --verify-only   # no network: check the local copy against data/*.sha256
```

`setup.sh` runs the first command. `fetch_data.py` downloads every dataset of the plan that has a `source`: a repository and a pinned tag or commit.

- It checks each file against the digest recorded in `data/`.
- It checks the format by counting, and prints counts and digests only.
- It refuses a file that does not match, and never overwrites it.
- `--dataset ID` (repeatable) limits it to some datasets.

The EWT is downloaded by `setup.sh` itself. The WSJ sections come with NLTK's `treebank` sample.

### 3. The models trained here

These models are trained by their training scripts only. A build or a run never trains one: a configuration whose model is missing fails with `ModelNotTrainedError`, naming its script – in `build.build`, in a stage and in `smoke_plan.py`. `setup.sh` runs the four scripts in this order; each does nothing when its model is there (`--force` trains again).

| Configuration | Trained by | Written to (`model_cache/`, ignored by git) |
|---|---|---|
| `udpipe_ewt` | `python scripts/train_udpipe.py` (about 100 min on a CPU) | `english-ewt-retrained.udpipe` and its record `english-ewt-retrained.udpipe.json` |
| `nltk_ewt`, `nltk_arc` | `python scripts/train_nltk.py` (about 25 s and 6 min; `--key` trains one, `--out-dir` writes elsewhere) | `nltk_perceptron_ud_ewt_train/` (three json files); `nltk_transition-arc-eager.pkl` and `.classifier.pkl` |
| `spacy_ewt` | `python scripts/train_spacy.py` (about an hour on a CPU) | `spacy_ewt/`: `model-best/`, `training_record.json`, the converted corpus, config and log |
| `rdr_ewt` | `python scripts/train_rdrpos.py` (about 20 min) | `rdrpos_ewt/`: the `.RDR` rules, the `.DICT` lexicon and `training_record.json`; the learner is kept outside the repository (`--learner-dir`, by default `~/.cache/rdrpostagger/<commit>`) |

**Which data they read:**
- All of them train on the EWT train split; none reads the test split.
- `train_udpipe.py` and `train_spacy.py` use the dev split for early stopping and model selection, and `train_rdrpos.py` checks its model on it.
- The wrappers of `spacy_ewt` and `rdr_ewt` refuse a model without its training record.
- `train_nltk.py` trains in an interpreter started with `PYTHONHASHSEED=20260101`, the value `train_rdrpos.py` gives its learner.

**How far a retraining repeats the model.** A retrained model gives the same outputs, but its files are not the same bytes:
- On all 2 077 sentences of the EWT test set, retrained models give the same tags, and the same heads and relations when both parsers get the same tags of `udpipe_ewt`.
- The bytes differ because the perceptron's classes file can list the classes in another order, and the parser pickle records the training times.
- So reproducibility is checked by identical outputs, not identical bytes.

### 4. Smoke test of the plan

```bash
python scripts/smoke_plan.py                    # every configuration, 20 sentences
python scripts/smoke_plan.py --key spacy_ewt --sentences 5
```

Builds every configuration of the plan as a stage builds it (`build.build`: constructed from the plan, given its inputs, loaded, checked against what the plan says it is) and runs it on the first sentences of the exploratory set of each of its tasks – at most 20. Output in `results/smoke/plan_<stamp>/` (`smoke.txt`, `smoke.json`); scores of 20 sentences say that a configuration runs, not how good it is.

### 5. The exploratory run

From `prog/`, while nothing else runs on the machine:

```bash
python scripts/run_pipeline.py
```

**What it runs.** `pos` → `dep` → `con` → `analyze` → `figures`, on the exploratory sets: EWT test for POS and dependencies, WSJ 01 for constituents.
- The output goes into `results/run_<YYYY-MM-DD>/`.
- It refuses a directory that already exists. `--out-dir` names another, and `--skip` / `--only` resume a run in it.

**How long it takes.** About 1 h 40 min to 2 h on an unloaded machine, about half of it `pcfg_tags`. With other work on the machine, times change up to 2.5-fold. The console must stay open until the run ends.

**A run downloads nothing.** `run_pipeline.py` starts every stage with `HF_HUB_OFFLINE=1` and `SYNTACTIC_ANALYSIS_OFFLINE=1`, whatever the shell has set (`src/syntactic_analysis/offline.py`):
- `HF_HUB_OFFLINE=1` keeps `transformers` off the Hugging Face hub.
- `SYNTACTIC_ANALYSIS_OFFLINE=1` is the project's own switch. It covers Stanza, NLTK, Benepar and SuPar, which download with their own code that the hub's switch does not reach.

A published model or data package that is not on disk fails the configuration with `NotCachedError` instead of being fetched in the middle of a run. Run `setup.sh`, then resume with `--out-dir` and `--skip`. Every stage's provenance records the two switches.

A smoke run of the whole pipeline on the first 20 sentences of each exploratory set goes to `results/smoke/v1_<stamp>/`:

```bash
python scripts/run_pipeline.py --max-sentences 20
```

The stages can run one by one; without `--out-dir` a stage writes into `results/run_<YYYY-MM-DD>/`, the directory a full run of that day then refuses, so give a smoke stage its own directory:

```bash
python scripts/pos_tagging.py --max-sentences 20 --out-dir results/smoke/pos20
python scripts/dependency_parsing.py --max-sentences 20 --keys stanza_ewt,stanza_comb --out-dir results/smoke/dep20
python scripts/constituency_parsing.py --gold wsj01 --max-sentences 20 --out-dir results/smoke/con20
# the analysis and the figures of a run whose stages have written their counts:
python scripts/analyze.py --run-dir results/run_<YYYY-MM-DD>
python scripts/make_figures.py --run-dir results/run_<YYYY-MM-DD>
```

**Stage options:**
- `pos_tagging.py` and `dependency_parsing.py` take `--dataset ID`; the default is the task's exploratory set.
- `constituency_parsing.py` takes `--gold wsj01|mctb`.
- All three take `--confirmatory`, `--out-dir`, `--max-sentences N` and `--keys K1,K2`.
- `analyze.py` and `make_figures.py` read only what a run wrote, and take `--plan` for another plan file.
- `analyze.py` analyses counts only under the plan they were measured under, identified by the plan's SHA-256 in their provenance.

### 6. The lock and the confirmatory run

**`plan/LOCK.json` is the record of the lock of version 1.0.** It was written on 25 Oct 2026, after the exploratory run and before the first confirmatory result. It holds:
- `plan_sha256`: the digest of `plan/v1.yaml`, CRLF read as LF.
- `commit`: the commit of the locked code.
- `data`: the SHA-256 of every file of every dataset of the plan.
- `models`: the SHA-256 of every file of the five models trained here.
- `downloaded_models`: for every configuration, the SHA-256 of every downloaded file its load reads, root by root.
- `versions`: the versions of the libraries and the models.
- `locked_at`: the time of the lock.

**Why it does not hold in this repository.** The commit the lock names belongs to the development history of the work, which is not part of this repository. `seal.verify_lock` therefore refuses this lock here. It would refuse it on another machine as well, because a model trained anew is not the same bytes (step 3). `src/`, `scripts/` and `plan/v1.yaml` in this repository are byte for byte the locked code and plan.

`seal.verify_lock` accepts a lock only if all of the following hold:
- the plan matches its digest;
- nothing under `src/`, `scripts/` or `plan/v1.yaml` has changed since the locked commit, committed or not;
- every dataset file, every model trained here and every downloaded file is exactly the one the lock records, with none missing and none added;
- `LOCK.json` itself is committed and unchanged.

The confirmatory run checks the lock before the first stage and again in every stage. Right before each configuration is loaded, it checks that configuration's model files once more (`seal.verify_model`).

**To score the confirmatory sets on your own machine,** lock your own checkout after `setup.sh` has downloaded the data and trained the models. From `prog/`:

```bash
git rm plan/LOCK.json && git commit -m "Remove the lock of version 1.0"
python scripts/lock_plan.py --commit HEAD --write
git add plan/LOCK.json && git commit -m "Lock of the plan"
python scripts/run_pipeline.py --confirmatory
```

`lock_plan.py` computes the lock from the files on your machine. It refuses, writing nothing, in any of these cases:
- a lock exists already;
- HEAD is not the named commit;
- anything in `prog/` is uncommitted or untracked;
- a dataset file or a model file is missing or does not match its recorded digest.

The confirmatory run scores whole sets, once. It takes the exploratory and the confirmatory sets (EWT test and GENTLE; WSJ 01 and MCTB_en), each with every configuration the plan scores on it. Then it analyzes them and draws the figures. `--confirmatory` with `--max-sentences` is refused. It took 3 h 11 min on the machine of the original run.

### What a run writes

```
results/run_<YYYY-MM-DD>/
├── csv/
│   ├── counts_<task>_<dataset>.csv        # one row per configuration and sentence: the counts
│   ├── counts_<task>_<dataset>.prov.json  # what produced them
│   ├── thesis_*.csv                       # 13 tables of analyze.py
│   └── thesis.prov.json                   # their provenance: the plan analysed, the plan that measured, digests
├── analysis.txt                           # the report of analyze.py
├── figures/                               # fig1_h1, fig2_h2, fig3_h2c (PNG 300 dpi and PDF), figures.prov.json
├── log/                                   # one console transcript per command, and pipeline.txt
└── run.json                               # the invocations and the record of each stage
```

- **Counts** – POS `correct`, `total`; dependencies `las_correct`, `uas_correct`, `total`, `clas_correct`, `clas_total`, `clas_pred_total`; constituents `matched`, `gold`, `test`, `parsed`, `evalb_error`, `failure`, `d2_sample` – each row with `config`, `dataset`, `doc_id`, `sent_id`, `index` and `n_tokens`.
- **Provenance** – the dataset (status, SHA-256, sentences, tokens, unit and units), the plan's SHA-256, the git commit, branch and whether `prog/` had uncommitted changes, Python, the platform, the CPU and 20 library versions, the offline switches and whether downloads were allowed, and per configuration the wrapper, its arguments, the tool versions, the level, the training data, `model_version`, the scheme, the domain and the seeds; the members of the task the plan does not score on the set (`not_run_by_plan`); the configurations that failed; the command, its start and end, and whether the run was a smoke run or confirmatory. No path of the machine.
- **Thesis tables** – `thesis_scores`, `thesis_h1a`, `thesis_h1b`, `thesis_h2a`, `thesis_h2b`, `thesis_h2c`, `thesis_d2_coverage`, `thesis_d2_f1`, `thesis_d3`, `thesis_d4`, `thesis_descriptive`, `thesis_sensitivity` and `thesis_verdicts` (a row per hypothesis part and status, and for the whole H1 and H2 per status, each decided from the parts the plan defines for that status), each row with its status; every class and verdict follows from the CI and δ of its own row. `thesis_sensitivity` ends with the descriptive rows of the plan's `statistics.subset_sensitivity` (`<row>/evalb_both_parsers=<sentences>`, no δ, class or verdict). `thesis.prov.json` beside them records the plan they were analysed under, the plan and commit each counts file was measured under, and the SHA-256 of every counts file read and every table written.
- **Figures** – H1a and H1b; H2 against Δ_step,max; H2c. Slovak text, as the thesis is; exploratory and confirmatory intervals are never merged, and a row that is not in the run is not drawn – the panel says what it awaits.
- Only the time of the run as a whole is recorded (`run.json`, `log/pipeline.txt`); no stage measures speed.

## Results of version 1.0

| Run | Date | Directory |
|---|---|---|
| exploratory | 24 Oct 2026 (1 h 40 min) | `results/evidence/exploratory_v1_2026-10-24/` |
| confirmatory | 25 Oct 2026, 12:47–15:58, after the lock | `results/evidence/confirmatory_v1_2026-10-25/` |

**Verdicts** (`csv/thesis_verdicts.csv`, report `analysis.txt`):
- **H1 holds**: all its parts hold on the confirmatory sets.
- **H2 does not hold**: the dependency part of H2c holds, the POS part does not, and H2b holds only in part.

**What the frozen runs hold.**
- The exploratory run holds two analyses of the same counts:
  - `analysis_8db582d/`: the analysis of the run itself;
  - the top level: the analysis under the locked plan, which adds the descriptive sensitivity row `evalb_both_parsers`.
- In the confirmatory run, the counts of EWT test and WSJ 01 are byte for byte those of the exploratory run.
- The commits named in the provenance are those of the development history.
- The runs hold counts, tables, figures and logs. No sentence of a corpus is in them.

The counts are deterministic: on the same files and in the same environment they come out byte for byte the same. A repetition must give these files of `csv/` (rows with the header line), whose bytes `.gitattributes` keeps as the run wrote them:

| counts file | rows | SHA-256 |
|---|---|---|
| `counts_pos_ewt_test.csv` | 18 694 | `adcb054616abe38501de45aa55ec437ac9c1e4b3a9edf37dafef1777a923e35e` |
| `counts_dep_ewt_test.csv` | 14 540 | `a615036397d0c8ff5c392ef969ba282993eb47afba3037630b8f617d2e0deb61` |
| `counts_con_wsj_01.csv` | 8 473 | `c27bdbfd0636d1989987bcc04226ccaeed2c5eadfc8146472eb270676d5d2903` |
| `counts_pos_gentle.csv` | 9 339 | `f3b596f1db4c41288a290c0ee7f5e7024630a0d0936e32622259204ee8b1980f` |
| `counts_dep_gentle.csv` | 8 005 | `d5c7d1ba403f888efcae9af559fb5f01a0e423beb434b963a57952758180b5c5` |
| `counts_con_mctb_en.csv` | 20 501 | `76ed6dc3baaa6c8128f858e80b57213f181fb528b79deaa9a820dc78657f14c1` |

The analysis and the figures follow from the counts and the plan alone. On a copy of a run directory, this draws them again:

```bash
python scripts/run_pipeline.py --out-dir <copy of the run> --only analyze figures
```

What was found after the confirmatory run is in `plan/deviations.md`: the MCTB_en sentences EVALB does not score, with their causes. That finding changes no verdict.

## Using one configuration from Python

```python
from syntactic_analysis.plan import load
from syntactic_analysis.build import build

plan = load(strict=True)
analyzer = build(plan, "udpipe_ewt")   # constructed, loaded and checked against the plan
parse = analyzer.analyze_dependency([["The", "cat", "sat", "on", "the", "mat", "."]])

for tok in parse.sentences[0].tokens:
    print(tok.index, tok.text, tok.pos, tok.dep_rel, tok.head_index)
```

Every analyzer takes pre-tokenized input (a list of sentences, each a list of words), which is how the stages call it, one gold sentence at a time.

## Partial installation

```bash
pip install -e ".[spacy]"          # spaCy only
pip install -e ".[spacy,stanza]"   # spaCy + Stanza
pip install -e ".[all]"            # everything the project measures with, and matplotlib for the figures
```

The extras are `spacy`, `stanza`, `benepar`, `supar`, `udpipe`, `nltk-extra` (scikit-learn, for `nltk_arc`), `rdrpos` (empty: the wrapper reads the model files itself), `figures` (matplotlib) and `all`.

## Project structure

```
prog/
├── plan/
│   ├── v1.yaml                 # the experiment, stated once
│   ├── LOCK.json               # the lock of version 1.0 (25 Oct 2026)
│   └── deviations.md           # what was found after the lock
├── scripts/
│   ├── run_pipeline.py         # pos -> dep -> con -> analyze -> figures
│   ├── pos_tagging.py          # the measuring stages: per-sentence counts and their
│   ├── dependency_parsing.py   #   provenance (thin command lines over stage.py)
│   ├── constituency_parsing.py
│   ├── analyze.py              # estimates, classes and verdicts from the counts alone
│   ├── make_figures.py         # the three figures, from the thesis tables alone
│   ├── train_udpipe.py         # udpipe_ewt
│   ├── train_nltk.py           # nltk_ewt, nltk_arc
│   ├── train_spacy.py          # spacy_ewt
│   ├── train_rdrpos.py         # rdr_ewt (runs the GPL learner outside the repository)
│   ├── fetch_data.py           # downloads, hashes and counts GENTLE and MCTB_en
│   ├── smoke_plan.py           # every configuration of the plan on up to 20 sentences
│   └── lock_plan.py            # computes plan/LOCK.json, writes it only with --write
├── src/syntactic_analysis/
│   ├── plan.py                 # load(): plan/v1.yaml into frozen data, checked
│   ├── build.py                # build(plan, key): one configuration, constructed,
│   │                           #   loaded and checked against the plan
│   ├── seal.py                 # open_dataset(), verify_lock(), model_digests(),
│   │                           #   downloaded_digests(): the seal and the lock
│   ├── stage.py                # the measuring stages: counts, provenance, the D2 sample
│   ├── base.py                 # BaseAnalyzer, the contract every wrapper implements
│   ├── models.py               # Token, Sentence, DependencyParse, ConstituencyParse
│   ├── normalize.py            # the output contract: alignment, escaping, root wrappers
│   ├── data_loader.py          # CoNLL-U loader (doc_id from # newdoc id or sent_id)
│   ├── deprel_map.py           # Stanford Dependencies -> UD relation mapping
│   ├── tagsets.py              # the 17 UPOS tags and the PTB -> UPOS table
│   ├── spacy_doc.py            # spaCy Doc -> tokens and parses (spaCy, Benepar)
│   ├── ptb_conventions.py      # Penn Treebank sections, labels and traces
│   ├── checksum.py             # SHA-256 of a file, the verified download
│   ├── offline.py              # a run downloads nothing: its switches, NotCachedError
│   ├── downloads.py            # the downloaded files a load reads: roots, declarations
│   ├── project.py              # the project root and the default paths
│   ├── evaluation/
│   │   ├── metrics.py          # per-sentence POS and dependency counts, corpus_metric
│   │   ├── evalb.py            # EVALB parameter sets and per-sentence bracket counts
│   │   ├── estimate.py         # the paired cluster bootstrap, classify, decide
│   │   ├── gold.py             # the gold sets: EWT, the WSJ sections
│   │   ├── ptb.py              # Penn Treebank gold trees, corpus_digest
│   │   ├── mctb.py             # the MCTB_en treebank
│   │   ├── provenance.py       # versions, commit, checksums
│   │   └── report.py           # console helpers of the scripts
│   └── wrappers/               # benepar, nltk, nltk_transition, rdrpos, spacy,
│                               #   stanza, supar, udpipe: one module per tool
├── data/                       # DATASET.md, the digests (*.sha256), the EWT relation
│                               #   inventory; the corpora themselves are not tracked
├── results/evidence/           # the frozen runs of version 1.0
├── results/run_*/, results/smoke/  # new runs (ignored by git)
├── model_cache/                # locally trained models (ignored)
├── setup.sh, pyproject.toml    # setup; the package, its dependencies (ranges) and extras
├── requirements-lock.txt       # the exact versions of version 1.0 (LOCKED=1 bash setup.sh)
└── LICENSE, NOTICE             # MIT for this code; the licences of the tools,
                                #   models and corpora
```

Comments in the code refer in places to `CHANGELOG_v1.md`, `dokumentacia.txt`, the tag `exploracia-2026-10-17` and the project's tests. These belong to the development record of the work and are not part of this repository. The code is kept as it was locked.

## The environment behind the numbers

`requirements-lock.txt` pins the exact version of every package of the environment version 1.0 was run in (Python 3.13.0, Windows, CPU only): among them stanza 1.11.1, spacy 3.8.11, nltk 3.9.3, supar 1.1.4, benepar 0.2.0, ufal.udpipe 1.4.0.1 and torch 2.10.0, the versions the provenance of its runs records.

Its header names what a run depends on that pip does not install:
- Stanza's resources index 1.11.0, whose models the Stanza wrapper names and checks;
- the revision of `google/electra-large-discriminator` the Stanza wrapper records.

**It is the way to reproduce version 1.0 exactly:** `LOCKED=1 bash setup.sh` on Python 3.13. By hand, that is `python -m pip install -r requirements-lock.txt`, then `python -m pip install -e . --no-deps`.

`pyproject.toml` keeps its version ranges, deliberately: `pip install -e ".[all]"`, which `setup.sh` runs by default, resolves them to the newest versions they allow, which need not be those of version 1.0. One range is narrower than the rest: nltk stays below 3.10, because nltk 3.10 reads one tree of MCTB_en differently and refuses it (`pyproject.toml`).

For a number, the authoritative record is the provenance beside the file it came from: `counts_<task>_<dataset>.prov.json`. Its `environment` block names Python, the platform, the CPU and the library versions of *that* run, together with the git commit and the SHA-256 of the gold set.

## Requirements

- Python 3.10–3.13 (3.14 is not supported: spaCy still needs pydantic v1). Every recorded run used 3.13.0. `requirements-lock.txt` was taken on 3.13 and is installed on it (several of its packages need 3.11 or newer, and `LOCKED=1 bash setup.sh` refuses another minor version); `pyproject.toml` states the ranges.
- The runs are on the CPU: the environment they used has the CPU build of torch, and the Stanza wrapper loads its pipelines with `use_gpu=False` unless told otherwise.
- `bash` and `curl` for `setup.sh` (on Windows, Git Bash has both), and `git`, which the provenance and the lock ask.
