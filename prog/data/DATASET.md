# Datasets — the four test sets of version 1.0, and the training data trained on here

`plan/v1.yaml` names four test sets, two exploratory and two confirmatory, and the training
data of every configuration. This file records the identity of each test set and of the
training data this project trains on – source, version, SHA-256, counts, licence – and the
properties a result on it depends on, so that any number in `results/` can be traced to the
exact data that produced it.

The corpus files are **not tracked**: the `.conllu` files (≈20 MB: the EWT ≈18 MB, GENTLE
≈2 MB) and `data/mctb_en/` are ignored, and only their digests (`*.sha256`) are. The Penn
Treebank sample is not in `data/` at all: it ships with NLTK and lives in `nltk_data`.

## The four test sets

| plan id | corpus | task | status | resampling unit | sentences | words | licence |
|---|---|---|---|---|---|---|---|
| `ewt_test` | UD_English-EWT test, release r2.17 | POS, dependencies | exploratory | document: 316 | 2 077 | 25 094 | CC BY-SA 4.0 |
| `wsj_01` | Penn Treebank WSJ section 01, NLTK sample | constituents | exploratory | file: 100 | 1 993 | 47 633 | LDC sample, non-commercial use only |
| `gentle` | UD_English-GENTLE, UD 2.18 | POS, dependencies | confirmatory | document: 26 | 1 334 | 17 799 (17 619 tokens) | CC BY-NC-SA 4.0 |
| `mctb_en` | MCTB_en (Yang et al. 2022), commit `1ad06f4` | constituents | confirmatory | sentence: 5 000 | 5 000 | 97 641 | MIT |

"Words" are what the tools are given and scored on: for the UD sets the syntactic words (word
lines with an integer id, which is what `data_loader.parse_conllu` reads; a multiword token
is one token over several syntactic words), for the treebanks the leaves of each tree after
trace removal. The input is always the gold tokenization and the gold sentence boundaries.

**Status.** The EWT test set and WSJ section 01 are exploratory because they were analysed
in the earlier phase of the work, and the hypotheses of version 1.0 were formulated after
those results. GENTLE and MCTB_en are confirmatory: their first and only evaluation took
place after the plan was locked, and the confirmatory verdicts come from them alone.

**Resampling.** The bootstrap of `scripts/analyze.py` draws the units of a set, as the plan's
`resampling` names them: a document of the EWT test set is its `# newdoc id`, or, before
the first one, the `sent_id` without its final "-number" (both give the same 316 documents);
a file of WSJ 01 is one of `wsj_0100` … `wsj_0199`; a GENTLE document is its `# newdoc id`;
MCTB_en is resampled by sentence. A full run whose number of units differs from the plan's
`expected_units` is refused. The sensitivity analysis also takes the sentence as the unit
(`statistics.unit_sensitivity`) on every set whose unit is not the sentence already.

## UD_English-EWT — the exploratory test set, and the training data

| split | sentences | words | bytes | SHA-256 |
|---|---|---|---|---|
| train | 12 544 | 204 577 | 15 029 817 | `6fe5692f2e021982…` |
| dev | 2 001 | 25 147 | 1 898 013 | `de507882611c54f1…` |
| test | **2 077** | **25 094** | 1 899 759 | `b837531faddd54f5c89cf5259cfe5306b075556a9ff19b2271393df29c6b02f8` |

Full digests are in `en_ewt-ud-<split>.conllu.sha256` (tracked); every counts file of a run
records the test set's digest in its provenance (`dataset.sha256`). Verify with:

```bash
(cd data && sha256sum -c en_ewt-ud-test.conllu.sha256)
```

The test split is the exploratory test set of POS and dependencies (`ewt_test`). The train
split is the training data of the five configurations this project trains on it
(`ewt_train_snapshot` in the plan; see "Training data" below), and the dev split is used only for early stopping,
model selection and checks while training. The test split is never read by a training script.

### Provenance and version

**The EWT of this project is release r2.17 (UD 2.17).** `setup.sh` downloads the three files
from the repository at the tag `r2.17` (`EWT_TAG`) with `curl -fSL`, checks each against its
recorded digest, and asserts that the test split loads exactly 2 077 sentences and 25 094
words — the figures in the table above.

**Where the files come from.**
- They were first downloaded from the repository's `master` branch on 23 Mar 2026, before
  the download was pinned to a tag. The plan's id of the train split, `ewt_train_snapshot`,
  comes from that.
- On 7 Nov 2026 their SHA-256 were compared with the files of the release tags. All three
  splits are byte for byte those of `r2.17`, and differ from those of `r2.15`, `r2.16` and
  `r2.18`.
- So every number of version 1.0 is a number on UD_English-EWT r2.17, and every counts
  file records this test-set digest in its provenance.

Another `EWT_TAG` downloads other files, and the checksum check of `setup.sh` then fails by
design: a changed corpus changes every score.

### One configuration is trained on another release

Every in-domain configuration of the plan is trained on the r2.17 train split except
`stanza_ewt`: Stanza trained its `ewt` package on UD_English-EWT **r2.15** train
(`ewt_train_r2_15`; Stanza 1.10.0 rebuilt all its models with UD 2.15). Both releases belong to
the same corpus, so the plan accepts `stanza_ewt` in the EWT chains of H1a, and records it there
as an approximation (`approximations: {stanza_ewt: ewt_train_r2_15}`) rather than as the same
training data.

### Licensing

UD_English-EWT is **CC BY-SA 4.0**. It is a hand-corrected dependency conversion of the
**LDC English Web Treebank (LDC2012T13)**, whose original PTB-style *constituency* trees are
**not** freely licensed — which is why the EWT sentences have no gold constituency layer here,
and the constituency task is scored on WSJ section 01 and MCTB_en instead.

Citation: Silveira et al. (2014), *A Gold Standard Dependency Corpus for English*, LREC 2014.

## Penn Treebank sample — WSJ section 01 (exploratory test set) and section 00 (PCFG training data)

Not in `data/`: it ships with NLTK (`nltk.corpus.treebank`, the `treebank` package
`setup.sh` downloads) and lives in `nltk_data`. It holds sections 00 and 01 only, 199 files
and 3 914 sentences.

| section | files | sentences | tokens | used as |
|---|---|---|---|---|
| 00 | `wsj_0001` … `wsj_0099` | 1 921 | — | training data of the PCFG (`pcfg_words`, `pcfg_tags`; `wsj_00_nltk_sample` in the plan) |
| 01 | `wsj_0100` … `wsj_0199` (100) | **1 993** | **47 633** | **the exploratory constituency test set** (`wsj_01`) |

**SHA-256 of WSJ 01:** `1a74ae5683cdc12bfee2bc3fe8db014dda535bd3ce8744b5eea5a901de494416`.
The corpus is not a file of this repository, so this is not the digest of a file and
`sha256sum` cannot check it: it is `evaluation/ptb.corpus_digest` over the 100 files of
section 01, the SHA-256 of their raw text as `nltk.corpus.treebank.raw()` returns it, in
order, encoded as UTF-8. Every constituency counts file of a run records it in its provenance
(`dataset.sha256`; for example
`results/evidence/confirmatory_v1_2026-10-25/csv/counts_con_wsj_01.prov.json`). Section 00 has no
digest of its own; the PCFG configurations record the digest of the grammar induced from it
in their `model_version`.

Section 01 works as a gold standard because it is held out from every parser in the plan:
Stanza, Benepar and SuPar are trained on the standard split, sections **02–21**, and the PCFG is
induced from section 00 alone. It is not known to be held out from every tagger: `pcfg_tags`
parses over the tags of NLTK's shipped perceptron (`nltk_pre`), whose training corpus NLTK's
documentation does not state, so whether that tagger has seen the test sentences is not known.

Two properties of this gold must travel with any number computed from it:

- it is the **original** PTB annotation, which has no `NML` nodes, while Benepar and Stanza are
  trained on the **revised** annotation (which introduced `NML` and changed tokenisation). The
  plan therefore scores constituents with `NML` collapsed on both sides
  (`tasks.con.metric`: `evalb: collins`, `nml: collapse_both_sides`; the parameter set
  `COLLINS_NML` of `evaluation/evalb.py`), so that a parser is not charged for brackets the gold
  annotation could not contain. Version 1.0 scores constituents with this one parameter set;
- it is **newswire**, not web text.

Licence: the sample's own README in `nltk_data` says it is "(C) LDC 1995", made available
"under fair use for the purposes of illustrating NLTK tools", and "for non-commercial use
only"; the plan records the same.

## UD_English-GENTLE, UD 2.18 — confirmatory test set (POS, dependencies)

| | |
|---|---|
| source | <https://github.com/UniversalDependencies/UD_English-GENTLE>, tag `r2.18` (the UD 2.18 release) |
| file | `en_gentle-ud-test.conllu` → `data/en_gentle-ud-test.conllu` (the treebank has a test split only) |
| bytes | 1 949 238 |
| SHA-256 | `42d12cdef99bcd3160a3241ba1811fe9e7fd4b37a77f9b9e8c2852f04d69f3fa` (`en_gentle-ud-test.conllu.sha256`) |
| documents | **26** (`# newdoc id`), the plan's `expected_units`; the resampling unit is the document |
| sentences | 1 334 |
| tokens | 17 619 (surface tokens: multiword-token ranges, plus the word lines no range covers) |
| syntactic words | 17 799 (word lines with an integer id; empty nodes are not words) — what the tools tag and parse |
| format | no malformed line; `data_loader.parse_conllu` reads the same 1 334 sentences, 17 799 syntactic words and 26 documents |
| status | confirmatory: sealed until the lock (see "The seal" below) |
| licence | **CC BY-NC-SA 4.0** (`LICENSE.txt` at the tag) — non-commercial; not redistributed here |

The two counts are UD's own: the treebank page
(<https://universaldependencies.org/treebanks/en_gentle/index.html>, read 23 Oct 2026) says
"This corpus contains 1334 sentences, 17619 tokens and 17799 syntactic words." A multiword
token (a range line such as `3-4`) is one token over several syntactic words.
`scripts/fetch_data.py` counts and prints both.

GENTLE (Genre Tests for Linguistic Evaluation) is an out-of-domain test set annotated like
GUM. Its README at `r2.18` lists eight genres: dictionary entries, live esports commentary,
legal documents, medical notes, poetry, mathematical proofs, course syllabuses and threat
letters. That it follows the conventions of GUM is what H2c's superiority test on GENTLE
rests on: Stanza's `combined` package is trained on GUM among others, its `ewt` package is not.
Neither package's training recipe reads GENTLE or MCTB_en (checked in the Stanza sources).

The resampling unit is the document, and there are only 26 of them; the plan's sensitivity
analysis resamples GENTLE by sentence as well.

**UPOS in GENTLE is annotated manually in non-UD style and automatically converted to UD.**
The treebank page (<https://universaldependencies.org/treebanks/en_gentle/index.html>, read
23 Oct 2026) says in its summary of the annotation:

```
UPOS: annotated manually in non-UD style, automatically converted to UD
XPOS: annotated manually
```

The machine-readable metadata at the end of the README at `r2.18` says the same in its own
terms:

```
UPOS: converted from manual
XPOS: manual native
```

So the part-of-speech annotation was done by hand in a scheme other than UD, and UPOS was
derived from it automatically; XPOS is the only part-of-speech layer listed as annotated by
hand. Neither source names the layer the conversion starts from or the XPOS tagset. A UPOS
accuracy on GENTLE is therefore agreement with a converted tag.

Citation: Aoyama, Behzad, Gessler, Levine, Lin, Liu, Peng, Zhu and Zeldes (2023), *GENTLE:
A Genre-Diverse Multilayer Challenge Set for English NLP and Linguistic Evaluation*,
Proceedings of the 17th Linguistic Annotation Workshop (LAW-XVII), pp. 166–178.

## MCTB_en — confirmatory test set (constituents)

| | |
|---|---|
| source | <https://github.com/RingoS/multi-domain-parsing-analysis>, commit `1ad06f47e3b08a4d1338473e67f9911cf795eba2` (2022-09-22) |
| files | `data/MCTB_en/<domain>.cleaned.txt` → `data/mctb_en/<domain>.cleaned.txt`, one bracketed tree per line |
| domains | 5: dialogue, forum, law, literature, review |
| trees | **5 000**, 1 000 per domain (the README says 1,000 instances per domain); the resampling unit is the sentence |
| tokens | 97 641 (the words of each tree after trace removal: what a parser is given) |
| format | every line parses as one tree under a TOP-like root; no blank line |
| token convention | brackets PTB-escaped (`-LRB-`, `-RRB-`), double quotes raw (`"`); `tokens: {brackets: ptb, quotes: raw}` in the plan |
| status | confirmatory: sealed until the lock (see "The seal" below) |
| licence | **MIT** (`LICENSE` at the commit, "Copyright (c) 2022 ys"); not redistributed here |

| domain | trees | tokens | bytes | SHA-256 |
|---|---|---|---|---|
| dialogue | 1 000 | 13 509 | 207 727 | `dd02b20b22e06b2aa98100f366fc81cd0b9f211d0768dcece806cdfd1841c132` |
| forum | 1 000 | 22 012 | 333 125 | `4b09da0e52aba452c8e1b47386873889a399f0fa3f15113fb07a09f24182d894` |
| law | 1 000 | 25 585 | 384 705 | `5eb2efe911a02555a3b6b0c04728cbed0e9c956f3f2ad7ab7d61192592c47dab` |
| literature | 1 000 | 23 238 | 333 674 | `3dd482e7bf12025b0eb44d225c3d21e4399b793dbdebed0b2431135214ead2da` |
| review | 1 000 | 13 297 | 200 191 | `db7ff6107bdf2cd47d32e4736bb62493615171a7031ad035826e8a62ff7f1ef1` |

The digests are recorded in `mctb_en.sha256`. The five token counts equal the
`# Total tokens` column of Table 1 of the paper for the five MCTB domains. The digest of the
whole gold set, which a run's provenance records as `dataset.sha256`, is the SHA-256 of the
bytes of the five files one after another, as `corpus_digest` is for WSJ.

Three properties of this gold must travel with any number computed from it:

- **It has no `NML` nodes**: 0 in all 5 000 trees (counted). It follows the original PTB
  guideline, like the WSJ sample, while Benepar and Stanza are trained on the revised
  annotation, which introduced `NML`. The plan scores constituents with `NML` collapsed on
  both sides (`tasks.con.metric.nml: collapse_both_sides`).
- **Its trees began as a parser's output.** The repository README does not describe the
  annotation; the paper does (Yang et al. 2022, §4.1, p. 115): the annotators were asked to
  "read the guideline, practice and correct the errors of the predicted parse tree, which is
  produced by a SOTA chart-based parser that is developed based on Berkeley Neural Parser".
  Benepar (`benepar_en3`) is the Berkeley Neural Parser, so the gold was corrected from the
  output of a parser of the family one of the scored parsers belongs to, which may favour
  Benepar on MCTB_en.
- **Its double quotes are raw, its brackets PTB-escaped.** The one format check of MCTB_en
  before the lock (24 Oct 2026; leaf counts over all five files, nothing printed, nothing
  scored) found the leaves `-LRB-` 341, `-RRB-` 276, ` `` ` 0,
  `''` 0, raw `(` 0, raw `)` 0 and raw `"` 1 016, and no line that does not parse. WSJ 01
  writes brackets the same way but a double quote as ` `` ` where it opens and `''`
  where it closes. The plan declares this per constituency dataset
  (`datasets.<id>.tokens`: `wsj_01` ptb/ptb, `mctb_en` ptb/raw), and every constituency
  configuration is given its sentence in the convention of WSJ 01: a raw `"` becomes
  ` `` ` or `''` by alternation within the sentence, the first opening, and the gold
  words are put back into the parser's tree before it is scored
  (`ptb_conventions.parser_tokens`, `restore_words`). Single and
  typographic quotes and the other PTB escapes (`\/`, `\*`, `-LCB-`, `-LSB-`) were not
  counted and are given to the parsers as MCTB_en writes them.

The texts come from five sources the paper names (§4.1): Wizard of Wikipedia (dialogue),
Reddit as released by Völske et al. 2017 (forum), the European Court of Human Rights
database (law), Project Gutenberg (literature) and the SNAP Amazon review data (review). The
MIT licence is the repository's.

Citation: Yang, Cui, Ning, Wu and Zhang (2022), *Challenges to Open-Domain Constituency
Parsing*, Findings of the Association for Computational Linguistics: ACL 2022, pp. 112–127,
doi:10.18653/v1/2022.findings-acl.11.

## The seal on GENTLE and MCTB_en

`plan/v1.yaml` names two test sets on which nothing had been measured when the hypotheses
were written. They are **confirmatory and sealed until the plan is locked**. Until then they
may be downloaded, hashed and counted, and nothing else: no sentence, word, tag or tree of
them is printed, no statistic beyond the counts above is taken, and no analyzer is run on
them. A stage can open them only through `syntactic_analysis.seal.open_dataset`, which
refuses them unless the stage is started with `--confirmatory` and `plan/LOCK.json`,
committed, matches the plan, the code (unchanged since the locked commit), these digests, the
files of the five models trained in the project and the downloaded model files. No other
dataset may have a path that reaches their local copies. The plan of version 1.0 was locked
on 25 Oct 2026; `README.md` says how to lock a checkout of one's own.

Where each comes from is stated once, in the `source` of its dataset in `plan/v1.yaml`
(repository, a tag or a commit, the files). `scripts/fetch_data.py` downloads them from
`raw.githubusercontent.com` at that ref, records their SHA-256 beside them and checks their
format by counting; it prints counts and digests only. Neither corpus is committed: the
files are ignored (`data/*.conllu`, `data/mctb_en/`) and only the digests are tracked.

```bash
python scripts/fetch_data.py                 # download, record the digests, count
python scripts/fetch_data.py --verify-only   # no network: digests and counts of the local copy
(cd data && sha256sum -c en_gentle-ud-test.conllu.sha256 mctb_en.sha256)
```

The counts and digests above were taken on 23 Oct 2026 by the first download and confirmed
by `--verify-only`.

## Training data

Every configuration names its training data by an id of the plan's `training_data`. Two of
them are data this project trains on; the others are the training data of published models,
as their providers state them.

| plan id | corpus | configurations | where |
|---|---|---|---|
| `ewt_train_snapshot` | UD_English-EWT r2.17 train (the files of 23 Mar 2026): 12 544 sentences, 204 577 words | `rdr_ewt`, `nltk_ewt`, `udpipe_ewt`, `spacy_ewt`, `nltk_arc` | `data/en_ewt-ud-train.conllu`, trained here |
| `wsj_00_nltk_sample` | WSJ section 00 of the NLTK Penn Treebank sample: 1 921 sentences | `pcfg_words`, `pcfg_tags` | `nltk_data`; the wrapper induces the grammar when it loads |
| `ewt_train_r2_15` | UD_English-EWT r2.15 train | `stanza_ewt` | Stanza's `ewt` package |
| `ud_combined_r2_15` | UD 2.15: EWT, GUM and GUMReddit train, PUD and Pronouns test, Stanza's hand-parsed English sentences | `stanza_comb`, `stanza_trf` | Stanza's `combined` package |
| `nltk_tagger_unstated` | not stated in NLTK's documentation (PTB tags) | `nltk_pre` (and the tags of `pcfg_tags`) | NLTK's `averaged_perceptron_tagger_eng` |
| `ontonotes5` | OntoNotes 5, dependencies in the ClearNLP conversion | `spacy_sm` | spaCy's `en_core_web_sm` |
| `ptb_revised_wsj_02_21` | Penn Treebank revised (LDC2015T13), WSJ sections 02–21 | `stanza_con`, `stanza_con_trf`, `benepar` | Stanza's constituency models, `benepar_en3` |
| `ptb3_wsj_02_21` | Penn Treebank 3, WSJ sections 02–21 | `supar_con` | SuPar's `crf-con-en` |

The models trained here are written to the ignored `model_cache/`: `udpipe_ewt` by
`scripts/train_udpipe.py`, `spacy_ewt` by `scripts/train_spacy.py`, `rdr_ewt` by
`scripts/train_rdrpos.py` (each with a training record), and `nltk_ewt` and `nltk_arc` by
`scripts/train_nltk.py`. They are trained before a run (`setup.sh` runs the four scripts)
and never during one: a run that finds one missing fails, and a lock records the SHA-256 of
their files. The dev split serves `udpipe_ewt` for early stopping, `spacy_ewt` for choosing
the best model, and the check of `rdr_ewt`; the test split is not read.

## Related tracked files

- `en_ewt-ud-{train,dev,test}.conllu.sha256` — digests
- `en_gentle-ud-test.conllu.sha256`, `mctb_en.sha256` — digests of the confirmatory test
  sets, written by `scripts/fetch_data.py`
- `ewt_deprel_inventory.txt` — the 49 DEPREL labels of the test split with counts.
  `deprel_map.load_gold_inventory()` reads it, so `deprel_map.validate_against_gold()` can
  check that every target of the SD → UD mapping `spacy_sm` is scored with occurs in the gold,
  without the 18 MB corpus. The project's tests ran that check; no stage calls it.
