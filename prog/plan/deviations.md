# Deviations from the plan of version 1.0 after the lock

The plan `plan/v1.yaml` is locked by `plan/LOCK.json`. After the lock the plan does not
change. Every deviation from it, or from the procedure it describes, is recorded here: what
deviated, when, why and what it means for the results. Changes before the lock were changes
of the plan and are not recorded here. The commits named below belong to the development
history of the work, which is not part of this repository.

## Check recorded before the confirmatory run (25 Oct 2026)

The confirmatory run (`scripts/run_pipeline.py --confirmatory`) scores the exploratory sets
EWT test and WSJ 01 again, from the same locked code as GENTLE and MCTB_en. Their
per-sentence counts must equal those of the full exploratory run of 24 Oct 2026
(`results/run_2026-10-24/`, commit `8db582d`, plan `872bad0c…`; its results were confirmed
by the author):

| counts file of the exploratory run | rows | SHA-256 |
|---|---|---|
| `csv/counts_pos_ewt_test.csv` | 18 694 | `adcb054616abe38501de45aa55ec437ac9c1e4b3a9edf37dafef1777a923e35e` |
| `csv/counts_dep_ewt_test.csv` | 14 540 | `a615036397d0c8ff5c392ef969ba282993eb47afba3037630b8f617d2e0deb61` |
| `csv/counts_con_wsj_01.csv` | 8 473 | `c27bdbfd0636d1989987bcc04226ccaeed2c5eadfc8146472eb270676d5d2903` |

(Rows including the header. The directory `results/run_2026-10-24/` is ignored by git, which
is why its digests are recorded here; the run is frozen in
`results/evidence/exploratory_v1_2026-10-24/`.)

The comparison is row by row: for every configuration and sentence (`config`, `dataset`,
`doc_id`, `sent_id`, `index`, `n_tokens`) all count columns, and for constituents also
`parsed`, `evalb_error`, `failure` and `d2_sample`. Every difference is an error that must
be explained in this file before the confirmatory results are used. The provenance may
differ only in the fields of the run (commit, plan, time, command) and in texts that changed
before the lock.

Between the exploratory run and the lock the plan changed (`872bad0c…` → `34f81ee5…`, the
descriptive sensitivity `evalb_both_parsers`) and so did the code (provenance, lock,
analysis), not the measurement; `analyze.py` proves this for the counts of the exploratory
run by comparing the parts of the plan that determine the counts.

## Deviations

None so far.

## Finding after the confirmatory run (25 Oct 2026)

This is not a change of the plan or of the code, and it changes no verdict. The verdicts of
the parts are preregistered and remain the main results. The record explains the MCTB_en
sentences that EVALB does not score.

Confirmatory run: `results/run_2026-10-25/` (frozen in
`results/evidence/confirmatory_v1_2026-10-25/`), commit `3b01035` (locked code `538c5ba`),
clean tree, 12:47–15:58. The check recorded above holds: the counts of EWT test and WSJ 01
equal, sentence by sentence, those of the run of 24 Oct 2026, and the files
`counts_pos_ewt_test.csv`, `counts_dep_ewt_test.csv` and `counts_con_wsj_01.csv` are
byte-identical (0 differences in each of the 22 configurations).

### Sentences EVALB does not score

On MCTB_en (5 000 sentences, 1 000 in each domain):

| configuration | total | dialogue | forum | law | literature | review | WSJ 01 |
|---|---|---|---|---|---|---|---|
| `stanza_con` | 773 | 58 | 167 | 256 | 230 | 62 | 31 |
| `stanza_con_trf` | 773 | 58 | 167 | 256 | 230 | 62 | 31 |
| `benepar` | 592 | 44 | 117 | 221 | 156 | 54 | 24 |
| `supar_con` | 48 | 1 | 8 | 20 | 19 | 0 | 0 |
| `pcfg_tags` (of the 177 processed sentences of the sample) | 6 | 0 | 1 | 2 | 3 | 0 | 0 |
| `pcfg_words` | 0 | | | | | | 0 |

EVALB leaves `stanza_con` and `stanza_con_trf` unscored on the same 773 sentences.

### Cause

EVALB deletes punctuation by tag (COLLINS.prm: `,` `:` ` `` ` `''` `.`, plus `?` `!`) and
skips a sentence whose words, after the deletion, differ from the gold ones. The diagnosis
covers all five domains of MCTB_en, 40 sentences in all that EVALB does not score for
`stanza_con`: the first 20 by index (all from the dialogue domain) and the first 5 by index
in each of the domains forum, law, literature and review. On 25 Oct 2026 they were run again
with the locked code, with `stanza_con` and `benepar`, exactly as in the stage. The counts of
all 80 analyses equal those of the confirmatory run.

For `stanza_con` the error in all 40 sentences is "Length unmatch", except one "Words
unmatch" (dialogue). `benepar` scores 13 of these sentences and leaves the other 27
unscored, for the same causes.

| domain | sentences | main tags (gold → parser), `stanza_con` | `benepar` |
|---|---|---|---|
| dialogue | 20 | `:` → `HYPH` 12 and `NFP` 1, `-LRB-`/`-RRB-` → `,` or `.` 7, `CC` → `,` (slash) 3, unusual gold tags 7 | `:` → `HYPH` 10, brackets 5, unusual gold tags 4, `:` → `CC` (slash) 1 |
| forum | 5 | `:` → `HYPH` 4, `-LRB-` → `,` 3, `CC` → `,` (slash) 2, a full stop with the gold tag `CC` 1 | `:` → `HYPH` 3, brackets 2, a full stop with the gold tag `CC` 1 |
| law | 5 | `:` → `HYPH` 3, a full stop with the gold tag `NNP` 2 | `:` → `HYPH` 3 |
| literature | 5 | apostrophe `'` with the gold tag of a word (`NNP`, `NN`, `DT`, `FW`) → quote ` `` ` or `''` 4, `:` → `HYPH` 1, a full stop with the gold tag `NNP` 1 | apostrophe → `''` or `,` 3, `:` → `HYPH` 1, a full stop with the gold tag `NNP` 1 |
| review | 5 | `:` → `HYPH` 5, `-LRB-`/`-RRB-` → `,` 4 | `:` → `HYPH` 5, brackets 3 |

- **The hyphen `-`** (all domains, the most frequent cause): the gold standard tags it `:`
  (deleted), the parsers `HYPH` (not deleted), once `NFP`. `HYPH` and `NFP` are tags of the
  revised Penn Treebank and of UD EWT, on which the parsers and Stanza's tagger were trained;
  the gold standard of MCTB_en was made by correcting a parser's output in the original PTB
  tag set, which uses `:` for the hyphen.
- **Brackets** (dialogue, forum, review): the gold standard tags them `-LRB-`/`-RRB-` (not
  deleted); Stanza's tagger gets the token `-LRB-` in the PTB convention and tags it `,` or
  `.` (deleted). Benepar sometimes changes the tag of a bracket to punctuation and sometimes,
  the other way round, gives a bracket tag to another character.
- **The slash `/`** (dialogue, forum): gold tag `CC`, Stanza `,`.
- **Another cause - the apostrophe `'`** (literature): the gold standard tags it as a word or
  part of one (`NNP`, `NN`, `DT`, `FW`), the parsers as a quote (` `` `, `''`), which EVALB
  deletes.
- **Unusual gold tags** (dialogue, forum, law, literature): a few words have the tag `.` in
  the gold standard, and a few full stops the tag `NNP`, `NN`, `DT`, `CC` or `FW`; a parser
  with its own tag always differs from the gold standard there.
- Other characters, such as `§` or an ellipsis, did not appear as a cause in these 40
  sentences.
- This concerns only the parsers that tag on their own (Stanza, Benepar, and for the PCFG
  over tags NLTK's tagger). `supar_con` writes no tags and gets the gold tags of its words
  (`evalb._with_gold_tags`), so punctuation is deleted for it as in the gold tree. Its 48
  sentences were not examined here.
- On WSJ 01 such sentences make up 1.2–1.6 %.

### Difficulty check of the skipped sentences

SuPar (with no skips from this cause) has an F1 of 75.66 on the sentences `stanza_con` does
not score (745 sentences; 28 of the 773 SuPar does not score either) and 83.71 on the others
(4 207 sentences). The mean length of the skipped sentences is 31.2 tokens, of the others
17.4. The skipped sentences are thus harder than average, so the F1 of Stanza and Benepar
on MCTB_en is rather overestimated, and the drop WSJ 01 → MCTB_en in H2b rather
underestimated. All three parts of H2b for constituents hold (statistic `stanza_con` 7.86
[6.68; 9.13], `stanza_con_trf` 6.61 [5.86; 7.40], `benepar` 5.73 [4.89; 6.62]), so these
verdicts are conservative.

### Preregistered sensitivity on the sentences both parsers have scored

(`statistics.subset_sensitivity: evalb_both_parsers`, `thesis_sensitivity.csv`)

- `stanza_con` → `benepar` on MCTB_en: 3.93 [3.58; 4.27] on 4 122 sentences; the main
  result is 4.09 [3.70; 4.48].
- `stanza_con` → `stanza_con_trf`: EVALB leaves both configurations unscored on the same
  sentences, so the result on the 4 227 common sentences equals the main one, 3.89
  [3.52; 4.28].

### Check of the lock

The result of `seal.verify_lock` is written neither into `run.json` nor into the logs. It
passed indirectly: the seal opened GENTLE and MCTB_en, which it does not do without a
successful check of the lock. On 25 Oct 2026, after the run, it passed again over the same
tree (commit `3b01035`, `prog/` unchanged).

The numbers in this record were checked by a computation of their own from
`results/run_2026-10-25/csv/counts_con_mctb_en.csv`.
