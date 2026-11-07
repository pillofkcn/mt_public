"""Analyzer wrappers — one module per NLP tool.

Each wrapper implements the BaseAnalyzer interface for a specific NLP tool
(spaCy, Stanza, NLTK, etc.). plan/v1.yaml names the wrapper of every
configuration (`wrapper: <module>.<Class>`), and syntactic_analysis.build
constructs, loads and checks it.

The wrappers, grouped by the method axis the thesis compares on. Each entry
ends with the configurations of plan/v1.yaml that use it.

  RULE-BASED
    rdrpos       — RDRPOSTagger: a Ripple-Down Rules cascade whose rules are
                   INDUCED from a corpus, not hand-written — the same family as
                   Brill's transformation-based tagger. Configuration: rdr_ewt.

  STATISTICAL
    nltk            — averaged perceptron POS tagger + PCFG constituency parser.
                      POS: nltk_pre (the shipped tagger, PTB->UPOS table plus
                      context rules) and nltk_ewt (the same perceptron trained
                      here on the EWT train split). Constituency: pcfg_words and
                      pcfg_tags (the PCFG over words, and over the tagger's tags).
    nltk_transition — arc-eager transition-based dependency parser with a linear
                      classifier, over the UPOS of udpipe_ewt. Configuration:
                      nltk_arc.

  NEURAL
    spacy        — CNN dependency parser + POS tagger. Configurations: spacy_sm
                   (en_core_web_sm) and spacy_ewt (trained here on EWT).
    stanza       — BiLSTM full-stack parser (Stanford NLP), with or without a
                   frozen transformer. Configurations: stanza_ewt, stanza_comb,
                   stanza_trf (POS and dependencies), stanza_con and
                   stanza_con_trf (constituency).
    benepar      — chart-based neural constituency parser on top of spaCy.
                   Configuration: benepar.
    supar        — CRF constituency parser. Configuration: supar_con.

  TWO LEVELS AT ONCE
    udpipe       — MorphoDiTa averaged-perceptron POS tagger (L1 statistical) and
                   Parsito, a transition-based parser whose classifier is a
                   feed-forward neural network (L2 neural, shallow). It has its own
                   heading because the method axis is ordinal and PER COMPONENT, so
                   a single coarse heading has to be wrong for one of the two:
                   filing it under STATISTICAL is what made the dependency stage
                   look as though its L1 row were UDPipe rather than
                   nltk_transition. The per-task truth
                   is `method_level` in wrappers/udpipe.py; `method_type` is NEURAL
                   after the parser. See base.py's module docstring.
                   Configuration: udpipe_ewt (refitted on the current EWT train
                   split by scripts/train_udpipe.py).

Note that the RULE-BASED row is populated for POS tagging only. No rule-based
English *parser* installs on this stack and produces UD-comparable output; that is
a documented finding, not an omission.

Until 23 Oct 2026 three more wrappers were here: regexp (a hand-written UPOS
cascade), tnt (a trigram HMM tagger) and linkgrammar (declared, never usable).
Version 1.0 runs none of them, and they were removed with the configurations
the plan does not run (CHANGELOG_v1.md, step 12).
"""
