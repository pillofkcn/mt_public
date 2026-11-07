"""The two POS tagsets this project converts between.

The gold standard and every UD-native tagger use the 17 Universal POS tags;
NLTK's pretrained tagger and the Penn Treebank sample use the PTB tagset.
`wrappers/nltk.py` needs both facts: it converts its tagger's output and
guards its EWT-trained model. Until 12 Oct 2026 the table was a private name of
that wrapper, imported by the TnT wrapper (removed with version 1.0), and the
tag set was declared in both.

At the top level of the package, beside `ptb_conventions.py`, for the same
reason that module is: a wrapper imports a shared table from here and never
from another wrapper, so removing one wrapper module takes no table away from
another (dokumentacia.txt §9.2). No wrapper imports another: until version 1.0
`wrappers/nltk_transition.py` built a UDPipe tagger as its default POS source;
its POS source is now always passed in (plan/v1.yaml, `inputs`).
"""

from __future__ import annotations

# The 17 Universal POS tags. Used to catch a tagset mix-up at load time rather
# than as a silently zeroed score.
UPOS_TAGS = frozenset({
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
})

# Mapping from Penn Treebank POS tags (XPOS) to Universal POS tags (UPOS).
#
# NLTK's pos_tag() returns Penn Treebank tags while the gold standard and every
# other analyzer use Universal tags, so this table is what makes the comparison
# possible at all. That also makes it a scoring component: a gap in it is charged
# to the tagger, not to the conversion.
#
# Two omissions used to do exactly that. nltk.pos_tag emits the LITERAL tags "("
# and ")" for parentheses — not -LRB-/-RRB- — and neither was in the table, so
# both fell through to "X" against a gold "PUNCT": 182 tokens (0.73%) wrong before
# the tagger was even consulted. "-NONE-" (empty-category traces) was missing too.
#
# The remaining mismatches are genuine scheme differences that no flat table can
# express, because UPOS depends on context where PTB does not:
#   VB* -> AUX vs VERB   (1075 tokens) — "is" is AUX in UD, VBZ in PTB
#   IN  -> SCONJ vs ADP  (360)         — depends on whether it heads a clause
#   TO  -> ADP vs PART   (208)         — "to the shop" vs "to go"
#   RB  -> PART vs ADV   (200)         — "n't"/"not" are PART in UD
# _refine_upos() in wrappers/nltk.py handles these with lexical and neighbour-tag
# rules. It is a declared normalisation tier, reported separately from the flat
# table, so the conversion-induced ceiling stays visible instead of being folded
# into the score.
PTB_TO_UPOS = {
    "CC": "CCONJ", "CD": "NUM", "DT": "DET", "EX": "PRON", "FW": "X",
    "IN": "ADP", "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ",
    "LS": "X", "MD": "AUX", "NN": "NOUN", "NNS": "NOUN",
    "NNP": "PROPN", "NNPS": "PROPN", "PDT": "DET", "POS": "PART",
    "PRP": "PRON", "PRP$": "PRON", "RB": "ADV", "RBR": "ADV", "RBS": "ADV",
    "RP": "PART", "SYM": "SYM", "TO": "PART", "UH": "INTJ",
    "VB": "VERB", "VBD": "VERB", "VBG": "VERB", "VBN": "VERB",
    "VBP": "VERB", "VBZ": "VERB", "WDT": "DET", "WP": "PRON",
    "WP$": "PRON", "WRB": "ADV",
    # Punctuation and symbols. The bare "(" and ")" forms are what pos_tag
    # actually produces; the -LRB-/-RRB- forms appear in treebank files.
    ".": "PUNCT", ",": "PUNCT", ":": "PUNCT", "``": "PUNCT", "''": "PUNCT",
    "(": "PUNCT", ")": "PUNCT", "-LRB-": "PUNCT", "-RRB-": "PUNCT",
    "-LSB-": "PUNCT", "-RSB-": "PUNCT", "-LCB-": "PUNCT", "-RCB-": "PUNCT",
    "HYPH": "PUNCT", "NFP": "PUNCT", '"': "PUNCT",
    "#": "SYM", "$": "SYM",
    # Empty-category traces and other non-word tags.
    "-NONE-": "X", "ADD": "X", "GW": "X", "XX": "X", "NIL": "X",
    "AFX": "ADJ",
}
