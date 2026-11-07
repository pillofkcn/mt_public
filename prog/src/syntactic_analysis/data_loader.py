"""Data loaders — CoNLL-U files, and raw text split into sentences.

This module provides functions for loading input data:

  Raw text:
    Split into sentences and words by NLTK (split_raw_text), for a wrapper
    given a string rather than tokens. No measuring stage does that.

  CoNLL-U (.conllu):
    Standard format from Universal Dependencies for annotated corpora.
    Used as gold-standard data for computing evaluation metrics.
    Each token occupies one line with 10 tab-separated columns:
        ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC
    Sentences are separated by blank lines.
    Lines starting with '#' are metadata comments.

Key characteristics:
  - load_conllu() returns a DependencyParse with analyzer_name="gold"
  - All indices in the returned data are 1-based (following CoNLL-U convention)
  - Multi-word tokens (e.g. ID="1-2") and empty nodes (e.g. ID="1.1") are skipped
  - every sentence of an annotated file carries its document (Sentence.doc_id):
    the last "# newdoc id", else document_of() its sent_id
"""

from __future__ import annotations

import re
from pathlib import Path

from syntactic_analysis.models import DependencyParse, Sentence, Token


def split_raw_text(text: str) -> list[list[str]]:
    """Split raw text into sentences, and each sentence into words.

    NLTK's Punkt sentence splitter, then its word tokenizer: the splitting the
    `nltk` and `rdrpos` wrappers apply to raw text, which each wrapper did for
    itself until 12 Oct 2026. No measuring stage comes through here - every one
    of them passes the gold tokens - so this decides no reported number.

    NLTK is imported inside the function, so that importing this module needs
    no NLP library. Its data package punkt_tab is downloaded when it is
    missing, except during a run (offline.nltk_package).
    """
    from syntactic_analysis.offline import nltk_package

    nltk_package("punkt_tab", "tokenizers/punkt_tab")
    from nltk.tokenize import sent_tokenize, word_tokenize

    return [word_tokenize(sentence) for sentence in sent_tokenize(text)]


def parse_conllu(
    text: str,
    *,
    analyzer_name: str = "gold",
    predicted: bool = False,
) -> DependencyParse:
    """Parse CoNLL-U text into a DependencyParse object.

    The one CoNLL-U reader of the package: `load_conllu` reads an annotated
    file through it, `wrappers/udpipe.py` reads its tool's output through it,
    and the training rows of `wrappers/nltk_transition.py` are a projection of
    what it returns. Until 12 Oct 2026 each of those two wrappers had a reader
    of its own.

    Args:
        text:          CoNLL-U text holding any number of sentences.
        analyzer_name: The name the returned parse carries; "gold" marks
                       manually annotated data.
        predicted:     False for an annotated file, True for a tool's output.
                       The two are read alike except in two points. An
                       annotated file is taken as written: Sentence.text comes
                       from "# text = ...", Sentence.sent_id from
                       "# sent_id = ...", Sentence.doc_id from the last
                       "# newdoc id = ..." (else `document_of` the sent_id),
                       and LEMMA, UPOS and DEPREL are kept even where they are
                       "_". In a tool's output no comment line is read, so
                       Sentence.text is the forms joined by one space, sent_id
                       and doc_id are "", and a "_" in LEMMA, UPOS or DEPREL -
                       a field the tool did not fill - becomes "".

    Returns a DependencyParse with one Sentence per sentence of the text; empty
    text gives a parse with no sentences.

    The CoNLL-U format has 10 tab-separated columns per token line:
        [0] ID      — token index (1-based), e.g. "1", "2", "3"
        [1] FORM    — the word as it appears in the text, e.g. "The", "cat"
        [2] LEMMA   — base/dictionary form, e.g. "the", "cat"
        [3] UPOS    — Universal POS tag, e.g. "DET", "NOUN"
        [4] XPOS    — language-specific POS tag, e.g. "DT", "NN"
        [5] FEATS   — morphological features (not used in this framework)
        [6] HEAD    — index of the head word (0 = root of the sentence)
        [7] DEPREL  — dependency relation label, e.g. "det", "nsubj"
        [8] DEPS    — enhanced dependencies (not used in this framework)
        [9] MISC    — miscellaneous annotations (not used in this framework)

    Special line types:
        "# text = ..." — original sentence text (metadata comment)
        "# sent_id = ..." — corpus sentence id, kept on Sentence.sent_id
        "# newdoc id = ..." — starts a document; its id is Sentence.doc_id
                          of this and every following sentence up to the
                          next newdoc
        Other "#" lines — comments, silently skipped
        Blank lines     — sentence boundaries
        IDs with "-"    — multi-word tokens (e.g. "1-2" for "don't"), skipped
        IDs with "."    — empty nodes (e.g. "1.1"), skipped
    """
    sentences: list[Sentence] = []
    current_tokens: list[Token] = []  # tokens accumulated for the current sentence
    current_text = ""                 # original text from "# text = ..." metadata
    current_id = ""                   # id from "# sent_id = ..." metadata
    current_doc = ""                  # id from the last "# newdoc id = ..."; kept
                                      # across sentences until the next newdoc

    for line in text.splitlines():
        line = line.strip()

        if predicted and line.startswith("#"):      # a tool's comments are not read
            continue

        if line.startswith("# text ="):            # metadata: original sentence text
            current_text = line[len("# text ="):].strip()

        elif line.startswith("# sent_id ="):         # metadata: corpus sentence id
            current_id = line[len("# sent_id ="):].strip()

        elif line == "# newdoc" or line.startswith("# newdoc "):  # a document starts
            # "# newdoc id = X", or a bare "# newdoc": a document without an
            # id, whose sentences fall back to their sent_id prefix.
            current_doc = line.partition("=")[2].strip()

        elif line.startswith("#"):                  # other comments — skip
            continue

        elif line == "":                            # blank line = sentence boundary
            if current_tokens:
                if predicted:                       # no "# text" was read: rebuild it
                    current_text = " ".join(token.text for token in current_tokens)
                sentences.append(Sentence(
                    text=current_text, tokens=current_tokens, sent_id=current_id,
                    doc_id=current_doc or document_of(current_id),
                ))
                current_tokens = []
                current_text = ""
                current_id = ""

        else:                                       # token line — 10 tab-separated columns
            fields = line.split("\t")
            if len(fields) < 8:                     # malformed line — skip
                continue

            if "-" in fields[0] or "." in fields[0]:  # skip multi-word tokens and empty nodes
                continue

            lemma, upos, deprel = fields[2], fields[3], fields[7]
            if predicted:                           # "_" = a field the tool did not fill
                lemma = "" if lemma == "_" else lemma
                upos = "" if upos == "_" else upos
                deprel = "" if deprel == "_" else deprel

            current_tokens.append(
                Token(
                    text=fields[1],                                          # FORM
                    index=int(fields[0]),                                     # ID
                    lemma=lemma,                                              # LEMMA
                    pos=upos,                                                # UPOS
                    xpos=fields[4] if fields[4] != "_" else "",              # XPOS ("_" = empty)
                    head_index=int(fields[6]) if fields[6] != "_" else -1,   # HEAD
                    dep_rel=deprel,                                          # DEPREL
                )
            )

    if current_tokens:  # last sentence if the text doesn't end with a blank line
        if predicted:
            current_text = " ".join(token.text for token in current_tokens)
        sentences.append(Sentence(
            text=current_text, tokens=current_tokens, sent_id=current_id,
            doc_id=current_doc or document_of(current_id),
        ))

    return DependencyParse(sentences=sentences, analyzer_name=analyzer_name)


def load_conllu(path: str | Path) -> DependencyParse:
    """Load a CoNLL-U file into a DependencyParse object.

    This function parses gold-standard (manually annotated) data used
    for computing evaluation metrics (POS accuracy, UAS, LAS).

    Args:
        path: Path to the .conllu file (absolute or relative).

    Returns a DependencyParse with analyzer_name="gold" containing all
    sentences and their fully annotated tokens. The columns and the special
    line types are described at `parse_conllu`, which does the reading.
    """
    return parse_conllu(Path(path).read_text(encoding="utf-8"))


# The sentence number an EWT sent_id ends in: "<document>-0001".
_SENTENCE_NUMBER = re.compile(r"-[0-9]+$")


def document_of(sent_id: str) -> str:
    """The document of a sentence from its id alone: the id without its trailing "-<digits>".

    "weblog-blogspot.com_zentelligence_20040423000200_ENG_20040423_000200-0002"
    -> "weblog-blogspot.com_zentelligence_20040423000200_ENG_20040423_000200".
    The fallback `parse_conllu` takes for a sentence that no "# newdoc id"
    precedes. On the EWT test split it assigns every sentence the document its
    newdoc names (316 documents either way). An id without such a suffix is
    returned whole, so its sentence is a document of its own; "" stays "".
    """
    return _SENTENCE_NUMBER.sub("", sent_id)
