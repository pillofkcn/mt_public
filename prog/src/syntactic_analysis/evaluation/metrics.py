"""Evaluation metrics — UPOS accuracy, LAS, UAS and CLAS from per-sentence counts.

Version 1.0 records COUNTS per sentence and computes one corpus (micro) figure
from their sums: `PosCounts` and `DepCounts` here, `evalb.ConCounts` for
constituency, and `corpus_metric` for all three, in points (0-100), the unit
the plan uses. The dependency counts follow the CoNLL 2018 evaluation script
(conll18_ud_eval.py): relation subtypes are ignored on both sides, and CLAS is
that script's F1 over gold and predicted content words.

Alignment is strict
-------------------
Predicted and gold must have the same number of tokens, and each token the
same form; otherwise the counts raise. Pairing by a plain ``zip()`` made the
denominator ``min(len(pred), len(gold))`` instead of ``len(gold)`` — so an
analyzer that emitted *fewer* tokens was rewarded, because the hard tail of the
sentence simply dropped out of the denominator. A tool returning 3 of 5 tokens,
all correct, scored 1.0 rather than 0.6. The base-class contract normally
catches a mismatch first; this is the second line of defence.

Constituency
------------
Not here. Bracket counts and the EVALB parameter sets (``COLLINS``,
``COLLINS_NML``) have been ``evalb.py`` since 9 Oct 2026: which labels are
deleted and which are treated as equal is itself a parameter of that comparison.

Until 23 Oct 2026 this module also held the metric functions of the stages
before version 1.0, which compared whole parses and returned ratios
(pos_accuracy, uas, las, clas and their ``*_counts``), the confusion counts of
a tagger and the per-relation counts of a parser. They went with the stage
scripts and figures that called them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Iterable

from syntactic_analysis.evaluation.evalb import ConCounts
from syntactic_analysis.models import Sentence

# --- CoNLL-2018 relation sets (conll18_ud_eval.py) ---------------------------
# Subtypes are matched on the part before ":", so obl:unmarked counts as obl.

CONTENT_DEPRELS = frozenset({
    "nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp", "obl", "vocative",
    "expl", "dislocated", "advcl", "advmod", "discourse", "nmod", "appos",
    "nummod", "acl", "amod", "conj", "fixed", "flat", "compound", "list",
    "parataxis", "orphan", "goeswith", "reparandum", "root", "dep",
})

FUNCTIONAL_DEPRELS = frozenset({
    "aux", "cop", "mark", "det", "clf", "case", "cc",
})


def base_deprel(label: str) -> str:
    """Strip a relation subtype: 'obl:unmarked' -> 'obl'.

    The CoNLL-2018 convention for grouping relations. Public since 12 Oct 2026:
    the dependency stage kept a copy of it for its per-relation table.
    """
    return label.split(":", 1)[0]


def _token_pairs(predicted: Sentence, gold: Sentence, where: str, *, same_forms: bool = False):
    """The (pred_token, gold_token) pairs of one sentence; raises on a length mismatch.

    `where` names the sentence in the error. With `same_forms`, as the version
    1.0 counts call it, a token whose form differs from the gold token at the
    same position is refused too: conll18_ud_eval.py refuses such a pair of
    files rather than score it. Until 23 Oct 2026 the metric functions of the
    earlier stages paired their tokens here without it.
    """
    if len(predicted.tokens) != len(gold.tokens):
        raise ValueError(
            f"token count mismatch in {where}: predicted has "
            f"{len(predicted.tokens)}, gold has {len(gold.tokens)}"
        )
    if same_forms:
        for position, (pred_tok, gold_tok) in enumerate(zip(predicted.tokens, gold.tokens), start=1):
            if pred_tok.text != gold_tok.text:
                raise ValueError(
                    f"token form mismatch in {where}: token {position} is "
                    f"{pred_tok.text!r} in the prediction, {gold_tok.text!r} in the gold"
                )
    return zip(predicted.tokens, gold.tokens)


# --- version 1.0: per-sentence counts and the corpus metric -----------------
# A stage records one counts object per sentence; corpus_metric sums them and
# divides once. A bootstrap replicate is then a sum of the same counts over
# resampled units, computed by the same function as the point estimate.


@dataclass(frozen=True)
class PosCounts:
    """One sentence's UPOS counts (or a sum of them).

    Attributes:
        correct: Tokens whose UPOS equals the gold UPOS.
        total:   Gold tokens.
    """

    correct: int
    total: int

    def __post_init__(self) -> None:
        """Refuse counts no sentence can produce."""
        if self.correct < 0 or self.correct > self.total:
            raise ValueError(f"impossible POS counts: {self}")


@dataclass(frozen=True)
class DepCounts:
    """One sentence's attachment counts, CoNLL 2018 conventions (or a sum of them).

    Attributes:
        las_correct:     Head and base relation both right.
        uas_correct:     Head right.
        total:           Gold tokens (punctuation included).
        clas_correct:    Gold content words with head and base relation right.
        clas_total:      Gold content words: base relation in CONTENT_DEPRELS.
        clas_pred_total: Predicted content words, by the predicted base
                         relation. Not in the brief's list of columns, and
                         needed: the official CLAS is an F1, precision over
                         these and recall over `clas_total`, and with gold
                         tokens the two differ whenever a parser labels a
                         function word as a content word or the reverse.
    """

    las_correct: int
    uas_correct: int
    total: int
    clas_correct: int
    clas_total: int
    clas_pred_total: int

    def __post_init__(self) -> None:
        """Refuse counts no sentence can produce."""
        if any(getattr(self, f.name) < 0 for f in fields(self)):
            raise ValueError(f"negative dependency count: {self}")
        if not (
            self.clas_correct <= self.las_correct <= self.uas_correct <= self.total
            and self.clas_correct <= min(self.clas_total, self.clas_pred_total)
            and max(self.clas_total, self.clas_pred_total) <= self.total
        ):
            raise ValueError(f"impossible dependency counts: {self}")


def _filled(value: str) -> str:
    """A CoNLL-U field as conll18_ud_eval.py compares it: empty is "_".

    `parse_conllu` turns a "_" in a tool's UPOS or DEPREL into "" and keeps
    the gold's "_" as written; the official script compares "_" with "_" and
    counts a match.
    """
    return value or "_"


def pos_sentence_counts(predicted: Sentence, gold: Sentence) -> PosCounts:
    """UPOS counts of one sentence: the plan's `upos_accuracy`.

    UPOS is compared as a string, an empty value and "_" being the same one.

    Raises:
        ValueError: If the two sentences differ in token count or in a form.
    """
    correct = total = 0
    for pred_tok, gold_tok in _token_pairs(predicted, gold, _name_of(gold), same_forms=True):
        total += 1
        correct += _filled(pred_tok.pos) == _filled(gold_tok.pos)
    return PosCounts(correct=correct, total=total)


def dep_sentence_counts(predicted: Sentence, gold: Sentence) -> DepCounts:
    """Attachment counts of one sentence, as conll18_ud_eval.py counts them.

    The rules, each as the official script applies it with gold tokenisation
    (every word aligned to the word at the same position):

      * a relation is compared by its base, `base_deprel`, on BOTH sides:
        ``obl:tmod`` predicted for ``obl`` is right, and so is ``obl`` for
        ``obl:tmod`` (the script cuts each DEPREL at the first ":" on reading);
      * punctuation is a token like any other, in UAS and LAS;
      * head 0 is the root, and a predicted root is right where the gold word
        is the root; a head the parser left undetermined (-1) is wrong;
      * a word is a content word when its base relation is in
        CONTENT_DEPRELS, the script's set (``root`` and ``dep`` are in it,
        ``punct`` and the seven FUNCTIONAL_DEPRELS are not); a gold content
        word counts towards CLAS when its head and base relation are right,
        and the predicted content words are counted from the predicted
        relations alone;
      * an empty relation and "_" are the same value (`_filled`).

    Raises:
        ValueError: If the two sentences differ in token count or in a form.
    """
    las = uas = total = clas_correct = clas_total = clas_pred_total = 0
    for pred_tok, gold_tok in _token_pairs(predicted, gold, _name_of(gold), same_forms=True):
        gold_rel = base_deprel(_filled(gold_tok.dep_rel))
        pred_rel = base_deprel(_filled(pred_tok.dep_rel))
        head_ok = pred_tok.head_index == gold_tok.head_index
        labelled_ok = head_ok and pred_rel == gold_rel
        gold_content = gold_rel in CONTENT_DEPRELS
        total += 1
        uas += head_ok
        las += labelled_ok
        clas_total += gold_content
        clas_pred_total += pred_rel in CONTENT_DEPRELS
        clas_correct += gold_content and labelled_ok
    return DepCounts(
        las_correct=las, uas_correct=uas, total=total,
        clas_correct=clas_correct, clas_total=clas_total,
        clas_pred_total=clas_pred_total,
    )


def _name_of(gold: Sentence) -> str:
    """How an alignment error names a sentence: its id, where it has one."""
    return f"sentence {gold.sent_id}" if gold.sent_id else "the sentence"


# Which metric each count type yields, the primary one first. The names are
# the plan's (tasks.<task>.metric.name, descriptive.d3.metrics); "uas" is the
# control figure the brief lists beside them.
_CORPUS_METRICS: dict[type, tuple[str, ...]] = {
    PosCounts: ("upos_accuracy",),
    DepCounts: ("las", "uas", "clas"),
    ConCounts: ("bracket_f1",),
}


def corpus_metric(
    counts: Iterable[PosCounts | DepCounts | ConCounts], metric: str | None = None
) -> float:
    """The corpus (micro) metric of per-sentence counts, in points (0-100).

    Sums the counts field by field and divides once, so a long sentence
    weighs by its tokens or brackets. The count type says which task it is:

      * PosCounts - ``upos_accuracy``: correct / total.
      * DepCounts - ``las`` (default), ``uas``: correct / total; ``clas``:
        the F1 of conll18_ud_eval.py, 2 * clas_correct / (clas_total +
        clas_pred_total), which is 2PR / (P + R) with P over the predicted
        and R over the gold content words.
      * ConCounts - ``bracket_f1``: F1 = 2PR / (P + R) with P = matched /
        test and R = matched / gold, as the EVALB program computes it from
        its totals. 0 when nothing matched; error sentences are already out
        of the counts.

    A metric whose denominator is 0 is NaN - not measured, which is not a
    score of zero.

    The figure equals the EVALB program's and conll18_ud_eval.py's for
    non-empty totals. At the edges they differ: with brackets on one side and
    none matched, this gives 0 where EVALB prints nan; with empty POS or
    dependency totals, NaN where the script reports 0.

    Args:
        counts: Per-sentence counts of one type, in any number (a sum of
            counts is a counts object too, so pre-summed units work as well).
        metric: Which metric of the type; None for its primary one.

    Raises:
        ValueError: If `counts` is empty, or `metric` is not one of the type's.
        TypeError: If the counts are not all of one of the three types.
    """
    counts = list(counts)
    if not counts:
        raise ValueError("no counts to pool")
    kind = type(counts[0])
    if kind not in _CORPUS_METRICS:
        raise TypeError(
            f"unknown counts type {kind.__name__}; expected one of "
            f"{[k.__name__ for k in _CORPUS_METRICS]}"
        )
    strangers = {type(c).__name__ for c in counts if type(c) is not kind}
    if strangers:
        raise TypeError(f"counts of one type only: {kind.__name__} mixed with {sorted(strangers)}")
    names = _CORPUS_METRICS[kind]
    name = names[0] if metric is None else metric
    if name not in names:
        raise ValueError(
            f"unknown metric {name!r} for {kind.__name__}; expected one of {list(names)}"
        )

    pooled = kind(**{f.name: sum(getattr(c, f.name) for c in counts) for f in fields(kind)})
    if kind is PosCounts:
        return _points(pooled.correct, pooled.total)
    if kind is DepCounts:
        if name == "las":
            return _points(pooled.las_correct, pooled.total)
        if name == "uas":
            return _points(pooled.uas_correct, pooled.total)
        return _points(2 * pooled.clas_correct, pooled.clas_total + pooled.clas_pred_total)
    if pooled.gold + pooled.test == 0:
        return math.nan
    precision = pooled.matched / pooled.test if pooled.test else 0.0
    recall = pooled.matched / pooled.gold if pooled.gold else 0.0
    if precision + recall == 0:
        return 0.0
    return 100 * 2 * precision * recall / (precision + recall)


def _points(numerator: int, denominator: int) -> float:
    """A ratio in points; NaN for an empty denominator."""
    return 100 * numerator / denominator if denominator else math.nan
