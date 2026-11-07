"""Stanford Dependencies → Universal Dependencies label mapping.

Converts dependency relation labels from Stanford Dependencies (SD) and its
ClearNLP/OntoNotes descendants to Universal Dependencies v2 (UD). spaCy's
en_core_web_sm (spacy_sm, D3) emits SD-style labels while the EWT gold standard
uses UD, so without this conversion the comparison would measure
annotation-scheme compatibility rather than parser quality. Until 23 Oct 2026
SuPar's dependency model was mapped here as well.

The mapping follows the conversion described by the authors of both schemes:

  de Marneffe, M.-C., Dozat, T., Silveira, N., Haverinen, K., Ginter, F.,
  Nivre, J., & Manning, C. D. (2014). Universal Stanford Dependencies:
  A Cross-Linguistic Typology. In Proc. LREC'14, pp. 4585-4592.

  de Marneffe, M.-C., Manning, C. D., Nivre, J., & Zeman, D. (2021).
  Universal Dependencies. Computational Linguistics, 47(2), pp. 255-308.

Two maps, three tiers, reported separately
------------------------------------------
``CASING_MAP`` is pure notation: spaCy writes the root relation as ``ROOT`` where
UD writes ``root``. It affects 8.6% of tokens and is worth several points of LAS
on its own. ``SD_TO_UD`` is genuine relabelling (``dobj`` → ``obj`` and friends).
Lumping them together would credit a string-casing fix to the linguistic
conversion, so ``map_deprel`` takes a ``tier`` argument and the two are ablated
independently: ``none`` applies neither map, ``casing`` the first, ``full`` both
(``MAPPING_TIERS``).

What this cannot do
-------------------
Label renaming only. SD and UD also differ in *head attachment* — prepositions,
copulas, auxiliaries, coordination — and that affects roughly a quarter of tokens.
No relabelling fixes it; only full tree restructuring would, which is out of
scope. This is why cross-scheme comparisons are reported as conversion cost, and
why CLAS (which excludes exactly those function words) is the fairer metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, get_args

from syntactic_analysis.project import project_root

# Tier 1 — notation only. spaCy writes the root relation in uppercase.
CASING_MAP: dict[str, str] = {
    "ROOT": "root",
}

# Tier 2 — genuine relabelling.
# Sources: UD guidelines, CoreNLP conversion tables, empirical comparison.
SD_TO_UD: dict[str, str] = {
    # --- direct renames (1:1) ---
    "dobj": "obj",
    "nn": "compound",
    "prt": "compound:prt",
    "poss": "nmod:poss",
    "possessive": "case",
    "nsubjpass": "nsubj:pass",
    "csubjpass": "csubj:pass",
    "auxpass": "aux:pass",
    "num": "nummod",
    "number": "flat",
    "rcmod": "acl:relcl",
    "relcl": "acl:relcl",
    "preconj": "cc:preconj",
    "neg": "advmod",
    # UD 2.12 renamed the :tmod and :npmod subtypes to :unmarked. Targeting the
    # old names meant every spaCy `npadvmod` (200 tokens on the test set) was a
    # guaranteed LAS miss even when the attachment was right, because the target
    # label does not occur anywhere in the gold.
    "tmod": "obl:unmarked",
    "npadvmod": "obl:unmarked",
    "infmod": "acl",
    "partmod": "acl",
    "mwe": "fixed",
    "predet": "det:predet",
    "quantmod": "advmod",
    "dative": "iobj",
    "intj": "discourse",

    # --- prepositional constructions ---
    # In SD, "prep" marks the preposition as a dependent of the verb; in UD the
    # preposition becomes "case" under the noun. Only the label is changed here,
    # so these tokens usually still fail on the head check.
    "prep": "case",
    "pobj": "obl",
    "pcomp": "advcl",

    # --- copula and predicate complements ---
    "acomp": "xcomp",
    "oprd": "xcomp",
    # "attr" is deliberately NOT mapped. It marks the predicate nominal in an SD
    # copula construction, where the copula is the head; in UD the predicate
    # nominal is itself the root. Mapping it to "root" — as this table used to —
    # stamps the root label on a token whose head index is not 0 (248 of 248
    # occurrences on the test set), so it can never match gold, and it injects a
    # second "root" per sentence that corrupts any per-label analysis. This is a
    # structural difference, and no label-only mapping can express it.

    # --- agent ---
    "agent": "obl:agent",
}


# The three arms of the conversion, weakest first: no mapping, the string-case
# fix alone (ROOT -> root), the linguistic relabelling on top. They are named
# here, where they are implemented. Until 11 Oct 2026 three wrappers, the
# dependency stage, the statistics layer's reader and the figures each spelled
# the triple out for themselves. Until 12 Oct 2026 each of the three wrappers
# also checked the name for itself.
MapTier = Literal["none", "casing", "full"]
MAPPING_TIERS: tuple[MapTier, ...] = get_args(MapTier)

# The arm every headline number is scored under. The full conversion is the
# configuration every published figure for an SD parser is comparable to; the
# other two say how much of such a parser's score is the mapping rather than
# the parser.
HEADLINE_TIER: MapTier = "full"


def validate_tier(tier: str) -> MapTier:
    """Return `tier` if it names a mapping tier, and refuse it otherwise.

    The one check of a tier name: `map_deprel` runs it on every call, and the
    wrappers that take a `map_tier` run it in their constructor, so a mistyped
    tier fails where it is written rather than at the first sentence.

    Raises:
        ValueError: If `tier` is not one of MAPPING_TIERS.
    """
    if tier not in MAPPING_TIERS:
        raise ValueError(
            f"unknown map_tier {tier!r}; expected one of {list(MAPPING_TIERS)}"
        )
    return tier


def map_deprel(
    label: str,
    *,
    tier: MapTier = HEADLINE_TIER,
) -> str:
    """Convert one dependency relation label from SD to UD.

    Args:
        label: The parser's raw relation label.
        tier:  "none"   — return unchanged (the no-mapping ablation arm)
               "casing" — apply only ROOT → root
               "full"   — apply casing plus genuine relabelling (default)

    Labels that are already UD-compatible, or have no known mapping, are returned
    unchanged.

    Raises:
        ValueError: If `tier` is not one of MAPPING_TIERS. Until 12 Oct 2026 an
            unknown tier fell through to the full mapping without a word.
    """
    validate_tier(tier)
    if tier == "none":
        return label
    if label in CASING_MAP:
        return CASING_MAP[label]
    if tier == "casing":
        return label
    return SD_TO_UD.get(label, label)


@dataclass
class MappingReport:
    """Result of validating the mapping targets against a gold label inventory.

    Attributes:
        dead_targets:     Mapping targets that never occur in the gold. Must be
                          empty: a target absent from the gold is a guaranteed
                          scoring miss regardless of parser quality.
        covered_targets:  Targets that do occur in the gold.
        gold_only:        Gold labels no mapping produces (informational — most
                          are labels SD parsers simply cannot express).
    """

    dead_targets: set[str] = field(default_factory=set)
    covered_targets: set[str] = field(default_factory=set)
    gold_only: set[str] = field(default_factory=set)

    @property
    def ok(self) -> bool:
        """True when no mapping target is missing from the gold."""
        return not self.dead_targets


def validate_against_gold(gold_labels: Iterable[str]) -> MappingReport:
    """Check that every mapping target actually occurs in the gold inventory.

    Until version 1.0 the dependency stage ran this before it evaluated and
    printed the result: it turns a silent, permanent scoring penalty into a
    visible warning. It is exactly the check that would have caught the
    ``obl:tmod`` / ``obl:npmod`` targets, which UD renamed years before the
    gold snapshot in use here. The stages of version 1.0 do not run it; the
    out-of-repository test suite does, against the inventory below, for the
    mapping spacy_sm is scored with.
    """
    gold = set(gold_labels)
    targets = set(SD_TO_UD.values()) | set(CASING_MAP.values())
    return MappingReport(
        dead_targets=targets - gold,
        covered_targets=targets & gold,
        gold_only=gold - targets,
    )


def load_gold_inventory(path: str | Path | None = None) -> dict[str, int]:
    """Load the tracked gold DEPREL inventory as {label: count}.

    The inventory is committed (a few hundred bytes) so that the
    out-of-repository test suite, which validates the mapping, does not need
    the 18 MB corpus. `path` defaults
    to data/ewt_deprel_inventory.txt under the repository root.
    """
    if path is None:
        path = project_root() / "data" / "ewt_deprel_inventory.txt"
    counts: dict[str, int] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        count, _, label = line.partition("\t")
        if label:
            counts[label] = int(count)
    return counts
