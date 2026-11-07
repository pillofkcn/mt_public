"""A spaCy Doc as this project's tokens and parses, and a spaCy pipeline as a row.

Two wrappers run a spaCy pipeline: `wrappers/spacy.py`, and `wrappers/benepar.py`,
whose POS tags and dependency arcs are spaCy's. Both turn a processed `Doc` into
`Token`s the same way - 1-based indices, the root's self-pointing head mapped to
0, the relation label passed through one SD->UD mapping tier. Until 12 Oct 2026
each wrapper carried its own copy of both conversions.

Both also describe the pipeline the same way: what it was trained on
(`PIPELINE_CORPUS`), whether it runs a transformer and so which level its tags
and arcs sit at (`pipeline_level`), and its name and version
(`pipeline_name`). Until 16 Oct 2026 each wrapper declared these for itself,
and analyzers.csv carried two corpus strings for one model. Since 24 Oct 2026
both also declare the files of an installed pipeline the same way, for the
lock (`package_files`), and since 25 Oct 2026 the data files spaCy's own
import reads (`library_files`).

At the top level of the package, beside `tagsets.py`, for the same reason that
module is: a wrapper imports shared code from here and never from another
wrapper, so removing one wrapper module takes nothing shared away from another
(dokumentacia.txt §9.2); no wrapper imports another (`tagsets.py`). Nothing
here imports spaCy: a `Doc`, a `Span`, a token and a loaded pipeline are read
by attribute.

Building the `Doc` is NOT shared, on purpose. The spaCy wrapper leaves sentence
starts unpinned and the Benepar wrapper pins them (`_analyze_pretokenized` in
the first, `_pretokenized_doc` in the second), and that difference is a
measured property of the two rows.
"""

from __future__ import annotations

import importlib.util
import json
import sysconfig
from pathlib import Path

from syntactic_analysis.base import MethodLevel
from syntactic_analysis.deprel_map import MapTier, map_deprel
from syntactic_analysis.downloads import DownloadedFiles, library_data
from syntactic_analysis.models import DependencyParse, Sentence, Token

# What an English spaCy pipeline (en_core_web_*) is trained on, for its tags and
# its arcs alike.
PIPELINE_CORPUS = "OntoNotes 5 (LDC2013T19), ClearNLP dependency conversion"

# The component names spaCy gives a transformer encoder.
_TRANSFORMER_PIPES = frozenset({"transformer", "curated_transformer"})


def is_transformer_pipeline(nlp) -> bool:
    """Whether a loaded pipeline runs a transformer encoder.

    Read off the pipeline's component names rather than off the model name (a
    "_trf" in it): a transformer pipeline loaded from a local path or under
    another name would be misreported by the name.
    """
    return bool(_TRANSFORMER_PIPES & set(nlp.pipe_names))


def pipeline_level(nlp) -> MethodLevel:
    """The level of a pipeline's tags and arcs: L4 with a transformer, L2 without.

    en_core_web_sm is a MultiHashEmbed + CNN tok2vec feeding a tagger and a
    transition-based parser, with no pretrained vectors and no recurrent
    encoder: L2 for both. `nlp` None - nothing loaded yet - reads as that model.
    """
    if nlp is not None and is_transformer_pipeline(nlp):
        return MethodLevel.NEURAL_PRETRAINED
    return MethodLevel.NEURAL_SHALLOW


def pipeline_name(nlp) -> str:
    """A loaded pipeline as its own meta.json names it: `en_core_web_sm-3.8.0`."""
    meta = nlp.meta
    return f"{meta.get('lang', '')}_{meta.get('name', '')}-{meta.get('version', '')}"


def package_files(name: str) -> DownloadedFiles:
    """The files spacy.load(name) reads of an installed pipeline package, under site_packages.

    As spaCy finds them (spacy.util.load_model_from_init_py, which the
    package's __init__.py calls): the package's meta.json, which names the
    data directory, and that directory, <lang>_<name>-<version>/, read as a
    whole (config.cfg, vocab/, the tokenizer and every component's files).
    The package is found by importlib without importing it; its code is not
    a model file. A package that is not installed, or has no meta.json, is
    declared by its meta.json in the interpreter's site-packages, so a lock
    check reports it missing. Since 24 Oct 2026.
    """
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        site = Path(sysconfig.get_paths()["purelib"])
        return DownloadedFiles.under("site_packages", site, [site / name / "meta.json"])
    package = Path(next(iter(spec.submodule_search_locations)))
    if not (package / "meta.json").is_file():
        return DownloadedFiles.under("site_packages", package.parent, [package / "meta.json"])
    meta = json.loads((package / "meta.json").read_text(encoding="utf-8"))
    data = f"{meta['lang']}_{meta['name']}-{meta['version']}"
    return DownloadedFiles.under(
        "site_packages", package.parent, [package / "meta.json"], [package / data],
    )


def library_files() -> tuple[DownloadedFiles, ...]:
    """The data files `import spacy` reads from spaCy's and thinc's package directories.

    spaCy's default_config.cfg, with whose defaults spacy.load fills every
    pipeline's config, the yml of training recommendations its cli module
    reads, and the source of thinc's GPU kernels (downloads.LIBRARY_DATA),
    under site_packages. Every load of a spaCy pipeline imports spacy, so
    both wrappers that run one declare them, the EWT pipeline trained here
    too. Since 25 Oct 2026, before the lock.
    """
    return library_data("spacy", "thinc")


def doc_tokens(span, *, tier: MapTier, start: int = 0) -> list[Token]:
    """The tokens of a `Doc` or of one sentence span, indexed from 1.

    Args:
        span:  A processed spaCy `Doc`, or a sentence span of one.
        tier:  The SD->UD mapping tier applied to every relation label.
        start: Document index of the first token of `span`: 0 for a whole
               `Doc`, `sent.start` for a sentence, so that indices and heads
               come out sentence-relative.
    """
    return [
        Token(
            text=tok.text,
            index=tok.i - start + 1,
            pos=tok.pos_,
            xpos=tok.tag_,
            lemma=tok.lemma_,
            dep_rel=map_deprel(tok.dep_, tier=tier),
            # A root points to itself in spaCy; 0 is the UD convention.
            head_index=(tok.head.i - start + 1) if tok.head != tok else 0,
        )
        for tok in span
    ]


def doc_to_depparse(doc, *, tier: MapTier, name: str) -> DependencyParse:
    """A processed `Doc` as one `Sentence` per sentence spaCy segmented."""
    return DependencyParse(
        sentences=[
            Sentence(text=sent.text, tokens=doc_tokens(sent, tier=tier, start=sent.start))
            for sent in doc.sents
        ],
        analyzer_name=name,
    )
