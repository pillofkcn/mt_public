"""No downloads during a run: the switches scripts/run_pipeline.py sets for every stage.

A run uses the models and the data already on disk and nothing else. A model or
data package that is missing fails the configuration that needs it, and with it
its stage and the run; it is never fetched in the middle of a run from whatever
its server holds that day. setup.sh downloads everything a run loads. Since
24 Oct 2026 scripts/run_pipeline.py starts every stage with OFFLINE_SWITCHES in
its environment:

  HF_HUB_OFFLINE=1              huggingface_hub, and transformers through it,
                                send no request to the Hugging Face hub: a
                                transformer, its tokenizer and its config come
                                from the local cache (HF_HOME) or not at all.
                                huggingface_hub reads it when it is imported.
  SYNTACTIC_ANALYSIS_OFFLINE=1  this project's wrappers and loaders download
                                nothing (downloads_allowed() is False), and
                                refuse with NotCachedError instead:
                                  wrappers/stanza.py   no stanza.download(); the
                                      pipeline reads the files in Stanza's
                                      resources directory, and ELECTRA-large
                                      comes from the local hub cache only;
                                  wrappers/benepar.py  no benepar.download():
                                      benepar_en3 in nltk_data or nothing;
                                  wrappers/supar.py    nothing fetched into
                                      SuPar's cache (~/.cache/supar);
                                  NLTK data            no nltk.download() of
                                      treebank, the shipped tagger or punkt_tab
                                      (nltk_package below).

Outside a run neither is set: setup.sh, the training scripts and
scripts/smoke_plan.py download what is missing, as before. A stage script
started by hand downloads too; the provenance of every stage records the two
switches and whether downloads were allowed (stage.py, `offline`).

A run downloads nothing on its other load paths either: a spaCy pipeline is an
installed package (spacy.load never downloads), the models trained here are
read from model_cache/, and the datasets are local files. A model trained here
that is not there fails its configuration like any other missing model: since
24 Oct 2026 no build or run trains one (base.require_trained); its training
script does, before the lock. Until then the wrappers of nltk_ewt and nltk_arc,
and the stage train_udpipe for udpipe_ewt, trained a missing one from the EWT
train split during the run.
"""

from __future__ import annotations

import os
from types import MappingProxyType
from typing import Any, Mapping

__all__ = [
    "PROJECT_SWITCH", "OFFLINE_SWITCHES", "NotCachedError", "downloads_allowed",
    "switches", "record", "not_cached", "nltk_package",
]

# The project's own switch, and every switch a run sets with the value it sets.
PROJECT_SWITCH = "SYNTACTIC_ANALYSIS_OFFLINE"
OFFLINE_SWITCHES: Mapping[str, str] = MappingProxyType({
    "HF_HUB_OFFLINE": "1",
    PROJECT_SWITCH: "1",
})

# The values that turn a switch on, as huggingface_hub reads its own
# (constants._is_true): 1, ON, YES, TRUE in any case.
_ON = frozenset({"1", "ON", "YES", "TRUE"})


class NotCachedError(RuntimeError):
    """A model or data package a run needs is not on disk, and a run downloads nothing."""


def downloads_allowed() -> bool:
    """False when SYNTACTIC_ANALYSIS_OFFLINE is on (1, on, yes, true), as in every stage of a run."""
    value = os.environ.get(PROJECT_SWITCH)
    return value is None or value.strip().upper() not in _ON


def switches() -> dict[str, str | None]:
    """Each switch of OFFLINE_SWITCHES as this process's environment holds it; None where unset."""
    return {name: os.environ.get(name) for name in OFFLINE_SWITCHES}


def record() -> dict[str, Any]:
    """What a stage's provenance records: the switches, and whether downloads were allowed."""
    return {"switches": switches(), "downloads_allowed": downloads_allowed()}


def not_cached(what: str, where: str) -> NotCachedError:
    """The error for `what`, missing from `where` during a run; one line, no path of this machine."""
    return NotCachedError(
        f"{what} is not in {where}, and a run downloads nothing "
        f"({PROJECT_SWITCH}={os.environ.get(PROJECT_SWITCH)}); setup.sh downloads it"
    )


def nltk_package(package: str, resource: str) -> None:
    """Make sure the NLTK data package `package` is there, found as `resource` (corpora/treebank).

    Outside a run: nltk.download(package, quiet=True), as every caller did
    until 24 Oct 2026. NLTK's downloader then reads its index from the network
    and downloads the package when it is missing or the index lists a newer
    one. During a run: nltk.data.find(resource), and nothing is downloaded.

    Raises:
        NotCachedError: During a run, if the package is not in nltk_data.
    """
    import nltk

    if downloads_allowed():
        nltk.download(package, quiet=True)
        return
    try:
        nltk.data.find(resource)
    except LookupError as exc:
        raise not_cached(f"the NLTK data package {package!r}", "nltk_data") from exc
