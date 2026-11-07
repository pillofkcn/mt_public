"""Console helpers the scripts share, and the banner of a training script.

Replaces the `log`/`out` pair that was copy-pasted into all four experiment
scripts, with a `LINE_WIDTH` that had drifted to 80 in three of them and 90 in the
fourth.

`quiet_third_party` suppresses warnings from the NLP libraries but never from
SciPy. A blanket `warnings.filterwarnings("ignore")` at the top of the statistics
script silenced exactly the diagnostics that would have revealed the tie problem —
"Exact p-value calculation does not work if there are zeros" — through six full
experiment runs.

Until 23 Oct 2026 this module also formatted the tables of the stage and
statistics reports before version 1.0: p-values, effect sizes, differences in
points, the level x domain grid, the Wilcoxon, McNemar, Friedman and Cochran's
Q lines. Version 1.0 writes its tables in scripts/analyze.py, so those
formatters went with the scripts that printed them; what is left is what a
script of version 1.0 calls.
"""

from __future__ import annotations

import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping, TextIO

LINE_WIDTH = 90

# Libraries whose import-time and inference-time chatter is noise. SciPy is
# deliberately absent: its warnings are about the statistics being computed.
_NOISY_MODULES = ("stanza", "supar", "benepar", "torch", "spacy", "transformers", "nltk")


@contextmanager
def quiet_third_party():
    """Silence NLP-library warnings for the duration of a block.

    Scoped by module and category rather than globally, so nothing about the
    statistics is hidden.
    """
    with warnings.catch_warnings():
        for module in _NOISY_MODULES:
            for category in (UserWarning, FutureWarning, DeprecationWarning):
                warnings.filterwarnings("ignore", category=category, module=module)
        yield


def utf8_console(*, stderr: bool = False) -> None:
    """Make stdout - and stderr on request - UTF-8, replacing what cannot be encoded.

    Training corpora and platform strings contain em dashes; a Windows console
    in a legacy code page raises UnicodeEncodeError on the first table row
    rather than at the end, losing the whole run. line_buffering keeps progress
    visible when the run is piped to a file, which for a full-corpus run is the
    difference between watching it and guessing whether it hung.

    Called first thing in a script's main() rather than at import, so that
    importing a script changes nothing about the importer's own stdout.
    `stderr=True` is for run_pipeline.py, which echoes every stage's output and
    so inherits every stage's characters on both streams. A stream without
    `reconfigure` is left as it is. Until 12 Oct 2026 the three measuring stages
    and the pipeline each carried these lines.
    """
    for stream in (sys.stdout, sys.stderr) if stderr else (sys.stdout,):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


class Reporter:
    """Writes a run log to a file and echoes it to the console.

    Used as a context manager so the file is closed even when an experiment
    aborts, which matters because a previous script wrote its final results block
    outside the file's `with` — so one quoted statistic ended up in no log at all.
    The training scripts print their banner with it (`header`).
    """

    def __init__(
        self,
        log_path: str | Path | None = None,
        *,
        width: int = LINE_WIDTH,
        echo: bool = True,
    ) -> None:
        self.log_path = Path(log_path) if log_path else None
        self.width = width
        self.echo = echo
        self._handle: TextIO | None = None

    def __enter__(self) -> "Reporter":
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(self.log_path, "w", encoding="utf-8", newline="\n")
        return self

    def __exit__(self, *exc_info) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    # --- primitives ---

    def out(self, message: str = "") -> None:
        """Write to both the log file and the console."""
        if self.echo:
            print(message)
        if self._handle is not None:
            self._handle.write(message + "\n")

    def rule(self, char: str = "-") -> None:
        self.out(char * self.width)

    def header(self, title: str, meta: Mapping[str, object] | None = None) -> None:
        """A section banner, optionally with a key-value block underneath."""
        self.rule("=")
        self.out(f"  {title}")
        if meta:
            width = max(len(str(k)) for k in meta)
            for key, value in meta.items():
                self.out(f"  {str(key):<{width}} : {value}")
        self.rule("=")
