"""File digests, for verifying a file that was downloaded rather than shipped.

Four callers wanted the SHA-256 of a file and three of them had written it out
again: the two wrappers that downloaded their own model and verified it before
trusting it (`wrappers/rdrpos.py`, `wrappers/udpipe.py`), the gold loader, and
`wrappers/nltk_transition.py`, which truncated the result for provenance. They
all call this now. Until 12 Oct 2026 the two wrappers that downloaded a model
each wrote the download out as well; download_verified is that routine, once.
No wrapper calls download_verified any more (UDPipe's published model went with
version 1.0, RDRPOSTagger's on 24 Oct 2026): scripts/fetch_data.py calls it for
the confirmatory test sets, and scripts/train_rdrpos.py for the RDRPOSTagger
learner.

A wrapper must not import from `evaluation/`, which is why the shared copy sits
at the top level of the package. `evaluation/ptb.py` keeps its own
`corpus_digest`, and that is not a duplicate of this: it hashes the text of
several treebank files, read through NLTK and concatenated in order, where
`sha256_of` hashes the raw bytes of one file, so there is nothing here it
could reuse.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

# 1 MiB. Chunked reading keeps memory flat regardless of file size (tens of MB
# for the NLTK transition parser's model files and the UDPipe model, a few
# hundred KB for an RDRPOSTagger file, but nothing here assumes that stays true).
_CHUNK = 1 << 20

# per connection; a stalled mirror should fail loudly, not hang forever
DOWNLOAD_TIMEOUT_S = 60


def sha256_of(path: str | Path) -> str:
    """The SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ChecksumMismatch(RuntimeError):
    """A downloaded file whose SHA-256 is not the recorded one.

    Attributes:
        expected: The recorded digest.
        actual:   The digest of what was downloaded.
    """

    def __init__(self, expected: str, actual: str) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"expected sha256={expected}, got {actual}")


def download_verified(
    url: str,
    path: Path,
    *,
    sha256: str | None,
    timeout: float = DOWNLOAD_TIMEOUT_S,
) -> None:
    """Download `url` to `path` atomically, verifying it against a recorded digest.

    The transfer goes to `<path>.part` and is `os.replace`d into place only
    after it is complete and, where a digest is recorded, verified. Without
    this, a Ctrl-C or a dropped connection leaves a truncated file at `path`
    whose mere existence makes every later run skip the download and fail at
    load time instead.

    Prints nothing and words no advice: the caller announces the download and
    turns a failure into its own message.

    Args:
        url:     Where the file is fetched from.
        path:    Final destination. Its directory must exist.
        sha256:  The recorded hex digest, or None for a file that has none to
                 be checked against.
        timeout: Seconds per connection.

    Raises:
        ChecksumMismatch: If the downloaded file's digest is not `sha256`.
            `path` is then left absent.
        urllib.error.URLError, OSError: A network or file error, unchanged.
    """
    part = path.with_name(path.name + ".part")
    try:
        # urlopen rather than urlretrieve: only the former takes a timeout, and
        # a stalled connection must fail rather than hang forever.
        with urllib.request.urlopen(url, timeout=timeout) as response:
            with open(part, "wb") as handle:
                shutil.copyfileobj(response, handle)

        if sha256 is not None:
            actual = sha256_of(part)
            if actual != sha256:
                raise ChecksumMismatch(sha256, actual)

        # Atomic on POSIX and Windows alike (os.replace uses MoveFileEx there),
        # so a reader never observes a partially written file at `path`.
        os.replace(part, path)
    finally:
        # Any failure path - network error, checksum mismatch, Ctrl-C - leaves
        # `path` absent; removing the partial file lets a retry start fresh
        # instead of accumulating stale .part files.
        if part.exists():
            part.unlink()
