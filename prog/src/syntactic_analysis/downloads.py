"""The downloaded files a configuration's load reads, and the directories they lie under.

plan/LOCK.json records the SHA-256 of every model file a run loads (since
24 Oct 2026, before the lock): under `models` the five models this project
trains, by their path under prog/ (base.TrainedModel), and under
`downloaded_models` every other file a configuration's load reads that was
downloaded rather than trained here - the Stanza models, ELECTRA-large,
benepar_en3, SuPar's crf-con-en, en_core_web_sm, the tagger NLTK ships and
the treebank sample the PCFG is induced from (seal.py). Each wrapper declares
them (BaseAnalyzer.downloaded_files) as DownloadedFiles: a root, and the files
under it by their path relative to the root. The lock records the root's name
and the relative paths, never a path of this machine; the root is found again
on the machine that checks the lock, as the library that reads the files finds
it.

The roots (ROOTS), and how each is found:

  stanza_resources  Stanza's resources directory: stanza.resources.common.
                    DEFAULT_MODEL_DIR, which is $STANZA_RESOURCES_DIR, or else
                    Stanza's user cache for its resources version (Windows:
                    %LOCALAPPDATA%/StanfordNLP/stanza/Cache/1.11.0/resources).
                    wrappers/stanza.py passes no directory to the pipeline, so
                    the pipeline reads this one.
  hf_hub_cache      the Hugging Face hub cache: huggingface_hub.constants.
                    HF_HUB_CACHE, which is $HF_HUB_CACHE, or $HF_HOME/hub, or
                    ~/.cache/huggingface/hub - where transformers finds
                    google/electra-large-discriminator for Stanza.
  nltk_data         the entry of nltk.data.path in which NLTK finds the
                    package ($NLTK_DATA first, then NLTK's defaults; on this
                    machine %APPDATA%/nltk_data): the tagger NLTK ships, the
                    treebank sample, benepar_en3. Each file is declared under
                    the entry NLTK found it in.
  supar_cache       ~/.cache/supar, the directory supar.utils.fn.download
                    builds with os.path.expanduser and keeps its models in.
  site_packages     the directory that holds an installed Python package: the
                    parent of the package's directory as importlib finds it
                    (en_core_web_sm, which spacy.load reads, and the data
                    files of LIBRARY_DATA).

The roots of the five models trained here are not among them: those files
are under prog/ (model_cache/), recorded under `models`.

A library also reads data files from its own package directory when it is
imported: Stanza's import reads the emoji package's emoji.json, spaCy's its
default_config.cfg and a yml of training recommendations, thinc's the source
of its GPU kernels, NLTK's its VERSION and scikit-learn's, which NLTK
imports, two style sheets (LIBRARY_DATA). They are not code (.py and
compiled modules, whose version the lock records under `versions`), so since
25 Oct 2026, before the lock, a wrapper whose load imports such a library
declares them too (library_data), under site_packages, and every
configuration declares those of NLTK and scikit-learn, which every stage
imports before it builds a configuration (build.STAGE_LIBRARIES): a run of
every (task, configuration) pair on one sentence, with every file it opened
recorded, found them opened and not declared.

No model library is imported when this module is imported; each root is found
by importing its library when it is asked for. Since 24 Oct 2026.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Iterable, Mapping

__all__ = [
    "ROOTS", "LIBRARY_DATA", "DownloadedFiles", "stanza_resources_dir", "hf_hub_cache",
    "hf_snapshot", "nltk_data_files", "nltk_data_absent", "nltk_data_directory", "supar_cache",
    "library_data",
]

# Each root's name, as LOCK.json records it, and how it is found (see above).
ROOTS: Mapping[str, str] = MappingProxyType({
    "stanza_resources": "stanza.resources.common.DEFAULT_MODEL_DIR "
                        "($STANZA_RESOURCES_DIR or Stanza's user cache)",
    "hf_hub_cache": "huggingface_hub.constants.HF_HUB_CACHE "
                    "($HF_HUB_CACHE, $HF_HOME/hub or ~/.cache/huggingface/hub)",
    "nltk_data": "the entry of nltk.data.path in which NLTK finds the package",
    "supar_cache": "~/.cache/supar (supar.utils.fn.download)",
    "site_packages": "the directory holding the installed package (importlib)",
})

# The data files each library reads from its own package directory when it is
# imported, by their path relative to that directory (25 Oct 2026). Each path
# is a literal in the library's code, cited here; none of the libraries
# exports it without being imported, and the declaration imports no library.
# A file a later release moves or drops is declared all the same, so a lock
# check reports it missing instead of passing over it.
LIBRARY_DATA: Mapping[str, tuple[str, ...]] = MappingProxyType({
    # emoji/unicode_codes/__init__.py: _load_default_from_json() at import.
    # stanza/pipeline/core.py imports stanza/pipeline/langid_processor.py,
    # which imports emoji.
    "emoji": ("unicode_codes/emoji.json",),
    # spacy/language.py: DEFAULT_CONFIG = util.load_config(DEFAULT_CONFIG_PATH),
    # the defaults spacy.load fills a pipeline's config with; and
    # spacy/cli/init_config.py: RECOMMENDATIONS = srsly.read_yaml(ROOT /
    # "quickstart_training_recommendations.yml"), read when spacy/__init__.py
    # imports spacy.cli. Both at import.
    "spacy": ("default_config.cfg", "cli/templates/quickstart_training_recommendations.yml"),
    # thinc/backends/_custom_kernels.py: KERNELS_SRC = (PWD /
    # "_custom_kernels.cu").read_text(...) at import (spaCy imports thinc).
    "thinc": ("backends/_custom_kernels.cu",),
    # nltk/__init__.py: version_file = os.path.join(os.path.dirname(__file__),
    # "VERSION"), read into nltk.__version__ at import.
    "nltk": ("VERSION",),
    # sklearn/utils/_repr_html/estimator.py: _CSS_STYLE = _get_css_style(),
    # which reads Path(__file__).parent / "estimator.css" and "params.css" at
    # import; nltk/classify/scikitlearn.py imports scikit-learn when nltk is
    # imported.
    "sklearn": ("utils/_repr_html/estimator.css", "utils/_repr_html/params.css"),
})


def _relative_posix(path: str) -> str:
    """`path` if it is a relative POSIX path inside its root; ValueError otherwise."""
    pure = PurePosixPath(path)
    if (not path or "\\" in path or pure.is_absolute() or ".." in pure.parts
            or ":" in pure.parts[0] or pure.as_posix() != path.rstrip("/")):
        raise ValueError(
            f"{path!r} is not a relative POSIX path inside its root, as LOCK.json records one"
        )
    return pure.as_posix()


@dataclass(frozen=True)
class DownloadedFiles:
    """Files one load reads under one root, by their path relative to it.

    Attributes:
        root:        A name of ROOTS: what LOCK.json records.
        base:        Where that root is on this machine, as its library finds
                     it: where the files are read.
        files:       Files the load reads, relative POSIX paths.
        directories: Directories the load reads as a whole (a spaCy pipeline
                     package, benepar_en3, a transformer's snapshot at its
                     revision), relative POSIX paths: every file under one, at
                     any depth, is a file of the load. A library that picks
                     among the files of such a directory (transformers between
                     tokenizer.json and vocab.txt, or pytorch_model.bin and a
                     model.safetensors added later) can read another file when
                     one is added, so each one counts.
    """

    root: str
    base: Path
    files: tuple[str, ...] = ()
    directories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.root not in ROOTS:
            raise ValueError(f"unknown root {self.root!r}; expected one of {list(ROOTS)}")
        object.__setattr__(self, "base", Path(self.base))
        object.__setattr__(self, "files", tuple(_relative_posix(f) for f in self.files))
        object.__setattr__(
            self, "directories", tuple(_relative_posix(d) for d in self.directories),
        )

    @classmethod
    def under(
        cls, root: str, base: str | Path, files: Iterable[str | Path] = (),
        directories: Iterable[str | Path] = (),
    ) -> "DownloadedFiles":
        """The declaration of absolute `files` and `directories`, all under `base`.

        Raises:
            ValueError: If a path is not under `base`.
        """
        base = Path(os.path.abspath(base))

        def relative(path: str | Path) -> str:
            try:
                return Path(os.path.abspath(path)).relative_to(base).as_posix()
            except ValueError:
                raise ValueError(
                    f"{path} is not under the {root} directory {base}, so LOCK.json "
                    f"cannot record it"
                ) from None

        return cls(
            root, base, tuple(relative(f) for f in files),
            tuple(relative(d) for d in directories),
        )

    def paths(self) -> dict[str, Path]:
        """Each declared file whether it is there or not, and every file now under a declared directory."""
        found = {rel: self.base / rel for rel in self.files}
        for directory in self.directories:
            folder = self.base / directory
            if folder.is_dir():
                for item in sorted(folder.rglob("*")):
                    if item.is_file():
                        found[item.relative_to(self.base).as_posix()] = item
        return dict(sorted(found.items()))

    def missing(self) -> list[str]:
        """The declared files that are not on disk, and the declared directories missing or empty ("dir/")."""
        absent = [rel for rel in self.files if not (self.base / rel).is_file()]
        for directory in self.directories:
            folder = self.base / directory
            if not folder.is_dir() or not any(item.is_file() for item in folder.rglob("*")):
                absent.append(f"{directory}/")
        return absent


# --- the roots --------------------------------------------------------------------


def stanza_resources_dir() -> Path:
    """Stanza's resources directory, where a pipeline built without `dir` reads its models."""
    from stanza.resources.common import DEFAULT_MODEL_DIR

    return Path(DEFAULT_MODEL_DIR)


def hf_hub_cache() -> Path:
    """The Hugging Face hub cache transformers reads a model from."""
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def hf_snapshot(repo_id: str, revision: str) -> DownloadedFiles:
    """The snapshot of a hub model at one revision, and refs/main, which points to it.

    The folder is named by huggingface_hub itself (repo_folder_name: models--
    <owner>--<name>), and the snapshot by the full revision: a directory of
    the hub cache. refs/main is declared as a file too. Stanza loads the
    transformer by its name, with no revision (stanza/models/common/
    bert_embedding.py, AutoModel.from_pretrained), so the hub cache follows
    refs/main to the snapshot it reads; the file holds that revision. A
    refs/main moved to another snapshot after the lock would make the load
    read files the lock does not record, and the check refuses it before the
    load, not only after it as the revision check of wrappers/stanza.py does.
    """
    from huggingface_hub.file_download import repo_folder_name

    folder = repo_folder_name(repo_id=repo_id, repo_type="model")
    return DownloadedFiles("hf_hub_cache", hf_hub_cache(),
                           files=(f"{folder}/refs/main",),
                           directories=(f"{folder}/snapshots/{revision}",))


def _nltk_entry(path: Path) -> Path:
    """The entry of nltk.data.path that holds `path`.

    Raises:
        ValueError: If no entry holds it.
    """
    import nltk.data

    absolute = Path(os.path.abspath(path))
    for entry in nltk.data.path:
        base = Path(os.path.abspath(entry))
        try:
            absolute.relative_to(base)
        except ValueError:
            continue
        return base
    raise ValueError(f"{path} is in no directory of nltk.data.path")


def nltk_data_files(paths: Iterable[str | Path]) -> tuple[DownloadedFiles, ...]:
    """Files NLTK found, each declared under the entry of nltk.data.path that holds it.

    `paths` are absolute, as NLTK resolves them (nltk.data.find, a corpus
    reader's abspath); a package NLTK reads from its zip file is the zip.
    One DownloadedFiles per entry, in the order of nltk.data.path.
    """
    grouped: dict[Path, list[Path]] = {}
    for path in paths:
        path = Path(path)
        grouped.setdefault(_nltk_entry(path), []).append(path)
    return tuple(
        DownloadedFiles.under("nltk_data", base, files) for base, files in grouped.items()
    )


def nltk_data_absent(files: Iterable[str] = (), directories: Iterable[str] = ()) -> DownloadedFiles:
    """The declaration of NLTK data that NLTK does not find: under the first entry of nltk.data.path.

    Where NLTK would look first, so that a lock check reports the files
    missing (DownloadedFiles.missing) rather than the declaration failing.
    """
    import nltk.data

    return DownloadedFiles(
        "nltk_data", Path(os.path.abspath(nltk.data.path[0])), tuple(files), tuple(directories),
    )


def nltk_data_directory(resource: str) -> DownloadedFiles:
    """A package of nltk_data read as a whole directory (benepar_en3), where nltk.data.find finds it.

    A package NLTK does not find is declared as a directory where it would
    look first (nltk_data_absent), so a lock check reports it missing.
    """
    import nltk.data

    try:
        pointer = nltk.data.find(resource)
    except LookupError:
        return nltk_data_absent(directories=(resource,))
    directory = Path(pointer.path)
    return DownloadedFiles.under("nltk_data", _nltk_entry(directory), directories=(directory,))


def supar_cache() -> Path:
    """SuPar's model cache, built as supar.utils.fn.download builds it."""
    return Path(os.path.expanduser("~/.cache/supar"))


def library_data(*packages: str) -> tuple[DownloadedFiles, ...]:
    """The data files of LIBRARY_DATA that the libraries `packages` read when imported.

    One DownloadedFiles per library, under site_packages: its package
    directory found by importlib without importing it (as
    spacy_doc.package_files finds a pipeline package), the files relative
    to the directory that holds it. A library that is not installed is
    declared under the interpreter's site-packages, so a lock check reports
    its files missing. Since 25 Oct 2026, before the lock.

    Raises:
        KeyError: If a package is not in LIBRARY_DATA.
    """
    import importlib.util
    import sysconfig

    declared = []
    for name in packages:
        relative = LIBRARY_DATA[name]
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.submodule_search_locations:
            package = Path(sysconfig.get_paths()["purelib"]) / name
        else:
            package = Path(next(iter(spec.submodule_search_locations)))
        files = [package / rel for rel in relative]
        declared.append(DownloadedFiles.under("site_packages", package.parent, files))
    return tuple(declared)
